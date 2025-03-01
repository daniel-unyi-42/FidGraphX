import yaml
import os
import numpy as np
import networkx as nx
import torch
from BAMotifsDataset import BAMotifsDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected, to_networkx
from GNN import GNN
from Explainer import Predictor, Selector, apply_mask, random_mask
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter


# load config yaml
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# set random seed
seed = config['seed']
torch.manual_seed(seed)
np.random.seed(seed)

# set data path and log path
data_path = config['data_path']
if not os.path.exists(data_path):
    os.makedirs(data_path)
log_path = config['log_path']
if not os.path.exists(log_path):
    os.makedirs(log_path)

# initialize logger
run_id = len(os.listdir(log_path))
log_path = os.path.join(log_path, f'run_{run_id}')
writer = SummaryWriter(log_path)
with open(f'{log_path}/config.yaml', 'w') as f:
    yaml.dump(config, f)

# load dataset
if config['dataset'] == 'BAMotifs':
    dataset = BAMotifsDataset(data_path, num_graphs=500, ba_nodes=25, attach_prob=0.1)

dataset = dataset.shuffle()
print(f'Dataset {config["dataset"]} loaded, number of graphs: {len(dataset)}')

# split dataset
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset,
    [config['train_size'], config['val_size'], config['test_size']],
)

# create data loaders
train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_set, batch_size=len(val_set))
test_loader = DataLoader(test_set, batch_size=len(test_set))

num_node_features = dataset[0].x.shape[1]
num_edge_features = 0 if dataset[0].edge_attr is None else dataset[0].edge_attr.shape[1]
num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else 1 # regression

# define baseline model
baseline = GNN(
    conv_type=config['conv_type'],
    task_type=config['task_type'],
    learning_rate=config['learning_rate'],
    in_channels=num_node_features,
    num_layers=config['num_layers'],
    hidden_channels=config['hidden_channels'],
    out_channels=num_classes,
    edge_dim=num_edge_features,
    use_norm=config['use_norm'],
)

print('Baseline model:\n', baseline)

if config['task_type'] == 'classification':
    y_pred, y_true = baseline.predict_batch(test_loader)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{log_path}/baseline_cm_initial.png')
    plt.clf()

# train baseline model
if config['train_baseline']:
    best_val_acc = float('inf') if config['task_type'] == 'regression' else 0
    for epoch in range(config['epochs']):
        train_loss, train_acc = baseline.train_batch(train_loader)
        val_loss, val_acc = baseline.test_batch(val_loader)
        if (config['task_type'] == 'classification' and best_val_acc < val_acc) or \
           (config['task_type'] == 'regression' and best_val_acc > val_acc):
            best_val_acc = val_acc
            print(f'Best validation accuracy updated: {best_val_acc:.4f}, saving baseline...')
            torch.save(baseline.state_dict(), os.path.join(log_path, 'baseline.pt'))
        if config['print_results']:
            print(f'Epoch: {epoch}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}')
            print(f'Epoch: {epoch}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')
        writer.add_scalar('BASELINE/train_loss', train_loss, epoch)
        writer.add_scalar('BASELINE/train_acc', train_acc, epoch)
        writer.add_scalar('BASELINE/val_loss', val_loss, epoch)
        writer.add_scalar('BASELINE/val_acc', val_acc, epoch)

        if config['task_type'] == 'classification':
            # see accuracy if ground truth was used
            metrics = []
            cms = []
            for data in val_loader:
                data = data.to(baseline.device)
                data = apply_mask(data, data.true.unsqueeze(1))
                logits = baseline(data)
                metric = baseline.metric(logits, data.y)
                metrics.append(metric.item())
                cm = confusion_matrix(data.y.cpu().numpy(), logits.argmax(dim=1).cpu().numpy())
                cms.append(cm)
            cms = sum(cms)
            if epoch % 20 == 0:
                sns.heatmap(cms, annot=True, fmt='d')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(f'{log_path}/baseline_cm_true_{epoch}.png')
                plt.clf()
            writer.add_scalar('BASELINE/baseline_val_acc_true', sum(metrics) / len(metrics), epoch)
            # see accuracy if random mask was used
            metrics = []
            cms = []
            for data in val_loader:
                data = data.to(baseline.device)
                data = apply_mask(data, random_mask(data))
                logits = baseline(data)
                metric = baseline.metric(logits, data.y)
                metrics.append(metric.item())
                cm = confusion_matrix(data.y.cpu().numpy(), logits.argmax(dim=1).cpu().numpy())
                cms.append(cm)
            cms = sum(cms)
            if epoch % 20 == 0:
                sns.heatmap(cms, annot=True, fmt='d')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(f'{log_path}/baseline_cm_random_{epoch}.png')
                plt.clf()
            writer.add_scalar('BASELINE/baseline_val_acc_random', sum(metrics) / len(metrics), epoch)
            # save tsne plot
            if epoch % 20 == 0:
                # save tsne plot if true mask was used
                for data in val_loader:
                    data = data.to(baseline.device)
                    data = apply_mask(data, data.true.unsqueeze(1))
                    logits = baseline(data)
                    break
                tsne = TSNE(n_components=2).fit_transform(logits.cpu().detach().numpy())
                plt.scatter(tsne[:, 0], tsne[:, 1], c=data.y.cpu().numpy())
                plt.savefig(f'{log_path}/baseline_tsne_true_{epoch}.png')
                plt.clf()
                # save tsne plot if random mask was used
                for data in val_loader:
                    data = data.to(baseline.device)
                    data = apply_mask(data, random_mask(data))
                    logits = baseline(data)
                    break
                tsne = TSNE(n_components=2).fit_transform(logits.cpu().detach().numpy())
                plt.scatter(tsne[:, 0], tsne[:, 1], c=data.y.cpu().numpy())
                plt.savefig(f'{log_path}/baseline_tsne_random_{epoch}.png')
                plt.clf()

baseline.load_state_dict(torch.load(os.path.join(log_path, 'baseline.pt'), weights_only=False))
# baseline.load_state_dict(torch.load("logs/run_0/baseline.pt", weights_only=False))
val_loss, val_acc = baseline.test_batch(val_loader)
print(f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')
test_loss, test_acc = baseline.test_batch(test_loader)
print(f'Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}')

# plot confusion matrix
if config['task_type'] == 'classification':
    y_preds, y_trues = baseline.predict_batch(test_loader)
    cm = confusion_matrix(y_trues, y_preds)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{log_path}/baseline_cm.png')
    plt.clf()

# plot regression plot
if config['task_type'] == 'regression':
    y_preds, y_trues = baseline.predict_batch(test_loader)
    plt.scatter(y_trues, y_preds)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.savefig(f'{log_path}/baseline_scatter.png')
    plt.clf()


# define predictor model
predictor = Predictor(
    baseline = baseline,
    conv_type=config['conv_type'],
    task_type=config['task_type'],
    learning_rate=config['learning_rate'],
    in_channels=num_node_features,
    num_layers=config['num_layers'],
    hidden_channels=config['hidden_channels'],
    out_channels=num_classes,
    edge_dim=num_edge_features,
    use_norm=config['use_norm'],
)

print('Predictor model:\n', predictor)

if config['task_type'] == 'classification':
    y_pred, y_true = predictor.predict_batch(test_loader)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{log_path}/predictor_cm_initial.png')
    plt.clf()

# train predictor model
if config['train_predictor']:
    best_val_acc = float('inf') if config['task_type'] == 'regression' else 0
    for epoch in range(config['epochs']):
        train_loss, train_acc = predictor.train_batch(train_loader)
        val_loss, val_acc = predictor.test_batch(val_loader)
        if (config['task_type'] == 'classification' and best_val_acc < val_acc) or \
              (config['task_type'] == 'regression' and best_val_acc > val_acc):
            best_val_acc = val_acc
            print(f'Best validation accuracy updated: {best_val_acc:.4f}, saving predictor...')
            torch.save(predictor.state_dict(), os.path.join(log_path, 'predictor.pt'))
        if config['print_results']:
            print(f'Epoch: {epoch}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}')
            print(f'Epoch: {epoch}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')
        writer.add_scalar('PREDICTOR/train_loss', train_loss, epoch)
        writer.add_scalar('PREDICTOR/train_acc', train_acc, epoch)
        writer.add_scalar('PREDICTOR/val_loss', val_loss, epoch)
        writer.add_scalar('PREDICTOR/val_acc', val_acc, epoch)

        if config['task_type'] == 'classification':
            # see accuracy if true mask was used: this should be as high as possible
            metrics = []
            cms = []
            for data in val_loader:
                data = data.to(predictor.device)
                logits = predictor(data, data.true.unsqueeze(1))
                metric = predictor.metric(logits, data.y)
                metrics.append(metric.item())
                cm = confusion_matrix(data.y.cpu().numpy(), logits.argmax(dim=1).cpu().numpy())
                cms.append(cm)
            cms = sum(cms)
            if epoch % 20 == 0:
                sns.heatmap(cms, annot=True, fmt='d')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(f'{log_path}/predictor_cm_true_{epoch}.png')
                plt.clf()
            writer.add_scalar('PREDICTOR/predictor_val_acc_true', sum(metrics) / len(metrics), epoch)
            # see accuracy if random mask was used
            metrics = []
            cms = []
            for data in val_loader:
                data = data.to(predictor.device)
                logits = predictor(data, random_mask(data))
                metric = predictor.metric(logits, data.y)
                metrics.append(metric.item())
                cm = confusion_matrix(data.y.cpu().numpy(), logits.argmax(dim=1).cpu().numpy())
                cms.append(cm)
            cms = sum(cms)
            if epoch % 20 == 0:
                sns.heatmap(cms, annot=True, fmt='d')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(f'{log_path}/predictor_cm_random_{epoch}.png')
                plt.clf()
            writer.add_scalar('PREDICTOR/predictor_val_acc_random', sum(metrics) / len(metrics), epoch)
            # save tsne plot
            if epoch % 20 == 0:
                # save tsne plot if true mask was used: this should be as clustered as possible
                for data in val_loader:
                    data = data.to(predictor.device)
                    logits = predictor(data, data.true.unsqueeze(1))
                    break
                tsne = TSNE(n_components=2).fit_transform(logits.cpu().detach().numpy())
                plt.scatter(tsne[:, 0], tsne[:, 1], c=data.y.cpu().numpy())
                plt.savefig(f'{log_path}/predictor_tsne_true_{epoch}.png')
                plt.clf()
                # save tsne plot if random mask was used
                for data in val_loader:
                    data = data.to(predictor.device)
                    logits = predictor(data, random_mask(data))
                    break
                tsne = TSNE(n_components=2).fit_transform(logits.cpu().detach().numpy())
                plt.scatter(tsne[:, 0], tsne[:, 1], c=data.y.cpu().numpy())
                plt.savefig(f'{log_path}/predictor_tsne_random_{epoch}.png')
                plt.clf()

predictor.load_state_dict(torch.load(os.path.join(log_path, 'predictor.pt'), weights_only=False))
val_loss, val_acc = predictor.test_batch(val_loader)
print(f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')
test_loss, test_acc = predictor.test_batch(test_loader)
print(f'Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}')

# plot confusion matrix
if config['task_type'] == 'classification':
    y_preds, y_trues = predictor.predict_batch(test_loader)
    cm = confusion_matrix(y_trues, y_preds)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{log_path}/predictor_cm.png')
    plt.clf()

# plot regression plot
if config['task_type'] == 'regression':
    y_preds, y_trues = predictor.predict_batch(test_loader)
    plt.scatter(y_trues, y_preds)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.savefig(f'{log_path}/predictor_scatter.png')
    plt.clf()
