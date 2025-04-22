import yaml
import os
import time
import numpy as np
import networkx as nx
import torch
from BAMotifs import BAMotifs
from BAMotifsVolumeDataset import BAMotifsVolumeDataset
from AlkaneCarbonylDataset import AlkaneCarbonylDataset
from BenzeneDataset import BenzeneDataset
from FluorideCarbonylDataset import FluorideCarbonylDataset
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from GNN import GNN
from Explainer import Selector, apply_mask
from sklearn.metrics import confusion_matrix
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
data_path = f"{config['data_path']}/{config['dataset']}"
os.makedirs(data_path, exist_ok=True)
log_path = f"{config['log_path']}/final/{config['dataset']}"
os.makedirs(log_path, exist_ok=True)


# initialize logger
run_id = str(int(time.time()))
log_path = os.path.join(log_path, f'run_{run_id}')
writer = SummaryWriter(log_path)
with open(f'{log_path}/config.yaml', 'w') as f:
    yaml.dump(config, f)

# load dataset
if config['dataset'] == 'BAMotifs':
    dataset = BAMotifs(data_path, num_graphs=500, attach_prob=0.2)

elif config['dataset'] == 'BAMotifsVolume':
    dataset = BAMotifsVolumeDataset(data_path, num_graphs=500, ba_nodes=25, attach_prob=0.1)
elif config['dataset'] == 'AlkaneCarbonyl':
    dataset = AlkaneCarbonylDataset(data_path)
elif config['dataset'] == 'Benzene':
    dataset = BenzeneDataset(data_path)
elif config['dataset'] == 'FluorideCarbonyl':
    dataset = FluorideCarbonylDataset(data_path)
elif config['dataset'] == 'MNIST':
    class SuperPixelTransform(object):
        def __call__(self, data):
            data = T.ToUndirected()(data)
            data.x = torch.cat([data.x, data.pos], dim=1)
            data.edge_attr = data.edge_attr.unsqueeze(-1)
            data.true =  torch.ones_like(data.x[:, 0])
            return data
    dataset = GNNBenchmarkDataset(
        data_path,
        config['dataset'],
        pre_transform=SuperPixelTransform(),
    )
else:
    raise ValueError(f"Dataset {config['dataset']} not supported")

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

# for data in train_loader:
#     print(data.true.sum().item() / data.num_nodes)

num_node_features = dataset[0].x.shape[1]
num_edge_features = 0 if dataset[0].edge_attr is None else dataset[0].edge_attr.shape[1]
num_classes = dataset.num_classes if config['task_type'] == 'classification' else 1

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

# baseline.load_state_dict(torch.load(os.path.join(log_path, 'baseline.pt'), weights_only=False))
baseline.load_state_dict(torch.load("logs/final/BAMotifs/run_1744564105/baseline.pt", weights_only=False))
val_loss, val_acc = baseline.test_batch(val_loader)
print(f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')
test_loss, test_acc = baseline.test_batch(test_loader)
print(f'Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}')

# plot confusion matrix
if config['task_type'] == 'classification':
    y_preds, y_trues = baseline.predict_batch(test_loader)
    y_preds = [pred.argmax() for pred in y_preds]
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

# define predictor for selected subgraph
pos_predictor = GNN(
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

# define predictor for unselected subgraph
neg_predictor = GNN(
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

# define selector
selector = Selector(
    baseline=baseline,
    pos_predictor=pos_predictor,
    neg_predictor=neg_predictor,
    sparsity=config['sparsity'],
    reward_coeff=config['reward_coeff'],
)

print('Selector model:\n', selector)

if config['train_selector']:
    best_val_fidelity_diff = 0
    train_pred = True
    train_sel = False
    for epoch in range(config['epochs']):
        train_pred = not train_pred
        train_sel = not train_sel
        train_loss, train_sparsity, train_pos_loss, train_neg_loss, train_pos_metric, train_neg_metric, train_fid_plus_probs, train_fid_minus_probs, train_fid_plus_acc, train_fid_minus_acc, train_auc, train_f1, train_iou = selector.train_batch(train_loader, train_pred=True, train_sel=True)
        val_loss, val_sparsity, val_pos_loss, val_neg_loss, val_pos_metric, val_neg_metric, val_fid_plus_probs, val_fid_minus_probs, val_fid_plus_acc, val_fid_minus_acc, val_auc, val_f1, val_iou = selector.test_batch(val_loader)
        val_fidelity_diff = val_fid_plus_probs - val_fid_minus_probs
        if val_fidelity_diff > best_val_fidelity_diff and val_sparsity < config['sparsity']:
            best_val_fidelity_diff = val_fidelity_diff
            print(f'Best validation fidelity difference updated: {best_val_fidelity_diff:.4f}, saving selector...')
            torch.save(selector.state_dict(), os.path.join(log_path, 'selector.pt'))
        if config['print_results']:
            print(f'Epoch: {epoch}, Train loss: {train_loss:.4f}, Train sparsity: {train_sparsity:.4f}, Train pos loss: {train_pos_loss:.4f}, Train neg loss: {train_neg_loss:.4f}, Train pos metric: {train_pos_metric:.4f}, Train neg metric: {train_neg_metric:.4f}, Train fid plus probs: {train_fid_plus_probs:.4f}, Train fid minus probs: {train_fid_minus_probs:.4f}, Train fid plus acc: {train_fid_plus_acc:.4f}, Train fid minus acc: {train_fid_minus_acc:.4f}, Train auc: {train_auc:.4f}, Train f1: {train_f1:.4f}, Train iou: {train_iou:.4f}')
            print(f'Epoch: {epoch}, Val loss: {val_loss:.4f}, Val sparsity: {val_sparsity:.4f}, Val pos loss: {val_pos_loss:.4f}, Val neg loss: {val_neg_loss:.4f}, Val pos metric: {val_pos_metric:.4f}, Val neg metric: {val_neg_metric:.4f}, Val fid plus probs: {val_fid_plus_probs:.4f}, Val fid minus probs: {val_fid_minus_probs:.4f}, Val fid plus acc: {val_fid_plus_acc:.4f}, Val fid minus acc: {val_fid_minus_acc:.4f}, Val auc: {val_auc:.4f}, Val f1: {val_f1:.4f}, Val iou: {val_iou:.4f}')
        writer.add_scalar('SELECTOR/train_loss', train_loss, epoch)
        writer.add_scalar('SELECTOR/train_sparsity', train_sparsity, epoch)
        writer.add_scalar('SELECTOR/train_pos_loss', train_pos_loss, epoch)
        writer.add_scalar('SELECTOR/train_neg_loss', train_neg_loss, epoch)
        writer.add_scalar('SELECTOR/train_pos_metric', train_pos_metric, epoch)
        writer.add_scalar('SELECTOR/train_neg_metric', train_neg_metric, epoch)
        writer.add_scalar('SELECTOR/train_fid_plus_probs', train_fid_plus_probs, epoch)
        writer.add_scalar('SELECTOR/train_fid_minus_probs', train_fid_minus_probs, epoch)
        writer.add_scalar('SELECTOR/train_fid_plus_acc', train_fid_plus_acc, epoch)
        writer.add_scalar('SELECTOR/train_fid_minus_acc', train_fid_minus_acc, epoch)
        writer.add_scalar('SELECTOR/train_auc', train_auc, epoch)
        writer.add_scalar('SELECTOR/train_f1', train_f1, epoch)
        writer.add_scalar('SELECTOR/train_iou', train_iou, epoch)
        writer.add_scalar('SELECTOR/val_loss', val_loss, epoch)
        writer.add_scalar('SELECTOR/val_sparsity', val_sparsity, epoch)
        writer.add_scalar('SELECTOR/val_pos_loss', val_pos_loss, epoch)
        writer.add_scalar('SELECTOR/val_neg_loss', val_neg_loss, epoch)
        writer.add_scalar('SELECTOR/val_pos_metric', val_pos_metric, epoch)
        writer.add_scalar('SELECTOR/val_neg_metric', val_neg_metric, epoch)
        writer.add_scalar('SELECTOR/val_fid_plus_probs', val_fid_plus_probs, epoch)
        writer.add_scalar('SELECTOR/val_fid_minus_probs', val_fid_minus_probs, epoch)
        writer.add_scalar('SELECTOR/val_fid_plus_acc', val_fid_plus_acc, epoch)
        writer.add_scalar('SELECTOR/val_fid_minus_acc', val_fid_minus_acc, epoch)
        writer.add_scalar('SELECTOR/val_auc', val_auc, epoch)
        writer.add_scalar('SELECTOR/val_f1', val_f1, epoch)
        writer.add_scalar('SELECTOR/val_iou', val_iou, epoch)

selector.load_state_dict(torch.load(os.path.join(log_path, 'selector.pt'), weights_only=False))
# selector.load_state_dict(torch.load("logs/final/BAMotifs/run_1744557584//selector.pt", weights_only=False))
for sparsity in [0.1, 0.2, 0.3, 0.4, 0.5]:
    val_loss, val_sparsity, val_pos_loss, val_neg_loss, val_pos_metric, val_neg_metric, val_fid_plus_probs, val_fid_minus_probs, val_fid_plus_acc, val_fid_minus_acc, val_auc, val_f1, val_iou = selector.test_batch(val_loader)
    test_loss, test_sparsity, test_pos_loss, test_neg_loss, test_pos_metric, test_neg_metric, test_fid_plus_probs, test_fid_minus_probs, test_fid_plus_acc, test_fid_minus_acc, test_auc, test_f1, test_iou = selector.test_batch(test_loader)
    print(f'Val loss: {val_loss:.4f}, Val sparsity: {val_sparsity:.4f}, Val pos loss: {val_pos_loss:.4f}, Val neg loss: {val_neg_loss:.4f}, Val pos metric: {val_pos_metric:.4f}, Val neg metric: {val_neg_metric:.4f}, Val fid plus probs: {val_fid_plus_probs:.4f}, Val fid minus probs: {val_fid_minus_probs:.4f}, Val fid plus acc: {val_fid_plus_acc:.4f}, Val fid minus acc: {val_fid_minus_acc:.4f}, Val auc: {val_auc:.4f}, Val f1: {val_f1:.4f}, Val iou: {val_iou:.4f}')
    print(f'Test loss: {test_loss:.4f}, Test sparsity: {test_sparsity:.4f}, Test pos loss: {test_pos_loss:.4f}, Test neg loss: {test_neg_loss:.4f}, Test pos metric: {test_pos_metric:.4f}, Test neg metric: {test_neg_metric:.4f}, Test fid plus probs: {test_fid_plus_probs:.4f}, Test fid minus probs: {test_fid_minus_probs:.4f}, Test fid plus acc: {test_fid_plus_acc:.4f}, Test fid minus acc: {test_fid_minus_acc:.4f}, Test auc: {test_auc:.4f}, Test f1: {test_f1:.4f}, Test iou: {test_iou:.4f}')

y_probs, y_masks, explanations, pos_preds, neg_preds, baseline_preds, y_trues = selector.predict_batch(test_loader)

if config['task_type'] == 'classification':
    pos_preds = [pred.argmax() for pred in pos_preds]
    neg_preds = [pred.argmax() for pred in neg_preds]
    cm_pos = confusion_matrix(y_trues, pos_preds)
    cm_neg = confusion_matrix(y_trues, neg_preds)
    sns.heatmap(cm_pos, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{log_path}/pos_predictor_cm.png')
    plt.clf()
    sns.heatmap(cm_neg, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{log_path}/neg_predictor_cm.png')
    plt.clf()

if config['task_type'] == 'regression':
    pos_preds = [pred.item() for pred in pos_preds]
    neg_preds = [pred.item() for pred in neg_preds]
    plt.scatter(y_trues, pos_preds)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.savefig(f'{log_path}/pos_predictor_scatter.png')
    plt.clf()
    plt.scatter(y_trues, neg_preds)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.savefig(f'{log_path}/neg_predictor_scatter.png')
    plt.clf()
