import yaml
import os
import logging
import time
import numpy as np
import networkx as nx
import torch
from BAMotifs import (
    BAMotifs, BAImbalancedMotifs, BAIgnoringMotifs,
    BAORMotifs, BAXORMotifs, BAANDMotifs
)
from BAMotifsVolumeDataset import BAMotifsVolumeDataset
from AlkaneCarbonylDataset import AlkaneCarbonylDataset
from BenzeneDataset import BenzeneDataset
from FluorideCarbonylDataset import FluorideCarbonylDataset
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from GNN import GNN
from Explainer import Explainer
from utils import log_metrics, log_metrics_tb
from metrics import save_tsneplot, save_cmplot, save_regplot
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# load config yaml
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# set random seed
seed = config['seed']
torch.manual_seed(seed)
np.random.seed(seed)

# initialize logger
log_path_base = f"{config['log_path']}/{config['dataset']}"
run_id = str(int(time.time()))
log_path = os.path.join(log_path_base, f'run_{run_id}')
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_path, 'log.txt')),
        logging.StreamHandler()
    ]
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
if config['tb_logging']:
    writer = SummaryWriter(log_path)
with open(f'{log_path}/config.yaml', 'w') as f:
    yaml.dump(config, f)

# load dataset
data_path = f"{config['data_path']}/{config['dataset']}"
os.makedirs(data_path, exist_ok=True)

def load_dataset(config, data_path):
    dataset_name = config['dataset']
    dataset_params = config['dataset_params']
    if dataset_name == 'BAMotifs':
        return BAMotifs(data_path, **dataset_params)
    elif dataset_name == 'BAImbalancedMotifs':
        return BAImbalancedMotifs(data_path, **dataset_params)
    elif dataset_name == 'BAIgnoringMotifs':
        return BAIgnoringMotifs(data_path, **dataset_params)
    elif dataset_name == 'BAORMotifs':
        return BAORMotifs(data_path, **dataset_params)
    elif dataset_name == 'BAXORMotifs':
        return BAXORMotifs(data_path, **dataset_params)
    elif dataset_name == 'BAANDMotifs':
        return BAANDMotifs(data_path, **dataset_params)
    #
    elif config['dataset'] == 'BAMotifsVolume':
        return BAMotifsVolumeDataset(data_path, num_graphs=500, ba_nodes=25, attach_prob=0.1)
    elif config['dataset'] == 'AlkaneCarbonyl':
        return AlkaneCarbonylDataset(data_path)
    elif config['dataset'] == 'Benzene':
        return BenzeneDataset(data_path)
    elif config['dataset'] == 'FluorideCarbonyl':
        return FluorideCarbonylDataset(data_path)
    #
    elif config['dataset'] == 'MNIST':
        class SuperPixelTransform(object):
            def __call__(self, data):
                data = T.ToUndirected()(data)
                data.x = torch.cat([data.x, data.pos], dim=1)
                data.edge_attr = data.edge_attr.unsqueeze(-1)
                data.true =  torch.ones_like(data.num_nodes, device=data.x.device)
                return data
        return GNNBenchmarkDataset(
            data_path,
            config['dataset'],
            pre_transform=SuperPixelTransform(),
        )
    else:
        raise ValueError(f"Dataset {config['dataset']} not supported")

def create_gnn(config, in_channels, out_channels, edge_dim):
    return GNN(
        conv_type=config['conv_type'],
        task_type=config['task_type'],
        learning_rate=config['learning_rate'],
        in_channels=in_channels,
        num_layers=config['num_layers'],
        hidden_channels=config['hidden_channels'],
        out_channels=out_channels,
        edge_dim=edge_dim,
        use_norm=config['use_norm'],
    )

dataset = load_dataset(config, data_path)
dataset = dataset.shuffle()
logging.info(f'Dataset {config["dataset"]} loaded, number of graphs: {len(dataset)}')

# split dataset
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset,
    [config['train_size'], config['val_size'], config['test_size']],
)

# create data loaders
train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_set, batch_size=len(val_set))
test_loader = DataLoader(test_set, batch_size=len(test_set))

expected_sparsity = np.array([data.true.sum().item() / data.num_nodes for data in train_loader]).mean()
logging.info(f"Expected sparsity: {expected_sparsity}")

num_node_features = dataset[0].x.shape[1]
num_edge_features = 0 if dataset[0].edge_attr is None else dataset[0].edge_attr.shape[1]
num_classes = dataset.num_classes if config['task_type'] == 'classification' else 1

# define baseline model
baseline = create_gnn(config, num_node_features, num_classes, num_edge_features)
# logging.info('Baseline model:\n', baseline)

baseline_pretrained = config['baseline_pretrained']

# train baseline model
if baseline_pretrained is None:
    best_val_metric = float('inf') if config['task_type'] == 'regression' else 0
    for epoch in range(config['baseline_epochs']):
        train_metrics = baseline.train_batch(train_loader)
        val_metrics = baseline.evaluate_batch(val_loader)
        val_metric = val_metrics['metric']
        if (config['task_type'] == 'classification' and best_val_metric < val_metric) or \
           (config['task_type'] == 'regression' and best_val_metric > val_metric):
            best_val_metric = val_metric
            logging.info(f'Best validation accuracy updated: {best_val_metric:.4f}, saving baseline...')
            torch.save(baseline.state_dict(), os.path.join(log_path, 'baseline.pt'))
        if config['logging']:
            log_metrics(logging, train_metrics, epoch, "Train")
            log_metrics(logging, val_metrics, epoch, "Val")
        if config['tb_logging']:
            log_metrics_tb(writer, train_metrics, epoch, "train")
            log_metrics_tb(writer, val_metrics, epoch, "val")
    baseline.load_state_dict(torch.load(os.path.join(log_path, 'baseline.pt'), weights_only=False))
else:
    baseline_path_pretrained = os.path.join(log_path_base, baseline_pretrained)
    baseline.load_state_dict(torch.load(os.path.join(baseline_path_pretrained, 'baseline.pt'), weights_only=False))
val_metrics = baseline.evaluate_batch(val_loader)
log_metrics(logging, val_metrics, epoch, "Val")
test_metrics = baseline.evaluate_batch(test_loader)
log_metrics(logging, test_metrics, epoch, "Test")

x_embs, y_preds, y_trues = baseline.predict_batch(test_loader)
save_tsneplot(y_trues, x_embs, save_path=f'{log_path}/baseline_tsne.png')
if config['task_type'] == 'classification':
    save_cmplot(y_trues, y_preds, save_path=f'{log_path}/baseline_cm.png')
elif config['task_type'] == 'regression':
    save_regplot(y_trues, y_preds, save_path=f'{log_path}/baseline_scatter.png')

# define predictor for selected subgraph
pos_predictor = create_gnn(config, num_node_features, num_classes, num_edge_features)

# define predictor for unselected subgraph
neg_predictor = create_gnn(config, num_node_features, num_classes, num_edge_features)

# define explainer
explainer = Explainer(
    baseline=baseline,
    pos_predictor=pos_predictor,
    neg_predictor=neg_predictor,
    sparsity=config['sparsity'],
    reward_coeff=config['reward_coeff'],
)
# logging.info('Explainer model:\n', explainer)

explainer_pretrained = config['explainer_pretrained']

if explainer_pretrained is None:
    best_val_fid_diff = 0
    # explainer.sparsity = 1.0
    for epoch in range(config['explainer_epochs']):
        # if epoch % 25 == 0 and not np.isclose(explainer.sparsity, config['sparsity']):
        #     explainer.sparsity -= 0.05
        #     logging.info(f'Sparsity updated: {explainer.sparsity:.4f}')
        train_metrics = explainer.train_batch(train_loader)
        val_metrics = explainer.evaluate_batch(val_loader)
        val_fid_diff = val_metrics['fidplus_prob'] - val_metrics['fidminus_prob']
        if val_fid_diff > best_val_fid_diff and val_metrics['sparsity'] < config['sparsity'] + 0.01:
            best_val_fid_diff = val_fid_diff
            logging.info(f'Best validation fidelity difference updated: {best_val_fid_diff:.4f}, saving explainer...')
            torch.save(explainer.state_dict(), os.path.join(log_path, 'explainer.pt'))
        if config['logging']:
            log_metrics(logging, train_metrics, epoch, "Train")
            log_metrics(logging, val_metrics, epoch, "Val")
        if config['tb_logging']:
            log_metrics_tb(writer, train_metrics, epoch, "train")
            log_metrics_tb(writer, val_metrics, epoch, "val")
    explainer.load_state_dict(torch.load(os.path.join(log_path, 'explainer.pt'), weights_only=False))
else:
    explainer_path_pretrained = os.path.join(log_path_base, explainer_pretrained)
    explainer.load_state_dict(torch.load(os.path.join(explainer_path_pretrained, 'explainer.pt'), weights_only=False))

val_metrics = explainer.evaluate_batch(val_loader)
test_metrics = explainer.evaluate_batch(test_loader)
log_metrics(logging, val_metrics, epoch, "Val")
log_metrics(logging, test_metrics, epoch, "Test")

val_metrics_random = explainer.evaluate_batch(val_loader, random=True)
test_metrics_random = explainer.evaluate_batch(test_loader, random=True)
log_metrics(logging, val_metrics_random, epoch, "Val Random")
log_metrics(logging, test_metrics_random, epoch, "Test Random")

y_probs, y_masks, explanations, pos_embs, pos_preds, neg_embs, neg_preds, y_trues = explainer.explain_batch(test_loader)
save_tsneplot(y_trues, pos_embs, save_path=f'{log_path}/pos_predictor_tsne.png')
save_tsneplot(y_trues, neg_embs, save_path=f'{log_path}/neg_predictor_tsne.png')
if config['task_type'] == 'classification':
    save_cmplot(y_trues, pos_preds, save_path=f'{log_path}/pos_predictor_cm.png')
    save_cmplot(y_trues, neg_preds, save_path=f'{log_path}/neg_predictor_cm.png')
elif config['task_type'] == 'regression':
    save_regplot(y_trues, pos_preds, save_path=f'{log_path}/pos_predictor_reg.png')
    save_regplot(y_trues, neg_preds, save_path=f'{log_path}/neg_predictor_reg.png')

if config['tb_logging']:
    writer.close()

# retrain new pos_predictor and neg_predictor
if config['retrain_predictors']:
    explainer.pos_predictor = create_gnn(config, num_node_features, num_classes, num_edge_features)
    explainer.neg_predictor = create_gnn(config, num_node_features, num_classes, num_edge_features)
    for epoch in range(config['baseline_epochs']):
        train_metrics = explainer.train_batch(train_loader)
        val_metrics = explainer.evaluate_batch(val_loader)
        if config['logging']:
            log_metrics(logging, train_metrics, epoch, "Train")
            log_metrics(logging, val_metrics, epoch, "Val")
    val_metrics = explainer.evaluate_batch(val_loader)
    test_metrics = explainer.evaluate_batch(test_loader)
    log_metrics(logging, val_metrics, epoch, "Val")
    log_metrics(logging, test_metrics, epoch, "Test")

