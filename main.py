import yaml
import os
import logging
import time
import numpy as np
import networkx as nx
import torch
from src.BAMotifs import (
    BaseBAMotifs, BAMotifs,
    BAImbalancedMotifs, BAIgnoringMotifs,
    BAORMotifs, BAXORMotifs, BAANDMotifs, BAVolumeMotifs
)
from src.MolecularDataset import (
    MolecularDataset, AlkaneCarbonylDataset,
    BenzeneDataset, FluorideCarbonylDataset
)
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from src.GNN import GNN
from src.Explainer import Explainer
from src.utils import log_metrics, log_metrics_tb
from src.metrics import save_tsneplot, save_cmplot, save_regplot
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
    elif dataset_name == 'BAVolumeMotifs':
        return BAVolumeMotifs(data_path, **dataset_params)
    elif config['dataset'] == 'AlkaneCarbonyl':
        return AlkaneCarbonylDataset(data_path)
    elif config['dataset'] == 'Benzene':
        return BenzeneDataset(data_path)
    elif config['dataset'] == 'FluorideCarbonyl':
        return FluorideCarbonylDataset(data_path)
    elif config['dataset'] == 'MNIST':
        class SuperPixelTransform(object):
            def __call__(self, data):
                data = T.ToUndirected()(data)
                data.x = torch.cat([data.x, data.pos], dim=1)
                data.edge_attr = data.edge_attr.unsqueeze(-1)
                indices = torch.topk(data.x[:,0], k=15).indices
                node_mask = torch.zeros(data.num_nodes, dtype=torch.long)
                node_mask[indices] = 1
                data.true = node_mask
                return data
        return GNNBenchmarkDataset(
            root=data_path,
            name=config['dataset'],
            split='train',
            pre_transform=SuperPixelTransform()
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
        class_weights=config['class_weights']
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

# calculate expected sparsity
if hasattr(dataset[0], 'true'):
    expected_sparsity = np.array([data.true.sum().item() / data.num_nodes for data in train_loader]).mean()
    logging.info(f"Expected sparsity: {expected_sparsity}")
# calculate class weights
if config['task_type'] == 'classification':
    class_counts = torch.zeros(dataset.num_classes)
    for data in dataset:
        class_counts[data.y] += 1
    class_weights = 1.0 / class_counts
    class_weights /= class_weights.sum()
    config['class_weights'] = class_weights
    logging.info(f"Class weights: {config['class_weights']}")
else:
    config['class_weights'] = None

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
            logging.info(f'Best validation metric updated: {best_val_metric:.4f}, saving baseline...')
            torch.save(baseline.state_dict(), os.path.join(log_path, 'baseline.pt'))
        if config['logging']:
            log_metrics(logging, train_metrics, "Train", epoch)
            log_metrics(logging, val_metrics, "Val", epoch)
        if config['tb_logging']:
            log_metrics_tb(writer, train_metrics, "BASELINE/train", epoch)
            log_metrics_tb(writer, val_metrics, "BASELINE/val", epoch)
    baseline.load_state_dict(torch.load(os.path.join(log_path, 'baseline.pt'), weights_only=False))
else:
    baseline_path_pretrained = os.path.join(log_path_base, baseline_pretrained)
    baseline.load_state_dict(torch.load(os.path.join(baseline_path_pretrained, 'baseline.pt'), weights_only=False))
val_metrics = baseline.evaluate_batch(val_loader)
log_metrics(logging, val_metrics, "Val")
test_metrics = baseline.evaluate_batch(test_loader)
log_metrics(logging, test_metrics, "Test")

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
        val_fid_diff = val_metrics['fidplus'] - val_metrics['fidminus']
        if val_fid_diff > best_val_fid_diff and val_metrics['sparsity'] < config['sparsity'] + 0.01:
            best_val_fid_diff = val_fid_diff
            logging.info(f'Best validation fidelity difference updated: {best_val_fid_diff:.4f}, saving explainer...')
            torch.save(explainer.state_dict(), os.path.join(log_path, 'explainer.pt'))
        if config['logging']:
            log_metrics(logging, train_metrics, "Train", epoch)
            log_metrics(logging, val_metrics, "Val", epoch)
        if config['tb_logging']:
            log_metrics_tb(writer, train_metrics, "EXPLAINER/train", epoch)
            log_metrics_tb(writer, val_metrics, "EXPLAINER/val", epoch)
    explainer.load_state_dict(torch.load(os.path.join(log_path, 'explainer.pt'), weights_only=False))
else:
    explainer_path_pretrained = os.path.join(log_path_base, explainer_pretrained)
    explainer.load_state_dict(torch.load(os.path.join(explainer_path_pretrained, 'explainer.pt'), weights_only=False))

val_metrics = explainer.evaluate_batch(val_loader)
test_metrics = explainer.evaluate_batch(test_loader)
log_metrics(logging, val_metrics, "Val")
log_metrics(logging, test_metrics, "Test")

val_metrics_random = explainer.evaluate_batch(val_loader, random=True)
test_metrics_random = explainer.evaluate_batch(test_loader, random=True)
log_metrics(logging, val_metrics_random, "Val Random")
log_metrics(logging, test_metrics_random, "Test Random")

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
            log_metrics(logging, train_metrics, "Train", epoch)
            log_metrics(logging, val_metrics, "Val", epoch)
    val_metrics = explainer.evaluate_batch(val_loader)
    test_metrics = explainer.evaluate_batch(test_loader)
    log_metrics(logging, val_metrics, "Val")
    log_metrics(logging, test_metrics, "Test")

# visualize predictions
plt.rcParams.update({"figure.dpi": 120})
if config["visualize_predictions"]:
    y_probs, y_masks, explanations, pos_embs, pos_preds, neg_embs, neg_preds, y_trues = explainer.explain_batch(test_loader)
    for idx, data in enumerate(test_set):
        edge_attrs = ["edge_attr"] if data.edge_attr is not None else None
        G = to_networkx(data, to_undirected=True, edge_attrs=edge_attrs)
        if isinstance(dataset, BaseBAMotifs):
            pos = nx.kamada_kawai_layout(G)
            # pos = nx.spring_layout(G)
            # motif_nodes_list = list(nx.connected_components(G.subgraph(true_nodes)))
            # for i, motif_nodes in enumerate(motif_nodes_list):
            #     motif_pos = nx.kamada_kawai_layout(G.subgraph(motif_nodes), scale=0.2)
            #     shift = np.array([0.5 * (1 if i == 0 else -1), 0.5])
            #     for node in motif_pos:
            #         pos[node] = motif_pos[node] + shift
        elif isinstance(dataset, MolecularDataset):
            pos = nx.kamada_kawai_layout(G)
        elif isinstance(dataset, GNNBenchmarkDataset):
            pos = data.pos.cpu().numpy()
            pos = np.array([pos[:, 1], -pos[:, 0]]).T
        true_nodes = set(np.flatnonzero(explanations[idx].astype(bool)))
        pred_nodes = set(np.flatnonzero(y_masks[idx][:, 0].astype(bool)))
        fig, ax = plt.subplots(figsize=(6, 6))
        tp_nodes = [n for n in G.nodes() if n in true_nodes and n in pred_nodes]
        fp_nodes = [n for n in G.nodes() if n in pred_nodes and n not in true_nodes]
        fn_nodes = [n for n in G.nodes() if n in true_nodes and n not in pred_nodes]
        for nodelist, color in [(fp_nodes, "#FF3300"), (fn_nodes, "#3366FF"), (tp_nodes, "#00FF00")]:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodelist, node_color=color,
                                   node_size=600, alpha=1.0, linewidths=0)
        if isinstance(dataset, BaseBAMotifs):
            node_color = "orange"
            edge_color = "#808080"
            edge_width = 1.5
        elif isinstance(dataset, MolecularDataset):
            ATOM_TYPES = [
                'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*'
            ]
            atom_indices = data.x.argmax(dim=1).cpu().numpy()
            labels_dict = {i: ATOM_TYPES[idx] for i, idx in enumerate(atom_indices)}
            nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8, font_color="black")
            node_color = "white"
            edge_color = "#000000"
            edge_width = [
                np.array(attr["edge_attr"]).argmax(-1) + 1
                for _, _, attr in G.edges(data=True)
            ]
        elif isinstance(dataset, GNNBenchmarkDataset):
            node_color = data.x.cpu().numpy()[:, 0]
            edge_color = "#000000"
            edge_width = 1.5
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(G.nodes()), node_color=node_color,
                        node_size=400, alpha=1.0, linewidths=0)    
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width, edge_color=edge_color, alpha=0.8)
        true_class = y_trues[idx].item()
        pos_pred_class = pos_preds[idx].argmax().item()
        neg_pred_class = neg_preds[idx].argmax().item()
        ax.set_title(f"graph {idx} | y={true_class} | pos={pos_pred_class} | neg={neg_pred_class}", fontsize=16)
        ax.axis("off")
        fig.tight_layout()
        filename = f'{log_path}/graph_{true_class}_{pos_pred_class}_{neg_pred_class}_{idx}.png'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(filename)
