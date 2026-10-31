import yaml
import os
import logging
import time
import numpy as np
import torch
from src.BAMotifs import (
    BAMotifs,
    BAORMotifs, BAXORMotifs, BAANDMotifs, BAVolumeMotifs
)
from src.MolecularDataset import (
    BenzeneDataset, BenzeneCountDataset
)
# from src.PowerGrid import PowerGrid
from src.DHCPScanAge import DHCPScanAgeDataset
from torch_geometric.datasets import GNNBenchmarkDataset, CoMA
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from src.GNN import GNN
from src.Explainer import Explainer
from src.RandomExplainer import RandomExplainer
from src.ProbeEvaluator import ProbeEvaluator
from src.utils import (
    log_metrics, log_metrics_tb, save_regplot,
    save_regression_comparison_plot, save_tsne_comparison_plot,
    save_tsneplot, visualize_predictions,
)
from src.metrics import save_cmplot
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
    if dataset_name == 'DHCPScanAge':
        return DHCPScanAgeDataset(root='dHCP', **dataset_params)
    elif dataset_name == 'BAMotifs':
        return BAMotifs(data_path, **dataset_params)
    elif dataset_name == 'BAORMotifs':
        return BAORMotifs(data_path, **dataset_params)
    elif dataset_name == 'BAXORMotifs':
        return BAXORMotifs(data_path, **dataset_params)
    elif dataset_name == 'BAANDMotifs':
        return BAANDMotifs(data_path, **dataset_params)
    elif dataset_name == 'BAVolumeMotifs':
        return BAVolumeMotifs(data_path, **dataset_params)
    elif config['dataset'] == 'Benzene':
        return BenzeneDataset(data_path)
    elif config['dataset'] == 'BenzeneCount':
        return BenzeneCountDataset(data_path)
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
    elif config['dataset'] == 'CoMA':
        class ModelNetTransform(object):
            def __call__(self, data):
                data.x = data.pos
                return data
        return CoMA(
            root=data_path,
            train=True,
            transform=None,
            pre_transform=T.Compose([T.FaceToEdge(remove_faces=False), T.ToUndirected(), ModelNetTransform()])
        )
    elif config['dataset'] == 'PowerGrid':
        return PowerGrid(data_path, name='IEEE24', datatype='regression', pre_filter=lambda data: data.y < 0.1)
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
train_loader = DataLoader(train_set, batch_size=config['train_batch_size'], shuffle=True)
eval_batch_size = config['eval_batch_size']
val_loader = DataLoader(val_set, batch_size=eval_batch_size)
test_loader = DataLoader(test_set, batch_size=eval_batch_size)

if hasattr(dataset[0], 'true'):
    expected_sparsity = np.array([data.true.sum().item() / data.num_nodes for data in train_loader]).mean()
    logging.info(f"Expected sparsity: {expected_sparsity}")
# calculate class weights
if config['task_type'] == 'classification':
    class_counts = torch.ones(dataset.num_classes)
    # class_counts = torch.zeros(dataset.num_classes)
    # for data in dataset:
    #     class_counts[data.y] += 1
    class_weights = 1.0 / class_counts
    # class_weights /= class_weights.sum()
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
if config['task_type'] == 'classification':
    save_tsneplot(y_trues, x_embs, save_path=f'{log_path}/baseline_tsne.png')
    save_cmplot(y_trues, y_preds, save_path=f'{log_path}/baseline_cm.png')
elif config['task_type'] == 'regression':
    save_regplot(y_trues, y_preds, save_path=f'{log_path}/baseline_scatter.png', title="Baseline Regplot")

# define predictor for selected subgraph
pos_predictor = create_gnn(config, num_node_features, num_classes, num_edge_features)

# define predictor for unselected subgraph
neg_predictor = create_gnn(config, num_node_features, num_classes, num_edge_features)

# define explainer
explainer = Explainer(
    baseline=baseline,
    pos_predictor=pos_predictor,
    neg_predictor=neg_predictor,
    target_type=config['target_type'],
    sparsity=config['sparsity'],
    reward_coeff=config['reward_coeff'],
)
# logging.info('Explainer model:\n', explainer)

explainer_pretrained = config['explainer_pretrained']

if explainer_pretrained is None:
    best_val_fid_diff = 0
    for epoch in range(config['explainer_epochs']):
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

y_probs, y_masks, explanations, pos_embs, pos_preds, neg_embs, neg_preds, y_trues = explainer.explain_batch(test_loader)
if config['task_type'] == 'classification':
    save_tsne_comparison_plot(
        y_trues,
        x_embs,
        pos_embs,
        neg_embs,
        save_path=f'{log_path}/predictors_tsne.png',
    )
    save_cmplot(y_trues, pos_preds, save_path=f'{log_path}/pos_predictor_cm.png')
    save_cmplot(y_trues, neg_preds, save_path=f'{log_path}/neg_predictor_cm.png')
elif config['task_type'] == 'regression':
    outpath = f"{log_path}/regression_models_comparison.png"
    save_regression_comparison_plot(y_trues, y_preds, pos_preds, neg_preds, outpath)
    logging.info(f"Saved combined regression plot to {outpath}")

if config['tb_logging']:
    writer.close()

# retrain new pos_predictor and neg_predictor
if config['retrain_predictors']:
    fid_probe_evaluator = ProbeEvaluator(
        mask_provider=explainer,
        target_model=baseline,
        probe_factory=lambda: create_gnn(config, num_node_features, num_classes, num_edge_features),
        task_type=config['task_type'],
        target_type=config['target_type'],
        log_path=log_path,
        checkpoint_tag='fidgraphx',
    )
    for epoch, train_metrics, val_metrics in fid_probe_evaluator.fit(
        train_loader,
        val_loader,
        config['baseline_epochs'],
    ):
        if config['logging']:
            log_metrics(logging, train_metrics, "Train Eval Probes (FidGraphX)", epoch)
            log_metrics(logging, val_metrics, "Val Eval Probes (FidGraphX)", epoch)
    fid_probe_evaluator.load_best_probes()
    val_metrics = fid_probe_evaluator.evaluate_epoch(val_loader)
    test_metrics = fid_probe_evaluator.evaluate_epoch(test_loader)
    log_metrics(logging, val_metrics, "Val Eval Probes (FidGraphX)")
    log_metrics(logging, test_metrics, "Test Eval Probes (FidGraphX)")
    fid_pos_embs, fid_pos_preds, fid_neg_embs, fid_neg_preds = fid_probe_evaluator.predict_batch(test_loader)
    if config['task_type'] == 'classification':
        save_tsne_comparison_plot(
            y_trues,
            x_embs,
            fid_pos_embs,
            fid_neg_embs,
            save_path=f'{log_path}/fidgraphx_probes_tsne.png',
        )
    elif config['task_type'] == 'regression':
        outpath = f'{log_path}/fidgraphx_probes_regression_comparison.png'
        save_regression_comparison_plot(y_trues, y_preds, fid_pos_preds, fid_neg_preds, outpath)
        logging.info(f"Saved retrained probe regression plot to {outpath}")

    random_explainer = RandomExplainer(
        sparsity=config['sparsity'],
        device=baseline.device,
    )
    random_probe_evaluator = ProbeEvaluator(
        mask_provider=random_explainer,
        target_model=baseline,
        probe_factory=lambda: create_gnn(config, num_node_features, num_classes, num_edge_features),
        task_type=config['task_type'],
        target_type=config['target_type'],
        log_path=log_path,
        checkpoint_tag='random',
    )
    for epoch, train_metrics, val_metrics in random_probe_evaluator.fit(
        train_loader,
        val_loader,
        config['baseline_epochs'],
    ):
        if config['logging']:
            log_metrics(logging, train_metrics, "Train Eval Probes (Random)", epoch)
            log_metrics(logging, val_metrics, "Val Eval Probes (Random)", epoch)
    random_probe_evaluator.load_best_probes()
    val_metrics = random_probe_evaluator.evaluate_epoch(val_loader)
    test_metrics = random_probe_evaluator.evaluate_epoch(test_loader)
    log_metrics(logging, val_metrics, "Val Eval Probes (Random)")
    log_metrics(logging, test_metrics, "Test Eval Probes (Random)")

visualize_predictions(
    dataset,
    test_set,
    test_loader,
    explainer,
    log_path,
    config["visualize_predictions"],
)
