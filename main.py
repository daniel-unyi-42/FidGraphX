import yaml
import os
import logging
import time
import numpy as np
import networkx as nx
import torch
from src.BAMotifs import (
    BaseBAMotifs, BAMotifs,
    BAORMotifs, BAXORMotifs, BAANDMotifs, BAVolumeMotifs
)
from src.MolecularDataset import (
    MolecularDataset, BenzeneDataset, BenzeneCountDataset
)
# from src.PowerGrid import PowerGrid
from src.DHCPScanAge import DHCPScanAgeDataset
from torch_geometric.datasets import GNNBenchmarkDataset, CoMA
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from src.GNN import GNN
from src.Explainer import Explainer
from src.RandomExplainer import RandomExplainer
from src.ProbeEvaluator import ProbeEvaluator
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
save_tsneplot(y_trues, x_embs, save_path=f'{log_path}/baseline_tsne.png')
if config['task_type'] == 'classification':
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
save_tsneplot(y_trues, pos_embs, save_path=f'{log_path}/pos_predictor_tsne.png')
save_tsneplot(y_trues, neg_embs, save_path=f'{log_path}/neg_predictor_tsne.png')
if config['task_type'] == 'classification':
    save_cmplot(y_trues, pos_preds, save_path=f'{log_path}/pos_predictor_cm.png')
    save_cmplot(y_trues, neg_preds, save_path=f'{log_path}/neg_predictor_cm.png')
elif config['task_type'] == 'regression':
    save_regplot(y_trues, pos_preds, save_path=f'{log_path}/pos_predictor_reg.png', title="Pos Predictor Regplot")
    save_regplot(y_trues, neg_preds, save_path=f'{log_path}/neg_predictor_reg.png', title="Neg Predictor Regplot")
    ########## paper regplot ##########
    from sklearn.metrics import r2_score

    # ---- GET BASELINE PREDICTIONS ----
    _, y_preds_baseline, _ = baseline.predict_batch(test_loader)

    def single_regplot(ax, y_true, y_pred, color, title, show_ylabel=True, xlim=None):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Scatter (swap axes: x=prediction, y=true)
        ax.scatter(y_pred, y_true, s=20, alpha=0.5, color=color)

        # Fitted line
        slope = np.sum(y_true * y_pred) / np.sum(y_pred ** 2)
        x_vals = np.array([y_pred.min(), y_pred.max()])
        y_vals = slope * x_vals
        ax.plot(x_vals, y_vals, color=color, linewidth=2.0)

        # 45-degree reference line (adjusted for swapped axes)
        y_min, y_max = y_true.min(), y_true.max()
        ax.plot(x_vals, x_vals, 'k--', linewidth=1.5)

        # Titles + R²
        r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{title}\n$R^2={r2:.3f}$", fontsize=16, pad=12)
        ax.set_xlabel("Predicted Value", fontsize=16)
        if show_ylabel:
            ax.set_ylabel("True Value", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
        if not show_ylabel:
            ax.tick_params(axis="y", labelleft=False)
        
        # Set x-axis limits based on the xlim parameter
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim([x_vals.min(), x_vals.max()])
        ax.set_ylim([y_min, y_max])  # Uniform y-scale
        
        # Add grid
        ax.grid(True, alpha=0.5)

    # Colors
    blue  = "#005BBB"  # Baseline
    green = "#009933"  # Pos
    red   = "#CC0000"  # Neg

    # Calculate uniform y-scale across all plots
    y_true_all = np.array(y_trues).flatten()
    y_min_global = y_true_all.min()
    y_max_global = y_true_all.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    single_regplot(axes[0], y_trues, y_preds_baseline, blue,  "Target GNN Reg. Plot", show_ylabel=True, xlim=[-1, 9])
    single_regplot(axes[1], y_trues, pos_preds,        green, "Pos. Predictor Reg. Plot", show_ylabel=False, xlim=[-1, 9])
    single_regplot(axes[2], y_trues, [-neg_pred for neg_pred in neg_preds],        red,   "Neg. Predictor Reg. Plot", show_ylabel=False, xlim=[-1, 17])

    # Set uniform y-scale for all subplots
    for ax in axes:
        ax.set_ylim([y_min_global, y_max_global])

    # Reduce spacing between subplots
    plt.subplots_adjust(wspace=0.0, left=0.08, right=0.95, top=0.9, bottom=0.15)
    outpath = f"{log_path}/regression_models_comparison.png"
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

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

# visualize predictions
plt.rcParams.update({"figure.dpi": 120})
if config["visualize_predictions"]:
    y_probs, y_masks, explanations, pos_embs, pos_preds, neg_embs, neg_preds, y_trues = explainer.explain_batch(test_loader)
    for idx, data in enumerate(test_set):
        true_class = y_trues[idx].item()
        pos_pred_class = pos_preds[idx].argmax().item()
        neg_pred_class = neg_preds[idx].argmax().item()
        if isinstance(dataset, CoMA):
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            def rotate(points, angle_deg, axis="z"):
                angle = np.deg2rad(angle_deg)
                c, s = np.cos(angle), np.sin(angle)
                if axis == "z":
                    R = np.array([
                        [ c, -s, 0],
                        [ s,  c, 0],
                        [ 0,  0, 1]
                    ])
                elif axis == "y":
                    R = np.array([
                        [ c,  0, s],
                        [ 0,  1, 0],
                        [-s,  0, c]
                    ])
                elif axis == "x":
                    R = np.array([
                        [1,  0,  0],
                        [0,  c, -s],
                        [0,  s,  c]
                    ])
                return points @ R.T
            points = data.pos.cpu().numpy()            # (N, 3)
            points = rotate(points, angle_deg=-15, axis="y")
            faces = data.face.t().cpu().numpy()        # (F, 3)
            # Explanation-based selection
            selected_vertices = set(np.flatnonzero(y_masks[idx][:, 0].astype(bool)))
            # Colors: gray for non-selected, green for selected
            colors = np.ones((points.shape[0], 3)) * 0.6
            if len(selected_vertices) > 0:
                sel_idx = np.array(list(selected_vertices))
                colors[sel_idx] = np.array([0.0, 1.0, 0.0])
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')
            # Build mesh triangles for matplotlib
            mesh_tris = []
            face_colors = []
            for f in faces:
                mesh_tris.append(points[f])
                # average color for the triangle (simple yet effective)
                tri_color = colors[f].mean(axis=0)
                face_colors.append(tri_color)
            poly = Poly3DCollection(
                mesh_tris,
                facecolors=face_colors,
                edgecolors='black',
                linewidths=0.1,
                alpha=1.0
            )
            ax.add_collection3d(poly)
            ax.auto_scale_xyz(points[:,0], points[:,1], points[:,2])
            ax.set_axis_off()
            ax.view_init(elev=90, azim=-90)
            filename = f"{log_path}/graph_{true_class}_{pos_pred_class}_{neg_pred_class}_{idx}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(filename)
        else:
            edge_attrs = ["edge_attr"] if data.edge_attr is not None else None
            G = to_networkx(data, to_undirected=True, edge_attrs=edge_attrs)
            selected_nodes = set(np.flatnonzero(y_masks[idx][:, 0].astype(bool)))
            unselected_nodes = [n for n in G.nodes if n not in selected_nodes]
            fig, ax = plt.subplots(figsize=(6, 6))
            if isinstance(dataset, BaseBAMotifs):
                pos = nx.kamada_kawai_layout(G)
                # pos = nx.spring_layout(G)
                # motif_nodes_list = list(nx.connected_components(G.subgraph(true_nodes)))
                # for i, motif_nodes in enumerate(motif_nodes_list):
                #     motif_pos = nx.kamada_kawai_layout(G.subgraph(motif_nodes), scale=0.2)
                #     shift = np.array([0.5 * (1 if i == 0 else -1), 0.5])
                #     for node in motif_pos:
                #         pos[node] = motif_pos[node] + shift
                nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, edge_color="#555555", alpha=0.8)
                nx.draw_networkx_nodes(G, pos, nodelist=unselected_nodes, node_color="#FFD580", edgecolors="#CC6600", linewidths=1.8, node_size=600, ax=ax)
                nx.draw_networkx_nodes(G, pos, nodelist=list(selected_nodes), node_color="#8B0000", edgecolors="#25D025", linewidths=3.6, node_size=650, ax=ax)
            elif isinstance(dataset, MolecularDataset):
                pos = nx.kamada_kawai_layout(G)
                # atom címkék
                ATOM_TYPES = ['C','N','O','S','F','P','Cl','Br','Na','Ca','I','B','H','*']
                atom_idx = data.x.argmax(dim=1).cpu().numpy()
                labels = {i: ATOM_TYPES[j] for i, j in enumerate(atom_idx)}
                edge_width = [np.array(attr["edge_attr"]).argmax(-1) + 1 for _, _, attr in G.edges(data=True)]
                nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width, edge_color="black", alpha=0.9)

                gt_nodes = [i for i, t in enumerate(data.true.cpu().numpy()) if t == 1]
                non_gt_nodes = [i for i, t in enumerate(data.true.cpu().numpy()) if t == 0]
                nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=non_gt_nodes, node_color="white", node_size=500, linewidths=0)
                nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=gt_nodes, node_color="#CF8D89", edgecolors="black", linewidths=1.2, node_size=600)

                # node_colors = ["white" for _ in G.nodes()]
                # nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=500, linewidths=0)
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=20, font_color="black")
                nx.draw_networkx_nodes(G, pos, nodelist=list(selected_nodes), node_color="none", edgecolors="#25D025", linewidths=3.6, node_size=650, ax=ax)
            elif isinstance(dataset, GNNBenchmarkDataset):
                pos = data.pos.cpu().numpy()
                pos = np.array([[ y, -x] for x, y in pos])
                nx.draw_networkx_edges(G, pos, ax=ax, width=1.2, edge_color="#444444", alpha=0.4)
                node_color = data.x.cpu().numpy()[:, 0]
                nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_color=node_color, cmap="gray_r", node_size=350, edgecolors="black", linewidths=0.5, ax=ax)
                nx.draw_networkx_nodes(G, pos, nodelist=list(selected_nodes), node_color="none", edgecolors="#25D025", linewidths=3.6, node_size=500, ax=ax)
            ax.set_title(f"graph {idx} | y={true_class} | pos={pos_pred_class} | neg={neg_pred_class}", fontsize=16)
            ax.axis("off")
            fig.tight_layout()
            filename = f'{log_path}/graph_{true_class}_{pos_pred_class}_{neg_pred_class}_{idx}.png'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(filename)
