import logging
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.datasets import GNNBenchmarkDataset, CoMA
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx
from src.BAMotifs import BaseBAMotifs
from src.MolecularDataset import MolecularDataset

def save_tsneplot(y_true, emb, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(np.array(emb))
    plt.figure(figsize=(4, 4))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_true, alpha=0.9)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
    plt.title("t-SNE of Graph-level Embeddings", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()

def save_tsne_comparison_plot(y_true, baseline_embs, pos_embs, neg_embs, save_path):
    embeddings = [baseline_embs, pos_embs, neg_embs]
    titles = ["Baseline", "Pos Predictor", "Neg Predictor"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, emb, title in zip(axes, embeddings, titles):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        emb_2d = tsne.fit_transform(np.array(emb))
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_true, alpha=0.9)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()

def save_regplot(y_true, y_pred, save_path, title="Regression Results"):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    r2 = r2_score(y_true, y_pred)
    slope = np.sum(y_true * y_pred) / np.sum(y_true ** 2)
    x_vals = np.array([y_true.min(), y_true.max()])
    y_vals = slope * x_vals
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, markers='x', color='blue')
    plt.plot(x_vals, x_vals, 'k--', lw=2)
    plt.plot(x_vals, y_vals, 'k-')
    plt.text(0.05, 0.95, f"$R^2$ = {r2:.3f}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')
    plt.xlabel("True values", fontsize=12)
    plt.ylabel("Predicted values", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_regression_comparison_plot(y_true, baseline_preds, pos_preds, neg_preds, save_path):
    def single_regplot(ax, y_true, y_pred, color, title, show_ylabel=True, xlim=None):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        ax.scatter(y_pred, y_true, s=20, alpha=0.5, color=color)

        slope = np.sum(y_true * y_pred) / np.sum(y_pred ** 2)
        x_vals = np.array([y_pred.min(), y_pred.max()])
        y_vals = slope * x_vals
        y_min, y_max = y_true.min(), y_true.max()
        ax.plot(x_vals, y_vals, color=color, linewidth=2.0)
        ax.plot(x_vals, x_vals, 'k--', linewidth=1.5)

        r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{title}\n$R^2={r2:.3f}$", fontsize=16, pad=12)
        ax.set_xlabel("Predicted Value", fontsize=16)
        if show_ylabel:
            ax.set_ylabel("True Value", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
        if not show_ylabel:
            ax.tick_params(axis="y", labelleft=False)

        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim([x_vals.min(), x_vals.max()])
        ax.set_ylim([y_min, y_max])
        ax.grid(True, alpha=0.5)

    y_true_all = np.array(y_true).flatten()
    y_min_global = y_true_all.min()
    y_max_global = y_true_all.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    single_regplot(axes[0], y_true, baseline_preds, "#005BBB", "Target GNN Reg. Plot", xlim=[-1, 9])
    single_regplot(axes[1], y_true, pos_preds, "#009933", "Pos. Predictor Reg. Plot", show_ylabel=False, xlim=[-1, 9])
    single_regplot(axes[2], y_true, [-neg_pred for neg_pred in neg_preds], "#CC0000", "Neg. Predictor Reg. Plot", show_ylabel=False, xlim=[-1, 17])

    for ax in axes:
        ax.set_ylim([y_min_global, y_max_global])

    plt.subplots_adjust(wspace=0.0, left=0.08, right=0.95, top=0.9, bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_predictions(dataset, test_set, test_loader, explainer, log_path, enabled):
    plt.rcParams.update({"figure.dpi": 120})
    if not enabled:
        return

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
                    rotation = np.array([
                        [c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]
                    ])
                elif axis == "y":
                    rotation = np.array([
                        [c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]
                    ])
                elif axis == "x":
                    rotation = np.array([
                        [1, 0, 0],
                        [0, c, -s],
                        [0, s, c]
                    ])
                return points @ rotation.T

            points = data.pos.cpu().numpy()
            points = rotate(points, angle_deg=-15, axis="y")
            faces = data.face.t().cpu().numpy()
            selected_vertices = set(np.flatnonzero(y_masks[idx][:, 0].astype(bool)))
            colors = np.ones((points.shape[0], 3)) * 0.6
            if len(selected_vertices) > 0:
                sel_idx = np.array(list(selected_vertices))
                colors[sel_idx] = np.array([0.0, 1.0, 0.0])
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')
            mesh_tris = []
            face_colors = []
            for f in faces:
                mesh_tris.append(points[f])
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
            ax.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2])
            ax.set_axis_off()
            ax.view_init(elev=90, azim=-90)
            filename = f"{log_path}/graph_{true_class}_{pos_pred_class}_{neg_pred_class}_{idx}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(filename)
        else:
            edge_attrs = ["edge_attr"] if data.edge_attr is not None else None
            graph = to_networkx(data, to_undirected=True, edge_attrs=edge_attrs)
            selected_nodes = set(np.flatnonzero(y_masks[idx][:, 0].astype(bool)))
            unselected_nodes = [node for node in graph.nodes if node not in selected_nodes]
            fig, ax = plt.subplots(figsize=(6, 6))
            if isinstance(dataset, BaseBAMotifs):
                pos = nx.kamada_kawai_layout(graph)
                nx.draw_networkx_edges(graph, pos, ax=ax, width=1.5, edge_color="#555555", alpha=0.8)
                nx.draw_networkx_nodes(graph, pos, nodelist=unselected_nodes, node_color="#FFD580", edgecolors="#CC6600", linewidths=1.8, node_size=600, ax=ax)
                nx.draw_networkx_nodes(graph, pos, nodelist=list(selected_nodes), node_color="#8B0000", edgecolors="#25D025", linewidths=3.6, node_size=650, ax=ax)
            elif isinstance(dataset, MolecularDataset):
                pos = nx.kamada_kawai_layout(graph)
                atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*']
                atom_idx = data.x.argmax(dim=1).cpu().numpy()
                labels = {i: atom_types[j] for i, j in enumerate(atom_idx)}
                edge_width = [np.array(attr["edge_attr"]).argmax(-1) + 1 for _, _, attr in graph.edges(data=True)]
                nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_width, edge_color="black", alpha=0.9)

                gt_nodes = [i for i, t in enumerate(data.true.cpu().numpy()) if t == 1]
                non_gt_nodes = [i for i, t in enumerate(data.true.cpu().numpy()) if t == 0]
                nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=non_gt_nodes, node_color="white", node_size=500, linewidths=0)
                nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=gt_nodes, node_color="#CF8D89", edgecolors="black", linewidths=1.2, node_size=600)
                nx.draw_networkx_labels(graph, pos, labels=labels, font_size=20, font_color="black")
                nx.draw_networkx_nodes(graph, pos, nodelist=list(selected_nodes), node_color="none", edgecolors="#25D025", linewidths=3.6, node_size=650, ax=ax)
            elif isinstance(dataset, GNNBenchmarkDataset):
                pos = data.pos.cpu().numpy()
                pos = np.array([[y, -x] for x, y in pos])
                nx.draw_networkx_edges(graph, pos, ax=ax, width=1.2, edge_color="#444444", alpha=0.4)
                node_color = data.x.cpu().numpy()[:, 0]
                nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_color=node_color, cmap="gray_r", node_size=350, edgecolors="black", linewidths=0.5, ax=ax)
                nx.draw_networkx_nodes(graph, pos, nodelist=list(selected_nodes), node_color="none", edgecolors="#25D025", linewidths=3.6, node_size=500, ax=ax)
            ax.set_title(f"graph {idx} | y={true_class} | pos={pos_pred_class} | neg={neg_pred_class}", fontsize=16)
            ax.axis("off")
            fig.tight_layout()
            filename = f'{log_path}/graph_{true_class}_{pos_pred_class}_{neg_pred_class}_{idx}.png'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(filename)

def log_metrics(logging, metrics, mode, epoch=None):
    if epoch is not None:
        msg = f"Epoch: {epoch}, {mode} "
    else:
        msg = f"{mode} "
    msg += ', '.join(f"{k}: {v:.4f}" for k, v in metrics.items())
    logging.info(msg)

def log_metrics_tb(writer, metrics, mode, epoch):
    for k, v in metrics.items():
        tag = f"{mode}_{k}"
        writer.add_scalar(tag, v, epoch)

def apply_mask(data, mask):
    mask = mask.squeeze().bool()
    node_idx = mask.nonzero().view(-1)
    edge_index, edge_attr = subgraph(
        node_idx,
        data.edge_index,
        data.edge_attr,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    return Data(
        x=data.x[node_idx] if data.x is not None else None,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=data.batch[node_idx] if data.batch is not None else None,
        batch_size=data.batch.max().item() + 1 if data.batch is not None else None,
        y=data.y,
    )

def tensor_to_list(tensor):
    result = []
    for i in range(len(tensor)):
        result.append(tensor[i].detach().cpu().numpy())
    return result

def tensor_batch_to_list(tensor, batch):
    result = []
    for i in range(batch.max() + 1):
        result.append(tensor[batch == i].detach().cpu().numpy())
    return result
