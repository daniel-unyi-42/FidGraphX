from torch_geometric.data import Data
from torch_geometric.utils import subgraph

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
