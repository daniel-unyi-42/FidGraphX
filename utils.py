from torch_geometric.data import Data

def log_metrics(logging, metrics, mode, epoch=None):
    if epoch is not None:
        msg = f"Epoch: {epoch}, {mode} "
    else:
        msg = f"{mode} "
    msg += ', '.join(f"{k}: {v:.4f}" for k, v in metrics.items())
    logging.info(msg)

def log_metrics_tb(writer, metrics, mode, epoch):
    for k, v in metrics.items():
        tag = f"EXPLAINER/{mode}_{k}"
        writer.add_scalar(tag, v, epoch)

def apply_mask(data, mask):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    x = (x * mask).float()
    mask = mask.squeeze().bool()
    edge_mask = (mask[edge_index[0]]) & (mask[edge_index[1]])
    edge_index = edge_index[:, edge_mask]
    if edge_attr is not None:
        edge_attr = edge_attr[edge_mask]
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=data.batch, y=data.y)

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
