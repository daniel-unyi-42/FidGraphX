import torch
from torch_geometric.data import Data
from sklearn.metrics import f1_score as f1_score_sklearn
from sklearn.metrics import precision_score as precision_score_sklearn
from sklearn.metrics import recall_score as recall_score_sklearn
from sklearn.metrics import jaccard_score as jaccard_score_sklearn
from sklearn.metrics import roc_auc_score as roc_auc_score_sklearn

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

# TODO: Fidelity metrics should support regression tasks as well

def fid_plus_prob(neg_preds, baseline_preds):
    neg_preds = torch.softmax(neg_preds, dim=1)
    baseline_preds = torch.softmax(baseline_preds, dim=1)
    idx = baseline_preds.argmax(1, keepdim=True)
    return (baseline_preds.gather(1, idx) - neg_preds.gather(1, idx)).abs().mean()

def fid_minus_prob(pos_preds, baseline_preds):
    pos_preds = torch.softmax(pos_preds, dim=1)
    baseline_preds = torch.softmax(baseline_preds, dim=1)
    idx = baseline_preds.argmax(1, keepdim=True)
    return (baseline_preds.gather(1, idx) - pos_preds.gather(1, idx)).abs().mean()

def fid_plus_acc(neg_preds, baseline_preds):
    return 1.0 - (neg_preds.argmax(axis=1) == baseline_preds.argmax(axis=1)).float().mean()

def fid_minus_acc(pos_preds, baseline_preds):
    return 1.0 - (pos_preds.argmax(axis=1) == baseline_preds.argmax(axis=1)).float().mean()

def precision_score(pred_explanations, true_explanations):
    pred_explanations = pred_explanations.detach().cpu().numpy()
    true_explanations = true_explanations.detach().cpu().numpy()
    return precision_score_sklearn(pred_explanations, true_explanations, average='macro', zero_division=0)

def recall_score(pred_explanations, true_explanations):
    pred_explanations = pred_explanations.detach().cpu().numpy()
    true_explanations = true_explanations.detach().cpu().numpy()
    return recall_score_sklearn(pred_explanations, true_explanations, average='macro', zero_division=0)

def f1_score(pred_explanations, true_explanations):
    pred_explanations = pred_explanations.detach().cpu().numpy()
    true_explanations = true_explanations.detach().cpu().numpy()
    return f1_score_sklearn(pred_explanations, true_explanations, average='macro')

def iou_score(pred_explanations, true_explanations):
    pred_explanations = pred_explanations.detach().cpu().numpy()
    true_explanations = true_explanations.detach().cpu().numpy()
    return jaccard_score_sklearn(pred_explanations, true_explanations, average='macro')

def auc_score(pred_explanations, true_explanations):
    pred_explanations = pred_explanations.detach().cpu().numpy()
    true_explanations = true_explanations.detach().cpu().numpy()
    return roc_auc_score_sklearn(true_explanations, pred_explanations, average='macro', multi_class='ovr')
