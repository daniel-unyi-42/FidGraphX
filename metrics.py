import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score as f1_score_sklearn
from sklearn.metrics import precision_score as precision_score_sklearn
from sklearn.metrics import recall_score as recall_score_sklearn
from sklearn.metrics import jaccard_score as jaccard_score_sklearn
from sklearn.metrics import roc_auc_score as roc_auc_score_sklearn
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Fidelity metrics should support regression tasks as well

def fid_plus_prob(neg_preds, baseline_preds):
    with torch.no_grad():
        neg_preds = torch.softmax(neg_preds, dim=1)
        baseline_preds = torch.softmax(baseline_preds, dim=1)
        idx = baseline_preds.argmax(1, keepdim=True)
        fid_plus_prob = (baseline_preds.gather(1, idx) - neg_preds.gather(1, idx)).abs().mean()
    return fid_plus_prob

def fid_minus_prob(pos_preds, baseline_preds):
    with torch.no_grad():
        pos_preds = torch.softmax(pos_preds, dim=1)
        baseline_preds = torch.softmax(baseline_preds, dim=1)
        idx = baseline_preds.argmax(1, keepdim=True)
        fid_minus_prob = (baseline_preds.gather(1, idx) - pos_preds.gather(1, idx)).abs().mean()
    return fid_minus_prob

def fid_plus_class(neg_preds, baseline_preds):
    return 1.0 - (neg_preds.argmax(axis=1) == baseline_preds.argmax(axis=1)).float().mean()

def fid_minus_class(pos_preds, baseline_preds):
    return 1.0 - (pos_preds.argmax(axis=1) == baseline_preds.argmax(axis=1)).float().mean()

def fid_plus_reg(neg_preds, baseline_preds):
    return F.mse_loss(neg_preds, baseline_preds, reduction="mean").item()

def fid_minus_reg(pos_preds, baseline_preds):
    return F.mse_loss(pos_preds, baseline_preds, reduction="mean").item()

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

def save_cmplot(y_true, y_pred, save_path):
    y_pred = [pred.argmax() for pred in y_pred]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def save_regplot(y_true, y_pred, save_path):
    plt.scatter(y_true, y_pred)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.savefig(save_path)
    plt.close()

def save_tsneplot(y_true, emb, save_path):
    tsne = TSNE(n_components=2)
    emb_2d = tsne.fit_transform(np.array(emb))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_true)
    plt.colorbar(scatter)
    plt.title("t-SNE of Graph-level Embeddings")
    plt.savefig(save_path)
    plt.close()
