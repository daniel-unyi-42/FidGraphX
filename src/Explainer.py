import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from src.GNN import GNBlock, MLPBlock
from src.utils import apply_mask, tensor_to_list, tensor_batch_to_list
from src.metrics import (
    fid_plus_prob, fid_minus_prob,
    fid_plus_acc, fid_minus_acc,
    fid_plus_tv, fid_minus_tv,
    fid_minus_reg, fid_plus_reg,
    precision_score, recall_score, f1_score,
    roc_auc_score, pr_auc_score
)

class Explainer(nn.Module):
    def __init__(self, baseline, pos_predictor, neg_predictor, sparsity, reward_coeff):
        super(Explainer, self).__init__()
        self.task_type = baseline.task_type
        self.sparsity = sparsity
        self.reward_coeff = reward_coeff
        self.baseline = baseline
        self.pos_predictor = pos_predictor
        self.neg_predictor = neg_predictor
        self.conv_type = baseline.conv_type
        self.learning_rate = baseline.learning_rate
        self.in_channels = baseline.in_channels
        self.num_layers = baseline.num_layers
        self.hidden_channels = baseline.hidden_channels
        self.edge_dim = baseline.edge_dim
        self.use_norm = baseline.use_norm
        self.convs = nn.ModuleList()
        self.convs.append(GNBlock(
            self.conv_type,
            self.in_channels,
            self.hidden_channels,
            self.hidden_channels,
            self.edge_dim,
            self.use_norm
        ))
        for _ in range(self.num_layers - 1):
            self.convs.append(GNBlock(
                self.conv_type,
                self.hidden_channels,
                self.hidden_channels,
                self.hidden_channels,
                self.edge_dim,
                self.use_norm
            ))
        self.head = MLPBlock(self.hidden_channels, self.hidden_channels, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = self.loss
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device != torch.device('cuda'):
            print('WARNING: GPU not available. Using CPU instead.')
        self.to(self.device)
    
    def loss(self, reward, y_true, y_pred, batch, batch_size):
        # the reward is calculated for each graph
        # binary cross-entropy between node selection probabilities and the node selection mask, averaged over the nodes for each graph
        cross_entropy = gnn.global_mean_pool(
            -torch.sum(y_true * torch.log(y_pred + 1e-8) + (1 - y_true) * torch.log(1 - y_pred + 1e-8), dim=1),
            batch, size=batch_size
        )
        # L1 norm of the probability so that the model learns to use as few nodes as possible
        # instead of calculation for each graph, we calculate the mean over the whole batch
        L1_norm = torch.mean(y_pred) # gnn.global_mean_pool(y_pred, batch)[batch]
        assert 0 <= L1_norm <= 1
        # the selection budget is the difference between the sparsity target and the L1 norm
        selection_budget = torch.abs(self.sparsity - L1_norm)
        # the custom actor loss is the sum of the reward and the selection budget
        custom_actor_loss = reward * cross_entropy + self.reward_coeff * selection_budget
        return torch.mean(custom_actor_loss)

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)
        x = self.head(x)
        return torch.sigmoid(x)
    
    def random_forward(self, data):
        return torch.rand((data.num_nodes, 1), device=self.device)
    
    def train_batch(self, loader):
        self.baseline.eval()
        metrics = {
            'explainer_loss': 0.0,
            'sparsity': 0.0,
            'pos_loss': 0.0,
            'pos_metric': 0.0,
            'neg_loss': 0.0,
            'neg_metric': 0.0,
            'fidplus_prob': 0.0,
            'fidminus_prob': 0.0,
            'fidplus_acc': 0.0,
            'fidminus_acc': 0.0,
            'fidplus': 0.0,
            'fidminus': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'pr_auc': 0.0,
            'roc_auc': 0.0
        }
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                baseline_logits = self.baseline(data)
                if self.task_type == 'classification':
                    target = baseline_logits.softmax(dim=1)
                else:
                    target = baseline_logits
            # train pos_predictor and neg_predictor
            self.eval()
            self.pos_predictor.train()
            self.pos_predictor.optimizer.zero_grad()
            self.neg_predictor.train()
            self.neg_predictor.optimizer.zero_grad()
            with torch.no_grad():
                probs = self(data)
                mask = torch.bernoulli(probs)
            pos_logits = self.pos_predictor(apply_mask(data, mask))
            pos_loss = self.pos_predictor.criterion(pos_logits, target, reduction='none')
            pos_loss.mean().backward()
            self.pos_predictor.optimizer.step()
            neg_logits = self.neg_predictor(apply_mask(data, 1.0 - mask))
            neg_loss = self.neg_predictor.criterion(neg_logits, target, reduction='none')
            neg_loss.mean().backward()
            self.neg_predictor.optimizer.step()
            with torch.no_grad():
                reward = -(pos_loss - neg_loss)
                reward = (reward - reward.mean()) / (reward.std() + 1e-8)
                self_loss = self.criterion(reward, mask, probs, data.batch, data.batch_size)
            # train explainer
            self.pos_predictor.eval()
            self.neg_predictor.eval()
            self.train()
            self.optimizer.zero_grad()
            with torch.no_grad():
                probs = self(data)
                mask = torch.bernoulli(probs)
                pos_logits = self.pos_predictor(apply_mask(data, mask))
                neg_logits = self.neg_predictor(apply_mask(data, 1.0 - mask))
                pos_loss = self.pos_predictor.criterion(pos_logits, target, reduction='none')
                neg_loss = self.neg_predictor.criterion(neg_logits, target, reduction='none')
                reward = -(pos_loss - neg_loss)
                reward = (reward - reward.mean()) / (reward.std() + 1e-8)
            probs = self(data)
            self_loss = self.criterion(reward, mask, probs, data.batch, data.batch_size)
            self_loss.backward()
            self.optimizer.step()
            # record metrics
            metrics['explainer_loss'] += self_loss.item()
            metrics['sparsity'] += mask.mean().item()
            metrics['pos_loss'] += pos_loss.mean().item()
            metrics['pos_metric'] += self.pos_predictor.metric(pos_logits, target.argmax(dim=1)).item()
            metrics['neg_loss'] += neg_loss.mean().item()
            metrics['neg_metric'] += self.neg_predictor.metric(neg_logits, target.argmax(dim=1)).item()
            if self.task_type == 'classification':
                metrics['fidplus_prob'] += fid_plus_prob(neg_logits, baseline_logits)
                metrics['fidminus_prob'] += fid_minus_prob(pos_logits, baseline_logits)
                metrics['fidplus_acc'] += fid_plus_acc(neg_logits, baseline_logits)
                metrics['fidminus_acc'] += fid_minus_acc(pos_logits, baseline_logits)
                metrics['fidplus'] += fid_plus_tv(neg_logits, baseline_logits)
                metrics['fidminus'] += fid_minus_tv(pos_logits, baseline_logits)
            else:
                metrics['fidplus'] += fid_plus_reg(neg_logits, baseline_logits)
                metrics['fidminus'] += fid_minus_reg(pos_logits, baseline_logits)
            if hasattr(data, 'true'):
                metrics['precision'] += precision_score(mask, data.true)
                metrics['recall'] += recall_score(mask, data.true)
                metrics['f1'] += f1_score(mask, data.true)
                metrics['pr_auc'] += pr_auc_score(mask, data.true)
                metrics['roc_auc'] += roc_auc_score(probs, data.true)
        for metric_name in metrics:
            metrics[metric_name] /= len(loader)
        return metrics

    @torch.no_grad()
    def evaluate_batch(self, loader, random=False):
        self.pos_predictor.eval()
        self.neg_predictor.eval()
        self.baseline.eval()
        self.eval()
        metrics = {
            'explainer_loss': 0.0,
            'sparsity': 0.0,
            'pos_loss': 0.0,
            'pos_metric': 0.0,
            'neg_loss': 0.0,
            'neg_metric': 0.0,
            'fidplus_prob': 0.0,
            'fidminus_prob': 0.0,
            'fidplus_acc': 0.0,
            'fidminus_acc': 0.0,
            'fidplus': 0.0,
            'fidminus': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'pr_auc': 0.0,
            'roc_auc': 0.0
        }
        for data in loader:
            data = data.to(self.device)
            baseline_logits = self.baseline(data)
            target = baseline_logits.softmax(dim=1) if self.task_type == 'classification' else baseline_logits
            probs = self(data) if not random else self.random_forward(data)
            mask = (probs > 0.5).float()
            # k = max(1, int(self.sparsity * probs.numel()))
            # threshold = torch.topk(probs.squeeze(-1), k).values.min()
            # mask = (probs >= threshold).float()
            pos_logits = self.pos_predictor(apply_mask(data, mask))
            neg_logits = self.neg_predictor(apply_mask(data, 1.0 - mask))
            pos_loss = self.pos_predictor.criterion(pos_logits, target, reduction='none')
            neg_loss = self.neg_predictor.criterion(neg_logits, target, reduction='none')
            reward = -(pos_loss - neg_loss)
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)
            self_loss = self.criterion(reward, mask, probs, data.batch, data.batch_size)
            metrics['explainer_loss'] += self_loss.item()
            metrics['sparsity'] += mask.mean().item()
            metrics['pos_loss'] += pos_loss.mean().item()
            metrics['pos_metric'] += self.pos_predictor.metric(pos_logits, target.argmax(dim=1)).item()
            metrics['neg_loss'] += neg_loss.mean().item()
            metrics['neg_metric'] += self.neg_predictor.metric(neg_logits, target.argmax(dim=1)).item()
            if self.task_type == 'classification':
                metrics['fidplus_prob'] += fid_plus_prob(neg_logits, baseline_logits)
                metrics['fidminus_prob'] += fid_minus_prob(pos_logits, baseline_logits)
                metrics['fidplus_acc'] += fid_plus_acc(neg_logits, baseline_logits)
                metrics['fidminus_acc'] += fid_minus_acc(pos_logits, baseline_logits)
                metrics['fidplus'] += fid_plus_tv(neg_logits, baseline_logits)
                metrics['fidminus'] += fid_minus_tv(pos_logits, baseline_logits)
            else:
                metrics['fidplus'] += fid_plus_reg(neg_logits, baseline_logits)
                metrics['fidminus'] += fid_minus_reg(pos_logits, baseline_logits)
            if hasattr(data, 'true'):
                metrics['precision'] += precision_score(mask, data.true)
                metrics['recall'] += recall_score(mask, data.true)
                metrics['f1'] += f1_score(mask, data.true)
                metrics['pr_auc'] += pr_auc_score(mask, data.true)
                metrics['roc_auc'] += roc_auc_score(probs, data.true)
        for metric_name in metrics:
            metrics[metric_name] /= len(loader)
        return metrics

    @torch.no_grad()
    def explain_batch(self, loader, random=False):
        self.pos_predictor.eval()
        self.neg_predictor.eval()
        self.eval()
        y_probs, y_masks, explanations = [], [], []
        pos_embs, neg_embs = [], []
        pos_preds, neg_preds = [], []
        y_trues = []
        for data in loader:
            data = data.to(self.device)
            probs = self(data) if not random else self.random_forward(data)
            mask = (probs > 0.5).float()
            # k = max(1, int(self.sparsity * probs.numel()))
            # threshold = torch.topk(probs.squeeze(-1), k).values.min()
            # mask = (probs >= threshold).float()
            y_probs += tensor_batch_to_list(probs, data.batch)
            y_masks += tensor_batch_to_list(mask, data.batch)
            if hasattr(data, 'true'):
                explanations += tensor_batch_to_list(data.true, data.batch)
            else:
                explanations += tensor_batch_to_list(torch.zeros_like(mask), data.batch)
            pos_emb = self.pos_predictor.embed(apply_mask(data, mask))
            pos_pred = self.pos_predictor.head(pos_emb)
            neg_emb = self.neg_predictor.embed(apply_mask(data, 1.0 - mask))
            neg_pred = self.neg_predictor.head(neg_emb)
            if self.task_type == 'classification':
                pos_pred = F.softmax(pos_pred, dim=1)
                neg_pred = F.softmax(neg_pred, dim=1)
            pos_embs += tensor_to_list(pos_emb)
            pos_preds += tensor_to_list(pos_pred)
            neg_embs += tensor_to_list(neg_emb)
            neg_preds += tensor_to_list(neg_pred)
            y_trues += tensor_to_list(data.y)
        return y_probs, y_masks, explanations, \
            pos_embs, pos_preds, neg_embs, neg_preds, y_trues
