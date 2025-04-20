import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from GNN import GNBlock, MLPBlock
from utilities import apply_mask, tensor_to_list, tensor_batch_to_list, fid_plus_prob, fid_minus_prob, fid_plus_acc, fid_minus_acc

class Selector(nn.Module):
    def __init__(self, baseline, pos_predictor, neg_predictor, sparsity, reward_coeff):
        super(Selector, self).__init__()
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
        self.conv0 = GNBlock(
            self.conv_type,
            self.in_channels,
            self.hidden_channels,
            self.hidden_channels,
            self.edge_dim,
            self.use_norm
        )
        for i in range(1, self.num_layers):
            setattr(self, f'conv{i}', GNBlock(
                self.conv_type,
                self.hidden_channels,
                self.hidden_channels,
                self.hidden_channels,
                self.edge_dim,
                self.use_norm
            ))
        self.head = MLPBlock(self.hidden_channels, self.hidden_channels, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device != torch.device('cuda'):
            print('WARNING: GPU not available. Using CPU instead.')
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = self.loss
    
    def loss(self, reward, y_true, y_pred, batch):
        # the reward is calculated for each graph
        # binary cross-entropy between node selection probabilities and the node selection mask, averaged over the nodes for each graph
        cross_entropy = gnn.global_mean_pool(
            -torch.sum(y_true * torch.log(y_pred + 1e-8) + (1 - y_true) * torch.log(1 - y_pred + 1e-8), dim=1),
            batch
        )
        # L1 norm of the probability so that the model learns to use as few nodes as possible
        # instead of calculation for each graph, we calculate the mean over the whole batch
        L1_norm = torch.mean(y_pred) # gnn.global_mean_pool(y_pred, batch)[batch]
        assert 0 <= L1_norm <= 1
        # the selection budget is the difference between the sparsity target and the L1 norm
        selection_budget = torch.abs(self.sparsity - L1_norm)
        # the custom actor loss is the sum of the reward and the selection budget
        custom_actor_loss = (reward + self.reward_coeff * selection_budget) * cross_entropy
        return torch.mean(custom_actor_loss)

    def forward(self, data):
        x = data.x
        for i in range(self.num_layers):
            x = getattr(self, f'conv{i}')(x, data.edge_index, data.edge_attr)
        x = self.head(x)
        return x

    def train_batch(self, loader, train_pred=True, train_sel=True):
        self.baseline.eval()
        self_losses, sparsities = [], []
        pos_losses, neg_losses, pos_metrics, neg_metrics = [], [], [], []
        fid_plus_probs, fid_minus_probs, fid_plus_accs, fid_minus_accs = [], [], [], []
        for data in loader:
            data = data.to(self.device)
            if self.task_type == 'regression':
                data.y = data.y.unsqueeze(1)
            # train pos_predictor and neg_predictor
            if train_pred:
                self.eval()
                self.pos_predictor.train()
                self.pos_predictor.optimizer.zero_grad()
                self.neg_predictor.train()
                self.neg_predictor.optimizer.zero_grad()
                with torch.no_grad():
                    probs = torch.sigmoid(self(data))
                    mask = torch.bernoulli(probs)
                pos_logits = self.pos_predictor(apply_mask(data, mask))
                pos_loss = self.pos_predictor.criterion(pos_logits, data.y)
                pos_loss.backward()
                self.pos_predictor.optimizer.step()
                neg_logits = self.neg_predictor(apply_mask(data, 1.0 - mask))
                neg_loss = self.neg_predictor.criterion(neg_logits, data.y)
                neg_loss.backward()
                self.neg_predictor.optimizer.step()
                with torch.no_grad():
                    reward = -(pos_loss - neg_loss)
                    self_loss = self.criterion(reward, mask, probs, data.batch)
            # train selector
            if train_sel:
                self.pos_predictor.eval()
                self.neg_predictor.eval()
                self.train()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    probs = torch.sigmoid(self(data))
                    mask = torch.bernoulli(probs)
                    pos_logits = self.pos_predictor(apply_mask(data, mask))
                    neg_logits = self.neg_predictor(apply_mask(data, 1.0 - mask))
                    pos_loss = self.pos_predictor.criterion(pos_logits, data.y, reduction='none')
                    neg_loss = self.neg_predictor.criterion(neg_logits, data.y, reduction='none')
                    reward = -(pos_loss - neg_loss)
                probs = torch.sigmoid(self(data))
                self_loss = self.criterion(reward, mask, probs, data.batch)
                self_loss.backward()
                self.optimizer.step()
            # record metrics
            self_losses.append(self_loss.item())
            sparsities.append(mask.mean().item())
            pos_losses.append(pos_loss.mean().item())
            neg_losses.append(neg_loss.mean().item())
            pos_metric = self.pos_predictor.metric(pos_logits, data.y)
            neg_metric = self.neg_predictor.metric(neg_logits, data.y)
            pos_metrics.append(pos_metric.item())
            neg_metrics.append(neg_metric.item())
            with torch.no_grad():
                baseline_preds = self.baseline(data)
            fid_plus_prob_metric = fid_plus_prob(neg_logits, baseline_preds)
            fid_minus_prob_metric = fid_minus_prob(pos_logits, baseline_preds)
            fid_plus_acc_metric = fid_plus_acc(neg_logits, baseline_preds)
            fid_minus_acc_metric = fid_minus_acc(pos_logits, baseline_preds)
            fid_plus_probs.append(fid_plus_prob_metric)
            fid_minus_probs.append(fid_minus_prob_metric)
            fid_plus_accs.append(fid_plus_acc_metric)
            fid_minus_accs.append(fid_minus_acc_metric)
        return sum(self_losses) / len(self_losses), \
            sum(sparsities) / len(sparsities), \
                sum(pos_losses) / len(pos_losses), \
                    sum(neg_losses) / len(neg_losses), \
                        sum(pos_metrics) / len(pos_metrics), \
                            sum(neg_metrics) / len(neg_metrics), \
                                sum(fid_plus_probs) / len(fid_plus_probs), \
                                    sum(fid_minus_probs) / len(fid_minus_probs), \
                                        sum(fid_plus_accs) / len(fid_plus_accs), \
                                            sum(fid_minus_accs) / len(fid_minus_accs)

    @torch.no_grad()
    def test_batch(self, loader):
        self.pos_predictor.eval()
        self.neg_predictor.eval()
        self.baseline.eval()
        self.eval()
        self_losses, sparsities = [], []
        pos_losses, neg_losses, pos_metrics, neg_metrics = [], [], [], []
        fid_plus_probs, fid_minus_probs, fid_plus_accs, fid_minus_accs = [], [], [], []
        for data in loader:
            data = data.to(self.device)
            if self.task_type == 'regression':
                data.y = data.y.unsqueeze(1)
            probs = torch.sigmoid(self(data))
            mask = (probs > 0.5).float()
            pos_logits = self.pos_predictor(apply_mask(data, mask))
            neg_logits = self.neg_predictor(apply_mask(data, 1.0 - mask))
            pos_loss = self.pos_predictor.criterion(pos_logits, data.y, reduction='none')
            neg_loss = self.neg_predictor.criterion(neg_logits, data.y, reduction='none')
            reward = -(pos_loss - neg_loss)
            self_loss = self.criterion(reward, mask, probs, data.batch)
            self_losses.append(self_loss.item())
            sparsities.append(mask.mean().item())
            pos_losses.append(pos_loss.mean().item())
            neg_losses.append(neg_loss.mean().item())
            pos_metric = self.pos_predictor.metric(pos_logits, data.y)
            neg_metric = self.neg_predictor.metric(neg_logits, data.y)
            pos_metrics.append(pos_metric.item())
            neg_metrics.append(neg_metric.item())
            baseline_preds = self.baseline(data)
            fid_plus_prob_metric = fid_plus_prob(neg_logits, baseline_preds)
            fid_minus_prob_metric = fid_minus_prob(pos_logits, baseline_preds)
            fid_plus_acc_metric = fid_plus_acc(neg_logits, baseline_preds)
            fid_minus_acc_metric = fid_minus_acc(pos_logits, baseline_preds)
            fid_plus_probs.append(fid_plus_prob_metric)
            fid_minus_probs.append(fid_minus_prob_metric)
            fid_plus_accs.append(fid_plus_acc_metric)
            fid_minus_accs.append(fid_minus_acc_metric)
        return sum(self_losses) / len(self_losses), \
            sum(sparsities) / len(sparsities), \
                sum(pos_losses) / len(pos_losses), \
                    sum(neg_losses) / len(neg_losses), \
                        sum(pos_metrics) / len(pos_metrics), \
                            sum(neg_metrics) / len(neg_metrics), \
                                sum(fid_plus_probs) / len(fid_plus_probs), \
                                    sum(fid_minus_probs) / len(fid_minus_probs), \
                                        sum(fid_plus_accs) / len(fid_plus_accs), \
                                            sum(fid_minus_accs) / len(fid_minus_accs)

    @torch.no_grad()
    def predict_batch(self, loader):
        self.pos_predictor.eval()
        self.neg_predictor.eval()
        self.baseline.eval()
        self.eval()
        y_probs, y_masks, explanations = [], [], []
        pos_preds, neg_preds, baseline_preds, y_trues = [], [], [], []
        for data in loader:
            data = data.to(self.device)
            if self.task_type == 'regression':
                data.y = data.y.unsqueeze(1)
            probs = torch.sigmoid(self(data))
            mask = (probs > 0.5).float()
            y_probs += tensor_batch_to_list(probs, data.batch)
            y_masks += tensor_batch_to_list(mask, data.batch)
            explanations += tensor_batch_to_list(data.true, data.batch)
            pos_logits = self.pos_predictor(apply_mask(data, mask))
            neg_logits = self.neg_predictor(apply_mask(data, 1.0 - mask))
            baseline_logits = self.baseline(data)
            if self.task_type == 'classification':
                pos_pred = F.softmax(pos_logits, dim=1)
                neg_pred = F.softmax(neg_logits, dim=1)
                baseline_pred = F.softmax(baseline_logits, dim=1)
            elif self.task_type == 'regression':
                pos_pred = pos_logits
                neg_pred = neg_logits
                baseline_pred = baseline_logits
            pos_preds += tensor_to_list(pos_pred)
            neg_preds += tensor_to_list(neg_pred)
            baseline_preds += tensor_to_list(baseline_pred)
            y_trues += tensor_to_list(data.y)
        return y_probs, y_masks, explanations, pos_preds, neg_preds, baseline_preds, y_trues
