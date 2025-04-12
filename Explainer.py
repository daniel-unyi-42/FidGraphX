import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from GNN import GNBlock, MLPBlock
from torch_geometric.data import Data


def random_mask(data):
    probs = torch.rand((data.num_nodes, 1), device=data.x.device)
    # rand_value = torch.rand((1,), device=data.x.device)
    mask = (probs > 0.5).detach().float() # 0.05?
    return mask

def apply_mask(data, mask):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    x = (x * mask).float()
    mask = mask.squeeze().bool()
    edge_mask = (mask[edge_index[0]]) & (mask[edge_index[1]])
    edge_index = edge_index[:, edge_mask]
    if edge_attr is not None:
        edge_attr = edge_attr[edge_mask]
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=data.batch, y=data.y)



class Selector(nn.Module):
    def __init__(self, pos_predictor, neg_predictor, sparsity, reward_coeff):
        super(Selector, self).__init__()
        self.sparsity = sparsity
        self.reward_coeff = reward_coeff
        self.pos_predictor = pos_predictor
        self.neg_predictor = neg_predictor
        self.conv_type = pos_predictor.conv_type
        self.learning_rate = pos_predictor.learning_rate
        self.in_channels = pos_predictor.in_channels
        self.num_layers = pos_predictor.num_layers
        self.hidden_channels = pos_predictor.hidden_channels
        self.edge_dim = pos_predictor.edge_dim
        self.use_norm = pos_predictor.use_norm
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
        selection_budget = L1_norm # torch.abs(self.sparsity - L1_norm)
        # the custom actor loss is the sum of the reward and the selection budget
        custom_actor_loss = (reward + self.reward_coeff * selection_budget) * cross_entropy
        return torch.mean(custom_actor_loss)

    def forward(self, data):
        x = data.x
        for i in range(self.num_layers):
            x = getattr(self, f'conv{i}')(x, data.edge_index, data.edge_attr)
        x = self.head(x)
        return x

    def train_batch(self, loader, train_sel=True, train_pred=True):
        pos_losses = []
        neg_losses = []
        sel_losses = []
        mask_sizes = []
        pos_metrics = []
        neg_metrics = []
        for data in loader:
            data = data.to(self.device)
            if self.pos_predictor.task_type == 'regression':
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
                    sel_loss = self.loss(reward, mask, probs, data.batch)
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
                sel_loss = self.criterion(reward, mask, probs, data.batch)
                sel_loss.backward()
                self.optimizer.step()
            # record metrics
            sel_losses.append(sel_loss.item())
            pos_losses.append(pos_loss.mean().item())
            neg_losses.append(neg_loss.mean().item())
            mask_sizes.append(torch.sum(mask).item() / len(mask))
            pos_metric = self.pos_predictor.metric(pos_logits, data.y)
            neg_metric = self.neg_predictor.metric(neg_logits, data.y)
            pos_metrics.append(pos_metric.item())
            neg_metrics.append(neg_metric.item())
        return sum(sel_losses) / len(sel_losses), \
            sum(pos_losses) / len(pos_losses), \
                sum(neg_losses) / len(neg_losses), \
                    sum(mask_sizes) / len(mask_sizes), \
                        sum(pos_metrics) / len(pos_metrics), \
                            sum(neg_metrics) / len(neg_metrics)

    @torch.no_grad()
    def test_batch(self, loader, sparsity=None):
        self.pos_predictor.eval()
        self.neg_predictor.eval()
        self.eval()
        pos_losses = []
        neg_losses = []
        sel_losses = []
        mask_sizes = []
        pos_metrics = []
        neg_metrics = []
        for data in loader:
            data = data.to(self.device)
            if self.pos_predictor.task_type == 'regression':
                data.y = data.y.unsqueeze(1)
            probs = torch.sigmoid(self(data))
            if sparsity is None:
                mask = (probs > 0.5).float()
            else:
                thresold = torch.quantile(probs, 1.0 - sparsity)
                mask = (probs > thresold).float()
            pos_logits = self.pos_predictor(apply_mask(data, mask))
            neg_logits = self.neg_predictor(apply_mask(data, 1.0 - mask))
            pos_loss = self.pos_predictor.criterion(pos_logits, data.y, reduction='none')
            neg_loss = self.neg_predictor.criterion(neg_logits, data.y, reduction='none')
            reward = -(pos_loss - neg_loss)
            sel_loss = self.loss(reward, mask, probs, data.batch)
            pos_losses.append(pos_loss.mean().item())
            neg_losses.append(neg_loss.mean().item())
            sel_losses.append(sel_loss.item())
            mask_sizes.append(torch.sum(mask).item() / len(mask))
            pos_metric = self.pos_predictor.metric(pos_logits, data.y)
            neg_metric = self.neg_predictor.metric(neg_logits, data.y)
            pos_metrics.append(pos_metric.item())
            neg_metrics.append(neg_metric.item())
        return sum(sel_losses) / len(sel_losses), \
            sum(pos_losses) / len(pos_losses), \
                sum(neg_losses) / len(neg_losses), \
                    sum(mask_sizes) / len(mask_sizes), \
                        sum(pos_metrics) / len(pos_metrics), \
                            sum(neg_metrics) / len(neg_metrics)

    @torch.no_grad()
    def predict_batch(self, loader, sparsity=None):
        self.pos_predictor.eval()
        self.neg_predictor.eval()
        self.eval()
        y_probs = []
        y_masks = []
        pos_preds = []
        neg_preds = []
        y_trues = []
        explanations = []
        batchlister = lambda tensor, batch: [tensor[batch == i].detach().cpu().numpy() for i in range(batch.max() + 1)]
        lister = lambda tensor: [tensor[i].detach().cpu().numpy() for i in range(len(tensor))]
        for data in loader:
            data = data.to(self.device)
            if self.pos_predictor.task_type == 'regression':
                data.y = data.y.unsqueeze(1)
            probs = torch.sigmoid(self(data))
            y_probs += batchlister(probs, data.batch)
            if sparsity is None:
                mask = (probs > 0.5).float()
            else:
                thresold = torch.quantile(probs, 1.0 - sparsity)
                mask = (probs > thresold).float()
            y_masks += batchlister(mask, data.batch)
            pos_out = F.softmax(self.pos_predictor(apply_mask(data, mask)), dim=1)
            pos_preds += lister(pos_out)
            neg_out = F.softmax(self.neg_predictor(apply_mask(data, 1.0 - mask)), dim=1)
            neg_preds += lister(neg_out)
            y_trues += lister(data.y)
            explanations += batchlister(data.true, data.batch)
        return y_probs, y_masks, pos_preds, neg_preds, y_trues, explanations
