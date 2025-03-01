import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from GNN import GNN, GNBlock, MLPBlock
from torch_geometric.data import Data

# def motif_preserving_mask(data):
#     mask = torch.ones((data.num_nodes, 1), device=data.x.device)  # Start with all nodes unmasked
#     non_motif_nodes = (data.true == 0).nonzero(as_tuple=True)[0]  # Indices of non-motif nodes
#     probs = torch.rand((len(non_motif_nodes),), device=data.x.device)
#     mask[non_motif_nodes] = (probs > 0.5).float().unsqueeze(1)  # Mask only non-motif nodes randomly
#     return mask

# from scipy.special import comb

# num_nodes = 30
# num_motif_nodes = 5
# num_selected_nodes = 1

# bad_cases = 0
# all_cases = comb(num_nodes, num_selected_nodes)
# for i in range(min(num_motif_nodes, num_selected_nodes), 0, -1):
#     bad_cases += comb(num_motif_nodes, i) * comb(num_nodes - num_motif_nodes, num_selected_nodes - i)
#     print(f'({num_motif_nodes} {i}), ({num_nodes - num_motif_nodes}, {num_selected_nodes - i})')
#     print(f'Probability of selecting at least {i} motif nodes: {bad_cases / all_cases}')


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


class Predictor(GNN):
    def __init__(self, baseline, **kwargs):
        super(Predictor, self).__init__(**kwargs)
        self.baseline = baseline
        self.baseline.eval()

    def forward(self, data, mask=None):
        if mask is None:
            mask = random_mask(data)
        data = apply_mask(data, mask)
        return super(Predictor, self).forward(data)


class Selector(nn.Module):
    def __init__(self, predictor, sparsity, reward_coeff):
        super(Selector, self).__init__()
        self.sparsity = sparsity
        self.reward_coeff = reward_coeff
        self.predictor = predictor
        for param in self.predictor.parameters():
            param.requires_grad = False
        self.conv_type = predictor.conv_type
        self.learning_rate = predictor.learning_rate
        self.in_channels = predictor.in_channels
        self.num_layers = predictor.num_layers
        self.hidden_channels = predictor.hidden_channels
        self.edge_dim = predictor.edge_dim
        self.use_norm = predictor.use_norm
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
        # the selection budget is the difference between the sparsity target and the L1 norm
        selection_budget = torch.relu(self.sparsity - L1_norm)
        # the custom actor loss is the sum of the reward and the selection budget
        custom_actor_loss = (reward + self.reward_coeff * selection_budget) * cross_entropy
        return torch.mean(custom_actor_loss)

    def forward(self, data):
        x = data.x
        for i in range(self.num_layers):
            x = getattr(self, f'conv{i}')(x, data.edge_index, data.edge_attr)
        x = self.head(x)
        return x

    def train_batch(self, loader):
        self.predictor.eval()
        losses = []
        mask_sizes = []
        pos_metrics = []
        neg_metrics = []
        for data in loader:
            data = data.to(self.device)
            self.eval()
            with torch.no_grad():
                probs = torch.sigmoid(self(data))
                mask = torch.bernoulli(probs)
                pos_logits = self.predictor(data, mask)
                neg_logits = self.predictor(data, 1.0 - mask)
                pos_loss = self.predictor.criterion(pos_logits, data.y, reduction='none')
                neg_loss = self.predictor.criterion(neg_logits, data.y, reduction='none')
                reward = -(pos_loss - neg_loss)
            self.train()
            self.optimizer.zero_grad()
            probs = torch.sigmoid(self(data))
            loss = self.criterion(reward, mask, probs, data.batch)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            mask_sizes.append(torch.sum(mask).item() / len(mask))
            pos_metric = self.predictor.metric(pos_logits, data.y)
            neg_metric = self.predictor.metric(neg_logits, data.y)
            pos_metrics.append(pos_metric.item())
            neg_metrics.append(neg_metric.item())
        return sum(losses) / len(losses), \
            sum(mask_sizes) / len(mask_sizes), \
                sum(pos_metrics) / len(pos_metrics), \
                    sum(neg_metrics) / len(neg_metrics)

    @torch.no_grad()
    def test_batch(self, loader):
        self.predictor.eval()
        self.eval()
        losses = []
        mask_sizes = []
        pos_metrics = []
        neg_metrics = []
        for data in loader:
            data = data.to(self.device)
            probs = torch.sigmoid(self(data))
            mask = torch.bernoulli(probs) # (probs > 0.5).float()
            pos_logits = self.predictor(data, mask)
            neg_logits = self.predictor(data, 1.0 - mask)
            pos_loss = self.predictor.criterion(pos_logits, data.y, reduction='none')
            neg_loss = self.predictor.criterion(neg_logits, data.y, reduction='none')
            reward = -(pos_loss - neg_loss)
            loss = self.loss(reward, mask, probs, data.batch)
            losses.append(loss.item())
            mask_sizes.append(torch.sum(mask).item() / len(mask))
            pos_metric = self.predictor.metric(pos_logits, data.y)
            neg_metric = self.predictor.metric(neg_logits, data.y)
            pos_metrics.append(pos_metric.item())
            neg_metrics.append(neg_metric.item())
        return sum(losses) / len(losses), \
            sum(mask_sizes) / len(mask_sizes), \
                sum(pos_metrics) / len(pos_metrics), \
                    sum(neg_metrics) / len(neg_metrics)

    @torch.no_grad()
    def predict_batch(self, loader):
        self.predictor.eval()
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
            probs = torch.sigmoid(self(data))
            y_probs += batchlister(probs, data.batch)
            mask = torch.bernoulli(probs) # (probs > 0.5).float()
            y_masks += batchlister(mask, data.batch)
            pos_out = F.softmax(self.predictor(data, mask), dim=1)
            pos_preds += lister(pos_out)
            neg_out = F.softmax(self.predictor(data, 1.0 - mask), dim=1)
            neg_preds += lister(neg_out)
            y_trues += lister(data.y)
            explanations += batchlister(data.true, data.batch)
        return y_probs, y_masks, pos_preds, neg_preds, y_trues, explanations
