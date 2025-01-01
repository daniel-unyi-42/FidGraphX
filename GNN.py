import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torchmetrics import Accuracy, MeanAbsoluteError
from GNLayer import GNLayer

class MLPBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


class GNBlock(nn.Module):
    def __init__(
        self,
        conv_type,
        in_channels,
        hidden_channels,
        out_channels,
        edge_dim=0,
        use_norm=True,
    ):
        super().__init__()
        self.conv_type = conv_type
        if conv_type == 'GCNConv':
          assert edge_dim <= 1, 'GCNConv only supports edge_dim <= 1'
          self.conv = gnn.GCNConv(in_channels, out_channels)
        elif conv_type == 'ChebConv':
          assert edge_dim <= 1, 'ChebConv only supports edge_dim <= 1'
          self.conv = gnn.ChebConv(in_channels, out_channels, K=2)
        elif conv_type == 'GATConv':
          self.conv = gnn.GATv2Conv(in_channels, out_channels, edge_dim=edge_dim)
        elif conv_type == 'GINConv':
          if edge_dim == 0:
            self.conv = gnn.GINConv(MLPBlock(in_channels, hidden_channels, out_channels))
          else:
            self.conv = gnn.GINEConv(MLPBlock(in_channels, hidden_channels, out_channels), edge_dim=edge_dim)
        elif conv_type == 'NNConv':
          self.conv = GNLayer(in_channels, hidden_channels, out_channels, edge_dim=edge_dim)
        else:
          raise ValueError(f'conv_type must be one of "GCNConv", "ChebConv", "GATConv", "GINConv", "NNConv", but got {conv_type}')
        self.act = nn.LeakyReLU()
        self.use_norm = use_norm
        if use_norm:
          self.norm = gnn.BatchNorm(out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr)
        x = self.act(x)
        if self.use_norm:
          x = self.norm(x)
        return x


class GNN(nn.Module):
    def __init__(
        self,
        conv_type,
        task_type,
        learning_rate,
        in_channels,
        num_layers,
        hidden_channels,
        out_channels,
        edge_dim=0,
        use_norm=True,
      ):
      super().__init__()

      self.conv_type = conv_type
      self.task_type = task_type # 'classification' or 'regression'
      self.learning_rate = learning_rate
      self.in_channels = in_channels
      self.num_layers = num_layers
      self.hidden_channels = hidden_channels
      self.out_channels = out_channels
      self.edge_dim = edge_dim
      self.use_norm = use_norm
      self.task_type = task_type

      self.conv0 = GNBlock(conv_type, in_channels, hidden_channels, hidden_channels, edge_dim, use_norm)
      for i in range(num_layers-1):
        setattr(self, f'conv{i+1}', GNBlock(conv_type, hidden_channels, hidden_channels, hidden_channels, edge_dim, use_norm))
      self.head = MLPBlock(hidden_channels, hidden_channels, out_channels)
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      if self.device != torch.device('cuda'):
        print('WARNING: GPU not available. Using CPU instead.')
      self.to(self.device)
      self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
      if task_type == 'classification':
        self.criterion = F.cross_entropy
        self.metric = Accuracy(task = "multiclass", num_classes = out_channels).to(self.device)
      elif task_type == 'regression':
        self.criterion = F.mse_loss
        self.metric = MeanAbsoluteError().to(self.device)
      else:
        raise ValueError('Task type must be either "classification" or "regression"')

    def forward(self, data):
      x = data.x
      for i in range(self.num_layers):
        x = getattr(self, f'conv{i}')(x, data.edge_index, data.edge_attr)
      x = gnn.global_mean_pool(x, data.batch)
      x = self.head(x)
      return x

    def train_batch(self, loader):
      self.train()
      losses = []
      metrics = []
      for data in loader:
        data = data.to(self.device)
        if data.y.dim() == 0:
          data.y = data.y.unsqueeze(1)
        self.optimizer.zero_grad()
        logits = self(data)
        loss = self.criterion(logits, data.y)
        loss.backward()
        self.optimizer.step()
        losses.append(loss.item())
        metric = self.metric(logits, data.y)
        metrics.append(metric.item())
      return sum(losses) / len(losses), sum(metrics) / len(metrics)

    @torch.no_grad()
    def test_batch(self, loader):
      self.eval()
      losses = []
      metrics = []
      for data in loader:
        data = data.to(self.device)
        if data.y.dim() == 0:
          data.y = data.y.unsqueeze(1)
        logits = self(data)
        loss = self.criterion(logits, data.y)
        losses.append(loss.item())
        metric = self.metric(logits, data.y)
        metrics.append(metric.item())
      return sum(losses) / len(losses), sum(metrics) / len(metrics)

    @torch.no_grad()
    def predict_batch(self, loader):
      self.eval()
      y_preds = []
      y_trues = []
      for data in loader:
        data = data.to(self.device)
        if data.y.dim() == 0:
          data.y = data.y.unsqueeze(1)
        logits = self(data)
        if self.task_type == 'classification':
          y_pred = logits.argmax(dim=1)
        elif self.task_type == 'regression':
          y_pred = logits
        y_preds.append(y_pred)
        y_trues.append(data.y)
      y_preds = torch.cat(y_preds).detach().cpu().numpy()
      y_trues = torch.cat(y_trues).detach().cpu().numpy()
      return y_preds, y_trues
