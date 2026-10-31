import torch


class RandomExplainer:
    def __init__(self, sparsity, device=None):
        self.sparsity = sparsity
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_probs(self, data):
        return torch.rand((data.num_nodes, 1), device=self.device)

    def sample_mask(self, data, mode='eval', probs=None):
        if probs is None:
            probs = self.get_probs(data)
        # Keep the exact random-mask semantics used previously.
        return (probs > (1.0 - self.sparsity)).float()
