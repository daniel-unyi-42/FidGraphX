"""RelSC-H loader for the Zenodo preprocessed PyG .pt files.

Usage (random split, as in FidGraphX main.py):
    from src.RelSC import RelSCDataset
    dataset = RelSCDataset('data/RelSCH', project='hadoop')
    dataset = dataset.shuffle()

All three source .pt files are pooled; original split boundaries are ignored.
Files may be placed directly under data/RelSCH/ or inside data/RelSCH/processed_data_pt/.
Only load .pt files you trust (PyTorch pickle).
"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections.abc import Sequence
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch, InMemoryDataset, Dataset as PyGDataset

PROJECTS = ('hadoop', 'rdf', 'dubbo', 'H2', 'systemds', 'ossbuilds')
SPLITS = ('train', 'val', 'test')


def _resolve_root(root):
    """Accept .pt files directly below root or in root/processed_data_pt."""
    root = Path(root)
    if list(root.glob('type1_*_train_data.pt')):
        return root
    nested = root / 'processed_data_pt'
    if list(nested.glob('type1_*_train_data.pt')):
        return nested
    raise FileNotFoundError(
        f'No type1_*_train_data.pt found in {root} or {nested}. '
        'Check where you extracted processed_data_pt.zip.'
    )


def _torch_load(path):
    # Preprocessed PyG objects require pickle; only load the trusted Zenodo distribution.
    return torch.load(path, map_location='cpu', weights_only=False)


def _unwrap(obj, filename):
    """Accept standard saved PyG representations without guessing keys."""
    if isinstance(obj, InMemoryDataset):
        return [obj[i] for i in range(len(obj))]
    if isinstance(obj, (Data, Batch)):
        if isinstance(obj, Batch):
            return obj.to_data_list()
        return [obj]
    if isinstance(obj, tuple) and len(obj) == 2:
        data, slices = obj
        if isinstance(data, Data) and isinstance(slices, dict):
            ds = _InMemoryBridge()
            ds.data, ds.slices = data, slices
            return [ds[i] for i in range(len(ds))]
    if isinstance(obj, (list, tuple)) and all(isinstance(x, Data) for x in obj):
        return list(obj)
    if isinstance(obj, dict):
        for key in ('data_list', 'graphs', 'dataset'):
            if key in obj:
                return _unwrap(obj[key], filename)
        if 'data' in obj and 'slices' in obj:
            return _unwrap((obj['data'], obj['slices']), filename)
    raise TypeError(
        f'{filename}: unknown .pt format {type(obj).__name__}; '
        'inspect with torch.load(..., weights_only=False) and send type(obj), '
        'and keys if a dict. The loader does not silently reinterpret tensors.'
    )


class _InMemoryBridge(InMemoryDataset):
    def __init__(self):
        super().__init__(root=None)


class RelSCSplit(Dataset):
    def __init__(self, graphs, *, source):
        self.source = source
        self.graphs = [self._validate(g, i) for i, g in enumerate(graphs)]
        if not self.graphs:
            raise ValueError(f'{source}: empty split')

    def _validate(self, graph, idx):
        if not isinstance(graph, Data):
            raise TypeError(f'{self.source}[{idx}] is not a PyG Data object: {type(graph)}')
        graph = graph.clone()
        if graph.x is None or graph.edge_index is None or graph.y is None:
            raise ValueError(f'{self.source}[{idx}]: expected x, edge_index, y')
        if graph.x.ndim != 2:
            raise ValueError(f'{self.source}[{idx}]: x must be [nodes, features], got {tuple(graph.x.shape)}')
        if graph.edge_index.ndim != 2 or graph.edge_index.size(0) != 2:
            raise ValueError(f'{self.source}[{idx}]: edge_index must be [2, E]')
        if graph.y.numel() != 1:
            raise ValueError(f'{self.source}[{idx}]: graph-level y must be scalar, got {tuple(graph.y.shape)}')
        graph.x = graph.x.float().contiguous()
        graph.edge_index = graph.edge_index.long().contiguous()
        graph.y = graph.y.float().reshape(1)
        if graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr.float()
        if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.y).all():
            raise ValueError(f'{self.source}[{idx}]: non-finite x or y')
        if graph.edge_index.numel() and (graph.edge_index.min() < 0 or graph.edge_index.max() >= graph.num_nodes):
            raise ValueError(f'{self.source}[{idx}]: out-of-range edge_index')
        return graph

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index].clone()


def load_relsc_h(root='data/RelSCH', project='hadoop'):
    """Load ready-to-use RelSC-H (Type1) train/val/test splits."""
    root = _resolve_root(root)
    if project not in PROJECTS:
        raise ValueError(f'Unknown project {project!r}. Allowed: {PROJECTS}')
    result = {}
    for split in SPLITS:
        path = root / f'type1_{project}_{split}_data.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; use extracted processed_data_pt.zip')
        obj = _torch_load(path)
        result[split] = RelSCSplit(_unwrap(obj, str(path)), source=str(path))
    dims = {result[k][0].num_node_features for k in SPLITS}
    if len(dims) != 1:
        raise ValueError(f'Inconsistent feature dimensions across split first samples: {dims}')
    return result



class RelSCDataset(PyGDataset):
    """PyG dataset pooling the supplied type1 train/val/test files.

    Supports ``dataset.shuffle()``, slicing and ``torch.utils.data.random_split``
    just like the other PyG datasets used by FidGraphX.
    """

    def __init__(self, root='data/RelSCH', project='hadoop', transform=None):
        self.project = project
        self.graphs = []
        root = _resolve_root(root)
        if project not in PROJECTS:
            raise ValueError(f'Unknown project {project!r}. Allowed: {PROJECTS}')
        for split in SPLITS:
            path = root / f'type1_{project}_{split}_data.pt'
            if not path.is_file():
                raise FileNotFoundError(f'Missing file: {path}')
            raw = _unwrap(_torch_load(path), str(path))
            validated = RelSCSplit(raw, source=str(path))
            self.graphs.extend(validated.graphs)
        if not self.graphs:
            raise ValueError('No graphs were loaded')
        feature_dims = {g.x.size(1) for g in self.graphs}
        if len(feature_dims) != 1:
            raise ValueError(f'Inconsistent node feature dimensions: {feature_dims}')
        self.num_node_feature_dims = next(iter(feature_dims))
        super().__init__(root=None, transform=transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        # PyG dataset slicing/index selection is handled by the base class.
        return self.graphs[idx].clone()


def make_random_splits(dataset, fractions=(0.8, 0.1, 0.1), seed=42):
    """Deterministic random train/val/test partition of one pooled dataset."""
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError('fractions must contain three numbers summing to one')
    n = len(dataset)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    n_test = n - n_train - n_val
    return torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

def main():
    p = argparse.ArgumentParser(description='Inspect pooled RelSC-H Type1 data')
    p.add_argument('--root', default='data/RelSCH')
    p.add_argument('--project', default='hadoop', choices=PROJECTS)
    args = p.parse_args()
    ds = RelSCDataset(args.root, args.project)
    sizes = torch.tensor([int(g.num_nodes) for g in ds.graphs], dtype=torch.float)
    ys = torch.cat([g.y for g in ds.graphs])
    g = ds[0]
    print(f'Pooled RelSC-H {args.project}: {len(ds)} graphs | nodes min/median/max '
          f'{int(sizes.min())}/{int(sizes.median())}/{int(sizes.max())} '
          f'| x_dim={g.x.size(1)} | edge_dim='
          f'{None if g.edge_attr is None else tuple(g.edge_attr.shape[1:])} '
          f'| y min/max={ys.min().item():.5g}/{ys.max().item():.5g}')
    print('Example:', g)
    train, val, test = make_random_splits(ds)
    print(f'Random split: train={len(train)} val={len(val)} test={len(test)}')
    from torch_geometric.loader import DataLoader
    batch = next(iter(DataLoader(train, batch_size=2, shuffle=False)))
    print('Batch smoke test:', batch, 'y.shape=', tuple(batch.y.shape))


if __name__ == '__main__':
    main()
