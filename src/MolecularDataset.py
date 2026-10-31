import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, download_url
from abc import ABC

class MolecularDataset(InMemoryDataset, ABC):
    url = None
    filename = None
    processed_filename = None

    def __init__(self, root, transform=None, pre_transform=None):
        if any(v is None for v in (self.url, self.filename, self.processed_filename)):
            raise ValueError("One of url, filename or processed_filename is None!")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        return [self.processed_filename]

    def download(self):
        download_url(f'{self.url}/{self.filename}', self.raw_dir)

    def process(self):
        self._process_graphs()

    def _graph_label(self, attr, labels, index):
        return torch.tensor([labels[index][0]]).long()

    def _process_graphs(self):
        data_list = []
        data = np.load(f'{self.raw_dir}/{self.raw_file_names}', allow_pickle=True)
        attr, X, y = data['attr'], data['X'], data['y']
        X = X[0]
        for index in range(len(X)):
            x = torch.from_numpy(X[index]['nodes'])
            edge_attr = torch.from_numpy(X[index]['edges'])
            graph_label = self._graph_label(attr, y, index)
            e1 = torch.from_numpy(X[index]['receivers']).long()
            e2 = torch.from_numpy(X[index]['senders']).long()
            edge_index = torch.stack([e1, e2])
            true = torch.from_numpy(attr[index][0]['nodes']).float()
            data = Data(
                x=x,
                y=graph_label,
                edge_attr=edge_attr,
                edge_index=edge_index,
                true=torch.any(true, dim=1).float(),
            )
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        torch.save(self.collate(data_list), self.processed_paths[0])


class BenzeneDataset(MolecularDataset):
    url = "https://github.com/mims-harvard/GraphXAI/raw/main/graphxai/datasets/real_world/benzene"
    filename = "benzene.npz"
    processed_filename = "benzene.pt"

class AlkaneCarbonylDataset(MolecularDataset):
    url = "https://github.com/mims-harvard/GraphXAI/raw/main/graphxai/datasets/real_world/fluoride_carbonyl"
    filename = "fluoride_carbonyl.npz"
    processed_filename = "alkane_carbonyl.pt"

class FluorideCarbonylDataset(MolecularDataset):
    url = "https://github.com/mims-harvard/GraphXAI/raw/main/graphxai/datasets/real_world/alkane_carbonyl"
    filename = "alkane_carbonyl.npz"
    processed_filename = "fluoride_carbonyl.pt"

class BenzeneCountDataset(BenzeneDataset):
    processed_filename = "benzene_count.pt"

    def _graph_label(self, attr, labels, index):
        mapping = {
            0: 0,
            6: 1,
            10: 2, 12: 2,
            14: 3, 16: 3, 18: 3,
            22: 4
        }
        count = attr[index][0]['nodes'].any(axis=1).sum()
        return torch.tensor([mapping[count]]).float()

