import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
import networkx as nx
import random

class BAMotifsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, num_graphs=500, ba_nodes=25, attach_prob=0.1):
        self.num_graphs = num_graphs
        self.ba_nodes = ba_nodes
        self.attach_prob = attach_prob

        super(BAMotifsDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        motifs = {
            0: ('house', nx.house_graph()),
            1: ('house_x', nx.house_x_graph()),
            2: ('diamond', nx.diamond_graph()),
            3: ('pentagon', nx.cycle_graph(n=5)),
            4: ('wheel', nx.wheel_graph(n=6)),
            5: ('star', nx.star_graph(n=5))
        }

        data_list = []

        for label, (motif_type, motif_graph) in motifs.items():
            for _ in range(self.num_graphs):
                ba_graph = nx.barabasi_albert_graph(self.ba_nodes, m=3)

                ba_graph = self._attach_motif(ba_graph, motif_graph)

                data = from_networkx(ba_graph)

                # Ensure only necessary attributes are in the Data object
                data.x = torch.full((data.num_nodes, 10), 0.1, dtype=torch.float)
                data.y = torch.tensor([label], dtype=torch.long)
                data.true = torch.cat([torch.zeros(self.ba_nodes).long(), torch.ones(len(motif_graph)).long()])

                # Remove unnecessary attributes, including 'name' if it exists
                for attr in list(data.keys()):
                    if attr not in {'x', 'edge_index', 'y', 'true'}:
                        del data[attr]

                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _attach_motif(self, base_graph, motif_graph):
        # Relabel motif nodes to ensure unique node identifiers
        motif_graph = nx.relabel_nodes(motif_graph, {n: n + len(base_graph) for n in motif_graph.nodes})

        # Add motif nodes and edges to the base graph
        base_graph.add_nodes_from(motif_graph.nodes(data=True))
        base_graph.add_edges_from(motif_graph.edges(data=True))

        base_nodes = list(base_graph.nodes)
        motif_nodes = list(motif_graph.nodes)

        # Ensure at least one motif node is connected to the base graph
        base_node = random.choice(base_nodes[:self.ba_nodes])  # base_nodes[:self.ba_nodes] ensures picking from original BA nodes
        motif_node = random.choice(motif_nodes)
        base_graph.add_edge(base_node, motif_node)

        # Probabilistically connect other motif nodes to the base graph
        for node in motif_nodes:
            if node != motif_node:  # Skip the node already connected
                if random.random() < self.attach_prob:
                    base_node = random.choice(base_nodes[:self.ba_nodes])
                    base_graph.add_edge(base_node, node)

        return base_graph
