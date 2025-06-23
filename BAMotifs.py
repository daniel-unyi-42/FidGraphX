import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
import networkx as nx
import random
from abc import ABC, abstractmethod

class BaseBAMotifs(InMemoryDataset, ABC):
    def __init__(self, root,
                 num_graphs, attach_prob,
                 min_nodes, max_nodes,
                 min_edges_per_node, max_edges_per_node,
                 transform=None, pre_transform=None,
        ):
        self.num_graphs = num_graphs
        self.attach_prob = attach_prob
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_edges_per_node = min_edges_per_node
        self.max_edges_per_node = max_edges_per_node
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # All possible motifs
        all_motifs = {
            'house':    nx.house_graph(),
            'house_x':  nx.house_x_graph(),
            'diamond':  nx.diamond_graph(),
            'pentagon': nx.cycle_graph(5),
            'wheel':    nx.wheel_graph(6),
            'star':     nx.star_graph(5),
            'grid':     nx.convert_node_labels_to_integers(
                            nx.grid_graph(dim=(3, 3)),
                            first_label=0, ordering='default'
                        )
        }
        data_list = []
        for _ in range(self.num_graphs):
            # 1) Generate BA base graph
            G = self._gen_base_graph()
            # 2) Attach motifs, compute label and ground-truth nodes and optional features X
            label, gt_nodes, X = self._attach_and_label(G, all_motifs)
            # 3) Convert to PyG Data and assign attributes
            data = from_networkx(G)
            data.x = X if X is not None else torch.full((data.num_nodes, 10), 0.1)
            data.y = torch.tensor([label], dtype=torch.long)
            # 4) Build node-level ground-truth mask named 'true'
            node_mask = torch.zeros(data.num_nodes, dtype=torch.long)
            node_mask[list(gt_nodes)] = 1
            data.true = node_mask
            data_list.append(data)
        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _gen_base_graph(self):
        n = random.randint(self.min_nodes, self.max_nodes)
        m = random.randint(self.min_edges_per_node, self.max_edges_per_node)
        return nx.barabasi_albert_graph(n, m)

    def _attach_motif(self, G, motif_graph):
        # Increase motif node labels by the number of nodes in G
        offset = G.number_of_nodes()
        motif = nx.relabel_nodes(motif_graph, {n: n + offset for n in motif_graph.nodes})
        G.add_nodes_from(motif.nodes(data=True))
        G.add_edges_from(motif.edges(data=True))
        # Always connect a random node in motif to a random node in G
        base_nodes = list(range(offset))
        anchor = random.choice(base_nodes)
        motif_nodes = list(range(offset, G.number_of_nodes()))
        connect = random.choice(motif_nodes)
        G.add_edge(anchor, connect)
        # Probabilistically attach other edges
        for node in motif_nodes:
            if node != connect and random.random() < self.attach_prob:
                G.add_edge(random.choice(base_nodes), node)
        return motif_nodes

    @abstractmethod
    def _attach_and_label(self, G, all_motifs):
        """
        Attach motifs, compute:
          - graph label
          - gt_nodes: set of node indices (for ground-truth mask)
          - X: optional feature tensor (num_nodes x feat_dim) or None
        """
        pass


class BAMotifs(BaseBAMotifs):
    def _attach_and_label(self, G, all_motifs):
        keys = list(all_motifs.keys())
        motif = random.choice(keys)
        nodes = self._attach_motif(G, all_motifs[motif])
        label = keys.index(motif)
        return label, set(nodes), None

class BAImbalancedMotifs(BaseBAMotifs):
    def _attach_and_label(self, G, motifs_all):
        keys = list(motifs_all.keys())
        weights = [0.30, 0.05, 0.10, 0.10, 0.10, 0.05, 0.30]
        motif = random.choices(keys, weights)[0]
        nodes = self._attach_motif(G, motifs_all[motif])
        label = keys.index(motif)
        return label, set(nodes), None

class BAIgnoringMotifs(BaseBAMotifs):
    def _attach_and_label(self, G, motifs_all):
        relevant = ['house', 'grid']
        distractors = ['house_x', 'diamond', 'pentagon', 'wheel', 'star']
        pos = random.choice(relevant)
        neg = random.choice(distractors)
        pos_nodes = self._attach_motif(G, motifs_all[pos])
        self._attach_motif(G, motifs_all[neg])
        label = relevant.index(pos)
        return label, set(pos_nodes), None

class BAORMotifs(BaseBAMotifs):
    def _attach_and_label(self, G, all_motifs):
        types = ['house', 'grid']
        pattern = random.choice([0, 1, 2, 3])
        present = []
        if pattern == 1:
            present = [types[0]]
        elif pattern == 2:
            present = [types[1]]
        elif pattern == 3:
            present = types[:]
        gt = set()
        for t in present:
            nodes = self._attach_motif(G, all_motifs[t])
            gt.update(nodes)
        label = 1 if len(present) >= 1 else 0
        return label, gt, None

class BAXORMotifs(BaseBAMotifs):
    def _attach_and_label(self, G, all_motifs):
        types = ['house', 'grid']
        pattern = random.choice([0, 1, 2, 3])
        present = []
        if pattern == 1:
            present = [types[0]]
        elif pattern == 2:
            present = [types[1]]
        elif pattern == 3:
            present = types[:]
        gt = set()
        for t in present:
            nodes = self._attach_motif(G, all_motifs[t])
            gt.update(nodes)
        label = 1 if len(present) == 1 else 0
        return label, gt, None

class BAANDMotifs(BaseBAMotifs):
    def _attach_and_label(self, G, all_motifs):
        types = ['house', 'grid']
        pattern = random.choice([0, 1, 2, 3])
        present = []
        if pattern == 3:
            present = types[:]
        elif pattern == 1:
            present = [types[0]]
        elif pattern == 2:
            present = [types[1]]
        gt = set()
        for t in present:
            nodes = self._attach_motif(G, all_motifs[t])
            gt.update(nodes)
        label = 1 if len(present) == 2 else 0
        return label, gt, None
