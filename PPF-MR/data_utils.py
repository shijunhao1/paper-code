import glob
import os
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


def _read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if len(line.strip()) > 0]


def _find_dataset_files(dataset_dir: str, dataset: str) -> Tuple[str, str, str]:
    cmty_file = os.path.join(dataset_dir, f"{dataset}-1.90.cmty.txt")
    edge_file = os.path.join(dataset_dir, f"{dataset}-1.90.ungraph.txt")
    feat_file = os.path.join(dataset_dir, f"{dataset}-1.90.nodefeat.txt")

    if os.path.exists(cmty_file) and os.path.exists(edge_file):
        return edge_file, cmty_file, feat_file if os.path.exists(feat_file) else ""

    cmty_candidates = sorted(glob.glob(os.path.join(dataset_dir, "*.cmty.txt")))
    edge_candidates = sorted(glob.glob(os.path.join(dataset_dir, "*.ungraph.txt")))
    feat_candidates = sorted(glob.glob(os.path.join(dataset_dir, "*.nodefeat.txt")))

    if len(cmty_candidates) == 0 or len(edge_candidates) == 0:
        raise FileNotFoundError(f"Cannot find cmty/ungraph files under {dataset_dir}")

    feat = feat_candidates[0] if len(feat_candidates) > 0 else ""
    return edge_candidates[0], cmty_candidates[0], feat


def _feature_augmentation(nodes: List[int], edges: List[List[int]], num_nodes: int, normalize: bool = True):
    g = nx.Graph(edges)
    g.add_nodes_from(nodes)

    degrees = [g.degree[node] for node in range(num_nodes)]
    feat_matrix = np.zeros([num_nodes, 5], dtype=np.float32)
    feat_matrix[:, 0] = np.asarray(degrees, dtype=np.float32)

    for node in range(num_nodes):
        neigh = list(g.neighbors(node))
        if len(neigh) == 0:
            continue
        neighbor_deg = feat_matrix[neigh, 0]
        feat_matrix[node, 1:] = neighbor_deg.min(), neighbor_deg.max(), neighbor_deg.mean(), neighbor_deg.std()

    if normalize:
        feat_matrix = (feat_matrix - feat_matrix.mean(0, keepdims=True)) / (feat_matrix.std(0, keepdims=True) + 1e-9)

    return feat_matrix, g


def _load_optional_node_features(feat_file: str, mapping: dict, num_nodes: int):
    if feat_file is None or feat_file == "" or (not os.path.exists(feat_file)):
        return None

    raw = _read_lines(feat_file)
    if len(raw) == 0:
        return None

    sample = raw[0].split()
    if len(sample) <= 1:
        return None

    dim = len(sample) - 1
    feat = np.zeros((num_nodes, dim), dtype=np.float32)

    for line in raw:
        seg = line.split()
        if len(seg) < 2:
            continue
        raw_node = int(seg[0])
        if raw_node not in mapping:
            continue
        idx = mapping[raw_node]
        feat[idx] = np.asarray([float(x) for x in seg[1:]], dtype=np.float32)

    if np.allclose(feat.std(0), 0.0):
        return feat
    feat = (feat - feat.mean(0, keepdims=True)) / (feat.std(0, keepdims=True) + 1e-9)
    return feat


def load_dataset(data_root: str, dataset: str):
    dataset_dir = os.path.join(data_root, dataset)
    edge_file, cmty_file, feat_file = _find_dataset_files(dataset_dir, dataset)

    communities = [[int(i) for i in x.split()] for x in _read_lines(cmty_file)]
    edges = [[int(i) for i in x.split()] for x in _read_lines(edge_file)]

    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]

    nodes = {node for e in edges for node in e}
    for com in communities:
        nodes.update(com)

    mapping = {u: i for i, u in enumerate(sorted(nodes))}

    edges = [[mapping[u], mapping[v]] for u, v in edges]
    communities = [[mapping[node] for node in com if node in mapping] for com in communities]

    num_nodes = len(mapping)
    num_edges = len(edges)
    num_comms = len(communities)

    opt_feat = _load_optional_node_features(feat_file, mapping, num_nodes)
    if opt_feat is None:
        features, nx_graph = _feature_augmentation(list(range(num_nodes)), edges, num_nodes)
    else:
        features = opt_feat
        nx_graph = nx.Graph(edges)
        nx_graph.add_nodes_from(range(num_nodes))

    converted_edges = [[v, u] for u, v in edges]
    edge_index = torch.LongTensor(np.asarray(edges + converted_edges)).transpose(0, 1)
    graph_data = Data(x=torch.FloatTensor(features), edge_index=edge_index)

    print(f"[{dataset.upper()}] #Nodes {num_nodes}, #Edges {num_edges}, #Communities {num_comms}")
    return num_nodes, num_edges, num_comms, graph_data, nx_graph, communities
