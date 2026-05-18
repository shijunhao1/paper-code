import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph

import metrics
from model import GNNEncoder


@dataclass
class GraphStruct:
    neighbors: List[List[int]]
    neighbor_sets: List[Set[int]]
    degrees: np.ndarray
    core_numbers: np.ndarray

    @classmethod
    def from_nx(cls, g: nx.Graph, num_nodes: int):
        neighbors: List[List[int]] = []
        neighbor_sets: List[Set[int]] = []
        for node in range(num_nodes):
            nbs = list(g.neighbors(node))
            neighbors.append(nbs)
            neighbor_sets.append(set(nbs))

        degrees = np.asarray([len(x) for x in neighbors], dtype=np.float32)
        core_map = nx.core_number(g)
        core_numbers = np.asarray([float(core_map.get(i, 0.0)) for i in range(num_nodes)], dtype=np.float32)
        return cls(neighbors=neighbors, neighbor_sets=neighbor_sets, degrees=degrees, core_numbers=core_numbers)


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    logits = torch.matmul(anchor, positive.t()) / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


def bfs_k_hop(struct: GraphStruct, center: int, hops: int, cap: int = 0) -> List[int]:
    visited = {center}
    frontier = {center}
    for _ in range(hops):
        nxt = set()
        for node in frontier:
            nxt.update(struct.neighbors[node])
        nxt -= visited
        if len(nxt) == 0:
            break
        visited |= nxt
        frontier = nxt

    nodes = sorted(visited)
    if cap > 0 and len(nodes) > cap:
        nodes = nodes[:cap]
        if center not in nodes:
            nodes[0] = center
            nodes = sorted(set(nodes))
    return nodes


def random_walk_context(struct: GraphStruct, start: int, walk_len: int, restart_prob: float, rng: np.random.Generator):
    cur = int(start)
    visited = {int(start)}

    for _ in range(walk_len):
        if restart_prob > 0.0 and rng.random() < restart_prob:
            cur = int(start)
        else:
            neigh = struct.neighbors[cur]
            if len(neigh) > 0:
                cur = int(rng.choice(neigh))
        visited.add(cur)

    return list(visited)


class NodeRetentionMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_emb: torch.Tensor, comm_emb: torch.Tensor):
        return self.net(torch.cat([node_emb, comm_emb], dim=1))


class PPFMR:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        gnn_type: str,
        device: torch.device,
        temperature: float = 0.5,
        lambda_struct: float = 1.0,
    ):
        self.device = device
        self.temperature = temperature
        self.lambda_struct = lambda_struct

        self.encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layer=n_layers,
            gnn_type=gnn_type,
        ).to(self.device)

    def fit_pattern_encoder(
        self,
        graph_data: Data,
        graph_struct: GraphStruct,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        walk_len: int = 128,
        restart_prob: float = 0.8,
        use_node_consistency: bool = True,
        use_struct_consistency: bool = True,
    ):
        if not use_node_consistency and not use_struct_consistency:
            raise ValueError("At least one of node consistency and structure consistency should be enabled.")

        optimizer = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=1e-5)
        rng = np.random.default_rng()

        graph_data = graph_data.to(self.device)
        num_nodes = graph_data.x.size(0)

        for epoch in range(1, epochs + 1):
            self.encoder.train()
            optimizer.zero_grad()

            sampled = rng.choice(num_nodes, size=min(batch_size, num_nodes), replace=False).tolist()
            node_emb = self.encoder(graph_data.x, graph_data.edge_index)

            anchor = node_emb[sampled]
            local_ctx, global_ctx = [], []

            for node in sampled:
                local_nodes = random_walk_context(
                    struct=graph_struct,
                    start=node,
                    walk_len=walk_len,
                    restart_prob=restart_prob,
                    rng=rng,
                )
                global_nodes = random_walk_context(
                    struct=graph_struct,
                    start=node,
                    walk_len=walk_len,
                    restart_prob=0.0,
                    rng=rng,
                )
                local_ctx.append(node_emb[local_nodes].mean(dim=0))
                global_ctx.append(node_emb[global_nodes].mean(dim=0))

            local_ctx = torch.stack(local_ctx, dim=0)
            global_ctx = torch.stack(global_ctx, dim=0)

            node_loss = info_nce(anchor, local_ctx, temperature=self.temperature)
            struct_loss = info_nce(local_ctx, global_ctx, temperature=self.temperature)

            loss = 0.0
            if use_node_consistency:
                loss = loss + node_loss
            if use_struct_consistency:
                coeff = self.lambda_struct if use_node_consistency else 1.0
                loss = loss + coeff * struct_loss

            loss.backward()
            optimizer.step()

            print(
                f"***pretrain epoch: {epoch:04d} | total_loss: {float(loss):.5f} | "
                f"node_loss: {float(node_loss):.5f} | struct_loss: {float(struct_loss):.5f}"
            )

    def get_node_embeddings(self, graph_data: Data):
        self.encoder.eval()
        with torch.no_grad():
            graph_data = graph_data.to(self.device)
            z = self.encoder(graph_data.x, graph_data.edge_index)
        return z

    def encode_communities(self, communities: Sequence[Sequence[int]], graph_data: Data, batch_size: int = 64):
        self.encoder.eval()
        graph_data = graph_data.to(self.device)
        num_nodes = graph_data.x.size(0)

        all_emb = []
        with torch.no_grad():
            nb = math.ceil(len(communities) / batch_size) if len(communities) > 0 else 0
            for i in range(nb):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(communities))
                batch_graphs = []

                for c in communities[start:end]:
                    c = sorted(set(int(x) for x in c))
                    if len(c) == 0:
                        continue
                    edge_index, _ = subgraph(c, graph_data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
                    batch_graphs.append(Data(x=graph_data.x[c], edge_index=edge_index))

                if len(batch_graphs) == 0:
                    continue

                batch = Batch().from_data_list(batch_graphs).to(self.device)
                _, comm_emb = self.encoder(batch.x, batch.edge_index, batch.batch)
                all_emb.append(comm_emb.detach().cpu())

        if len(all_emb) == 0:
            return torch.zeros((0, self.encoder.output_dim), dtype=torch.float32)
        return torch.cat(all_emb, dim=0)


def build_seed_scores(struct: GraphStruct, use_propagation_feature: bool = True):
    if not use_propagation_feature:
        return struct.degrees.copy()

    scores = np.zeros_like(struct.degrees, dtype=np.float32)
    for node, nbs in enumerate(struct.neighbors):
        if len(nbs) == 0:
            neigh_core = 0.0
        else:
            neigh_core = float(struct.core_numbers[nbs].mean())
        scores[node] = float(struct.degrees[node] + neigh_core)
    return scores


def structural_strength(v: int, candidate_nodes: Sequence[int], struct: GraphStruct) -> float:
    neigh_v = struct.neighbor_sets[v]
    deg_v = max(1.0, float(struct.degrees[v]))

    vals = []
    for u in candidate_nodes:
        neigh_u = struct.neighbor_sets[u]
        cn = len(neigh_v & neigh_u)
        denom = math.sqrt(deg_v * max(1.0, float(struct.degrees[u])))
        vals.append(cn / (denom + 1e-9))

    if len(vals) == 0:
        return 0.0
    score = float(np.mean(vals))
    return float(min(max(score, 0.0), 1.0))


def global_association(v: int, candidate_set: Set[int], struct: GraphStruct) -> float:
    linked = len(struct.neighbor_sets[v] & candidate_set)
    return linked / max(1.0, float(len(candidate_set)))


def embedding_similarity(v_emb: torch.Tensor, c_emb: torch.Tensor) -> float:
    sim = F.cosine_similarity(v_emb.unsqueeze(0), c_emb.unsqueeze(0)).item()
    return float((sim + 1.0) / 2.0)


def generate_candidates(
    node_emb: torch.Tensor,
    struct: GraphStruct,
    num_candidates: int,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    seed_hops: int = 2,
    tau_scale: float = 0.0,
    max_candidate_size: int = 40,
    scope_cap: int = 300,
    assignment_rounds: int = 2,
    use_propagation_feature: bool = True,
):
    z = node_emb.detach().cpu()
    num_nodes = z.size(0)

    seed_scores = build_seed_scores(struct, use_propagation_feature=use_propagation_feature)
    seeds = np.argsort(-seed_scores)[: min(num_candidates, num_nodes)].tolist()

    candidate_members: List[Set[int]] = [{int(seed)} for seed in seeds]
    scopes = [bfs_k_hop(struct, int(seed), hops=seed_hops, cap=scope_cap) for seed in seeds]

    for round_id in range(assignment_rounds):
        node_bag: Dict[int, List[Tuple[int, float]]] = {}
        candidate_score_cache: List[Dict[int, float]] = [dict() for _ in seeds]

        for cid, seed in enumerate(seeds):
            c_nodes = sorted(candidate_members[cid])
            c_set = set(c_nodes)
            c_emb = z[c_nodes].mean(dim=0)

            for v in scopes[cid]:
                s1 = structural_strength(v=v, candidate_nodes=c_nodes, struct=struct)
                s2 = global_association(v=v, candidate_set=c_set, struct=struct)
                s3 = embedding_similarity(v_emb=z[v], c_emb=c_emb)
                score = alpha * s1 + beta * s2 + gamma * s3

                candidate_score_cache[cid][v] = float(score)
                if v not in node_bag:
                    node_bag[v] = []
                node_bag[v].append((cid, float(score)))

        new_members: List[Set[int]] = [{int(seed)} for seed in seeds]
        for v, score_list in node_bag.items():
            vals = np.asarray([s for _, s in score_list], dtype=np.float32)
            tau_v = float(vals.mean() + tau_scale * vals.std())

            for cid, score in score_list:
                if score >= tau_v:
                    new_members[cid].add(int(v))

        if max_candidate_size > 0:
            for cid, seed in enumerate(seeds):
                nodes = list(new_members[cid])
                if len(nodes) <= max_candidate_size:
                    continue

                ranked = sorted(nodes, key=lambda x: candidate_score_cache[cid].get(x, -1e9), reverse=True)
                ranked = ranked[:max_candidate_size]
                if seed not in ranked:
                    ranked[-1] = int(seed)
                new_members[cid] = set(ranked)

        candidate_members = new_members
        print(f"***candidate assignment round {round_id + 1}/{assignment_rounds} finished")

    candidates = [sorted(list(x)) for x in candidate_members if len(x) > 0]

    uniq = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def train_refiner(
    refiner: NodeRetentionMLP,
    node_emb: torch.Tensor,
    train_comms: Sequence[Sequence[int]],
    struct: GraphStruct,
    lr: float = 1e-3,
    epochs: int = 30,
    neg_ratio: float = 1.0,
    extend_hops: int = 2,
    max_anchor_per_comm: int = 5,
):
    device = next(refiner.parameters()).device
    z = node_emb.to(device)

    optimizer = optim.Adam(refiner.parameters(), lr=lr, weight_decay=1e-5)
    bce = nn.BCELoss()

    num_nodes = z.size(0)

    for epoch in range(1, epochs + 1):
        refiner.train()
        optimizer.zero_grad()

        feat_nodes, feat_comms, labels = [], [], []

        for comm in train_comms:
            c_nodes = sorted(set(int(x) for x in comm))
            if len(c_nodes) == 0:
                continue

            c_set = set(c_nodes)
            c_emb = z[c_nodes].mean(dim=0)

            anchor_num = min(max_anchor_per_comm, len(c_nodes))
            anchors = random.sample(c_nodes, k=anchor_num)

            extended = set()
            for a in anchors:
                extended.update(bfs_k_hop(struct, center=a, hops=extend_hops, cap=0))

            neg_pool = [x for x in extended if x not in c_set]
            if len(neg_pool) == 0:
                neg_pool = [x for x in range(num_nodes) if x not in c_set]
                if len(neg_pool) == 0:
                    continue

            neg_size = max(1, int(len(c_nodes) * neg_ratio))
            if len(neg_pool) > neg_size:
                neg_nodes = random.sample(neg_pool, k=neg_size)
            else:
                neg_nodes = neg_pool

            for v in c_nodes:
                feat_nodes.append(z[v])
                feat_comms.append(c_emb)
                labels.append(1.0)
            for v in neg_nodes:
                feat_nodes.append(z[v])
                feat_comms.append(c_emb)
                labels.append(0.0)

        if len(labels) == 0:
            continue

        node_batch = torch.stack(feat_nodes, dim=0)
        comm_batch = torch.stack(feat_comms, dim=0)
        label_batch = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)

        pred = refiner(node_batch, comm_batch)
        loss = bce(pred, label_batch)

        loss.backward()
        optimizer.step()

        print(f"***refine epoch: {epoch:04d} | loss: {float(loss):.5f} | samples: {len(labels)}")

    return refiner


def refine_candidate_set(
    candidates: Sequence[Sequence[int]],
    refiner: NodeRetentionMLP,
    node_emb: torch.Tensor,
    threshold: float = 0.5,
):
    device = next(refiner.parameters()).device
    z = node_emb.to(device)

    refined = []
    refiner.eval()

    with torch.no_grad():
        for c in candidates:
            c_nodes = sorted(set(int(x) for x in c))
            if len(c_nodes) == 0:
                continue

            c_emb = z[c_nodes].mean(dim=0).unsqueeze(0).repeat(len(c_nodes), 1)
            prob = refiner(z[c_nodes], c_emb).squeeze(1)

            keep_idx = torch.where(prob >= threshold)[0].tolist()
            if len(keep_idx) == 0:
                keep_idx = [int(torch.argmax(prob).item())]

            keep_nodes = sorted({c_nodes[i] for i in keep_idx})
            refined.append(keep_nodes)

    uniq = []
    seen = set()
    for c in refined:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def match_candidates(
    train_emb: np.ndarray,
    candidate_emb: np.ndarray,
    candidate_comms: Sequence[Sequence[int]],
    num_pred: int,
):
    if len(candidate_comms) == 0 or train_emb.shape[0] == 0:
        return []

    num_shot = train_emb.shape[0]
    num_each = max(1, int(num_pred / max(1, num_shot)))

    pred = []
    for i in range(num_shot):
        q = train_emb[i]
        dist = np.sqrt(np.sum((candidate_emb - q) ** 2, axis=1))
        rank = list(np.argsort(dist))

        count = 0
        for idx in rank:
            if count >= num_each:
                break
            comm = list(candidate_comms[idx])
            if comm not in pred:
                pred.append(comm)
                count += 1

    return pred


def evaluate_predictions(pred_comms, test_comms, verbose: bool = True):
    return metrics.eval_scores(pred_comms, test_comms, tmp_print=verbose)
