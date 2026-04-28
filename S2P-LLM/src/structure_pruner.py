import random

import networkx as nx
import torch
import torch.nn.functional as F

from .process_kge import load_pretrain_kge


class StructurePruner:
    def __init__(self, graph, kge_path, id2entity, id2relation, device="cpu", score_metric="neg_l2"):
        self.graph = graph
        self.id2ent = id2entity
        self.id2rel = id2relation
        self.device = device
        self.score_metric = score_metric

        print(f"Loading KGE from {kge_path}...")
        ent_embs, rel_embs = load_pretrain_kge(kge_path)
        self.ent_embs = ent_embs.to(device)
        self.rel_embs = rel_embs.to(device)

        self.hidden_dim = self.rel_embs.shape[1]
        print(f"Embedding loaded. Relation Dim: {self.hidden_dim}")

    def _compose_path_embedding(self, path_r_ids):
        if not path_r_ids:
            return None

        composed = self.rel_embs[path_r_ids[0]].clone()
        for rid in path_r_ids[1:]:
            composed = composed * self.rel_embs[rid]
        return composed

    def calculate_structural_score(self, target_r_id, path_r_ids, metric=None):
        if not path_r_ids:
            return float("-inf")

        metric = metric or self.score_metric
        target_vec = self.rel_embs[target_r_id]
        path_vec = self._compose_path_embedding(path_r_ids)

        if path_vec is None:
            return float("-inf")

        # Paper-aligned default: structural support = negative geometric distance.
        if metric == "neg_l2":
            return (-torch.norm(path_vec - target_vec, p=2)).item()
        if metric == "cosine":
            return F.cosine_similarity(target_vec.unsqueeze(0), path_vec.unsqueeze(0)).item()

        raise ValueError(f"Unsupported metric: {metric}")

    def _parse_edge_path(self, edge_path):
        nodes = []
        relations = []

        for idx, edge in enumerate(edge_path):
            if len(edge) == 3:
                u, v, key = edge
                edge_data = self.graph.get_edge_data(u, v, key)
            else:
                u, v = edge
                edge_dict = self.graph.get_edge_data(u, v)
                if edge_dict is None:
                    return None
                key = next(iter(edge_dict.keys()))
                edge_data = edge_dict[key]

            if edge_data is None or "relation" not in edge_data:
                return None

            if idx == 0:
                nodes.append(u)
            nodes.append(v)
            relations.append(edge_data["relation"])

        return {"nodes": nodes, "relations": relations}

    def find_paths(self, h_id, t_id, k_hop=3, max_paths=200):
        if h_id not in self.graph or t_id not in self.graph:
            return []

        parsed_paths = []

        try:
            edge_path_iter = nx.all_simple_edge_paths(
                self.graph,
                source=h_id,
                target=t_id,
                cutoff=k_hop,
            )
            for edge_path in edge_path_iter:
                parsed = self._parse_edge_path(edge_path)
                if parsed is not None:
                    parsed_paths.append(parsed)
                if len(parsed_paths) >= max_paths:
                    break
        except AttributeError:
            # Fallback for old NetworkX versions.
            try:
                node_paths = nx.all_simple_paths(self.graph, source=h_id, target=t_id, cutoff=k_hop)
            except nx.NetworkXNoPath:
                return []

            for node_path in node_paths:
                relations = []
                valid = True

                for i in range(len(node_path) - 1):
                    u, v = node_path[i], node_path[i + 1]
                    edge_dict = self.graph.get_edge_data(u, v)
                    if not edge_dict:
                        valid = False
                        break

                    first_key = next(iter(edge_dict.keys()))
                    relation = edge_dict[first_key].get("relation")
                    if relation is None:
                        valid = False
                        break
                    relations.append(relation)

                if valid:
                    parsed_paths.append({"nodes": node_path, "relations": relations})
                if len(parsed_paths) >= max_paths:
                    break

        return parsed_paths

    def get_candidate_paths(self, h_id, r_id, t_id, max_hops=3, max_paths=200):
        candidates = self.find_paths(h_id, t_id, k_hop=max_hops, max_paths=max_paths)
        for candidate in candidates:
            candidate["score"] = self.calculate_structural_score(r_id, candidate["relations"])
        return candidates

    def select_paths(self, candidates, top_k=2, strategy="s2p", min_score=None):
        if not candidates:
            return []

        if strategy == "s2p":
            ranked = sorted(candidates, key=lambda x: x.get("score", float("-inf")), reverse=True)
        elif strategy == "shortest":
            ranked = sorted(candidates, key=lambda x: len(x.get("relations", [])))
        elif strategy == "random":
            ranked = candidates.copy()
            random.shuffle(ranked)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        if min_score is not None and strategy == "s2p":
            ranked = [p for p in ranked if p.get("score", float("-inf")) >= min_score]

        return ranked[:top_k]

    def linearize_path(self, path):
        nodes = path.get("nodes", [])
        rels = path.get("relations", [])

        if not nodes:
            return ""

        chunks = []
        for idx, rel_id in enumerate(rels):
            ent_name = self.id2ent.get(nodes[idx], f"Ent_{nodes[idx]}")
            rel_name = self.id2rel.get(rel_id, f"Rel_{rel_id}")
            chunks.append(f"{ent_name} --[{rel_name}]--> ")

        chunks.append(self.id2ent.get(nodes[-1], f"Ent_{nodes[-1]}"))
        return "".join(chunks)

    def linearize_paths(self, paths, include_score=True):
        lines = []
        for idx, path in enumerate(paths, start=1):
            path_str = self.linearize_path(path)
            score = path.get("score")
            if include_score and score is not None:
                lines.append(f"Path {idx} (Score: {score:.4f}): {path_str}")
            else:
                lines.append(f"Path {idx}: {path_str}")
        return "\n".join(lines)

    def get_pruned_context(
        self,
        h_id,
        r_id,
        t_id,
        top_k=2,
        max_hops=3,
        max_paths=200,
        min_support_score=None,
        return_paths=False,
    ):
        candidates = self.get_candidate_paths(h_id, r_id, t_id, max_hops=max_hops, max_paths=max_paths)
        best_paths = self.select_paths(
            candidates=candidates,
            top_k=top_k,
            strategy="s2p",
            min_score=min_support_score,
        )

        if not best_paths:
            empty_context = "No multi-hop path found in Knowledge Graph."
            if return_paths:
                return empty_context, 0.0, []
            return empty_context, 0.0

        context = self.linearize_paths(best_paths, include_score=True)
        max_score = best_paths[0].get("score", 0.0)

        if return_paths:
            return context, max_score, best_paths
        return context, max_score
