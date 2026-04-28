from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


JsonDict = Dict[str, object]


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: str, indent: int = 2) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def flatten_doc_text(doc: JsonDict) -> str:
    return " ".join(" ".join(sent) for sent in doc["sents"])


def collect_relation_set(docs: Sequence[JsonDict]) -> List[str]:
    rels = set()
    for doc in docs:
        for label in doc.get("labels", []):
            rels.add(label["r"])
    return sorted(rels)


def build_gold_pair_relations(doc: JsonDict) -> Dict[Tuple[int, int], List[str]]:
    pair2rels: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for label in doc.get("labels", []):
        pair = (label["h"], label["t"])
        if label["r"] not in pair2rels[pair]:
            pair2rels[pair].append(label["r"])
    return pair2rels


def get_entity_name(doc: JsonDict, e_idx: int) -> str:
    return doc["vertexSet"][e_idx][0]["name"]


def get_entity_type(doc: JsonDict, e_idx: int) -> str:
    return doc["vertexSet"][e_idx][0].get("type", "UNK")


def iter_all_entity_pairs(num_entities: int) -> Iterable[Tuple[int, int]]:
    for h in range(num_entities):
        for t in range(num_entities):
            if h != t:
                yield h, t


def dedup_predictions(preds: Sequence[JsonDict]) -> List[JsonDict]:
    seen = set()
    out = []
    for item in preds:
        key = (item["title"], item["h_idx"], item["t_idx"], item["r"])
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out
