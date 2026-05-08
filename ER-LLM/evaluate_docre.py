from __future__ import annotations

import argparse
from typing import Dict, List, Set, Tuple

from cr_llm_exp.common import dedup_predictions, read_json


def build_train_fact_set(train_docs: List[dict]) -> Set[Tuple[str, str, str]]:
    facts = set()
    for doc in train_docs:
        vset = doc["vertexSet"]
        for label in doc.get("labels", []):
            h_idx, t_idx, rel = int(label["h"]), int(label["t"]), label["r"]
            for h_m in vset[h_idx]:
                for t_m in vset[t_idx]:
                    facts.add((h_m["name"], t_m["name"], rel))
    return facts


def main():
    parser = argparse.ArgumentParser(description="Evaluate DocRE predictions with F1 and Ign-F1.")
    parser.add_argument("--truth_path", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    args = parser.parse_args()

    truth_docs = read_json(args.truth_path)
    preds = dedup_predictions(read_json(args.pred_path))
    train_docs = read_json(args.train_path)

    std = set()
    title2vertex = {}

    for doc in truth_docs:
        title = doc["title"]
        title2vertex[title] = doc["vertexSet"]
        for label in doc.get("labels", []):
            std.add((title, label["r"], int(label["h"]), int(label["t"])))

    total_gold = len(std)
    total_pred = len(preds)

    train_facts = build_train_fact_set(train_docs)

    correct = 0
    correct_in_train = 0

    for p in preds:
        key = (p["title"], p["r"], int(p["h_idx"]), int(p["t_idx"]))
        if key not in std:
            continue

        correct += 1

        title = p["title"]
        if title not in title2vertex:
            continue
        vertex = title2vertex[title]
        h_idx, t_idx = int(p["h_idx"]), int(p["t_idx"])

        in_train = False
        for h_m in vertex[h_idx]:
            for t_m in vertex[t_idx]:
                if (h_m["name"], t_m["name"], p["r"]) in train_facts:
                    in_train = True
                    break
            if in_train:
                break

        if in_train:
            correct_in_train += 1

    precision = correct / total_pred if total_pred > 0 else 0.0
    recall = correct / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    ign_denom = max(1, total_pred - correct_in_train)
    ign_precision = (correct - correct_in_train) / ign_denom
    ign_f1 = (2 * ign_precision * recall / (ign_precision + recall)) if (ign_precision + recall) > 0 else 0.0

    print(f"Predictions: {total_pred}")
    print(f"Gold triples: {total_gold}")
    print(f"Correct: {correct}")
    print(f"Correct & seen in train: {correct_in_train}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1: {f1:.6f}")
    print(f"Ign-Precision: {ign_precision:.6f}")
    print(f"Ign-F1: {ign_f1:.6f}")


if __name__ == "__main__":
    main()
