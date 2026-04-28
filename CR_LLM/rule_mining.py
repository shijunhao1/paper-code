from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from cr_llm_exp.common import get_entity_type, read_json, write_json


def load_candidate_pairs(candidate_path: str) -> Dict[str, set[Tuple[int, int]]]:
    title2pairs: Dict[str, set[Tuple[int, int]]] = defaultdict(set)
    if not candidate_path:
        return title2pairs

    candidates = read_json(candidate_path)
    for item in candidates:
        title2pairs[item["title"]].add((int(item["h_idx"]), int(item["t_idx"])))
    return title2pairs


def mine_rules(
    train_docs: List[dict],
    candidate_path: str,
    use_candidate_filter: bool,
    min_support: float,
    min_confidence: float,
    min_type_consistency: float,
    top_k: int,
):
    title2cands = load_candidate_pairs(candidate_path) if use_candidate_filter else {}

    antecedent_count = Counter()
    joint_count = Counter()
    rule_type_counter: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)

    total_joint_instances = 0

    for doc in train_docs:
        title = doc["title"]

        pair2rels: Dict[Tuple[int, int], set[str]] = defaultdict(set)
        for label in doc.get("labels", []):
            h, t, r = int(label["h"]), int(label["t"]), label["r"]
            if use_candidate_filter:
                if title not in title2cands:
                    continue
                if (h, t) not in title2cands[title]:
                    continue
            pair2rels[(h, t)].add(r)

        num_entities = len(doc["vertexSet"])
        e_types = [get_entity_type(doc, i) for i in range(num_entities)]

        for b in range(num_entities):
            heads = [a for a in range(num_entities) if a != b and (a, b) in pair2rels]
            tails = [c for c in range(num_entities) if c != b and (b, c) in pair2rels]

            for a in heads:
                for c in tails:
                    if a == c:
                        continue
                    if (a, c) not in pair2rels:
                        continue

                    for r1 in pair2rels[(a, b)]:
                        for r2 in pair2rels[(b, c)]:
                            antecedent = (r1, r2)
                            antecedent_count[antecedent] += 1

                            for r3 in pair2rels[(a, c)]:
                                rule_key = (r1, r2, r3)
                                joint_count[rule_key] += 1
                                total_joint_instances += 1

                                type_triplet = (e_types[a], e_types[b], e_types[c])
                                rule_type_counter[rule_key][type_triplet] += 1

    rules = []
    if total_joint_instances == 0:
        return rules

    for (r1, r2, r3), cnt_joint in joint_count.items():
        cnt_ant = antecedent_count[(r1, r2)]
        support = cnt_joint / total_joint_instances
        confidence = cnt_joint / cnt_ant if cnt_ant > 0 else 0.0

        type_counter = rule_type_counter[(r1, r2, r3)]
        dominant_type, dominant_cnt = type_counter.most_common(1)[0]
        type_consistency = dominant_cnt / sum(type_counter.values())

        if support < min_support:
            continue
        if confidence < min_confidence:
            continue
        if type_consistency < min_type_consistency:
            continue

        rules.append(
            {
                "premise": [r1, r2],
                "conclusion": r3,
                "count_joint": cnt_joint,
                "count_antecedent": cnt_ant,
                "support": round(support, 8),
                "confidence": round(confidence, 8),
                "type_consistency": round(type_consistency, 8),
                "dominant_type": list(dominant_type),
                "rule_text": f"IF r1={r1}(a,b) AND r2={r2}(b,c) THEN r3={r3}(a,c)",
            }
        )

    rules.sort(key=lambda x: (x["confidence"], x["support"], x["count_joint"]), reverse=True)
    if top_k > 0:
        rules = rules[:top_k]

    return rules


def main():
    parser = argparse.ArgumentParser(description="Mine two-hop logical rules for CR-LLM stage-2.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--candidate_path", type=str, default="")
    parser.add_argument("--use_candidate_filter", action="store_true")
    parser.add_argument("--min_support", type=float, default=3e-4)
    parser.add_argument("--min_confidence", type=float, default=0.65)
    parser.add_argument("--min_type_consistency", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()

    train_docs = read_json(args.train_path)

    rules = mine_rules(
        train_docs=train_docs,
        candidate_path=args.candidate_path,
        use_candidate_filter=args.use_candidate_filter,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_type_consistency=args.min_type_consistency,
        top_k=args.top_k,
    )

    payload = {
        "meta": {
            "train_path": args.train_path,
            "use_candidate_filter": args.use_candidate_filter,
            "candidate_path": args.candidate_path,
            "min_support": args.min_support,
            "min_confidence": args.min_confidence,
            "min_type_consistency": args.min_type_consistency,
            "top_k": args.top_k,
            "num_rules": len(rules),
        },
        "rules": rules,
    }

    write_json(payload, args.output_path)
    print(f"Mined {len(rules)} rules -> {args.output_path}")


if __name__ == "__main__":
    main()
