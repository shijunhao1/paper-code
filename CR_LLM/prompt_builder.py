from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

from cr_llm_exp.common import (
    build_gold_pair_relations,
    collect_relation_set,
    get_entity_name,
    get_entity_type,
    read_json,
    write_json,
)


def load_rules(rules_path: str) -> List[dict]:
    payload = read_json(rules_path)
    if isinstance(payload, dict) and "rules" in payload:
        return payload["rules"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported rules format: {rules_path}")


def select_rules_for_pair(rules: List[dict], h_type: str, t_type: str, max_rules: int) -> List[dict]:
    matched = []
    for rule in rules:
        dom_type = rule.get("dominant_type", ["UNK", "UNK", "UNK"])
        if len(dom_type) == 3 and dom_type[0] == h_type and dom_type[2] == t_type:
            matched.append(rule)

    if not matched:
        matched = rules

    return matched[:max_rules]


def build_local_context(doc: dict, h_idx: int, t_idx: int, sent_window: int) -> str:
    sent_ids = set()
    for m in doc["vertexSet"][h_idx]:
        sent_ids.add(m["sent_id"])
    for m in doc["vertexSet"][t_idx]:
        sent_ids.add(m["sent_id"])

    expanded = set()
    max_sid = len(doc["sents"]) - 1
    for sid in sent_ids:
        for d in range(-sent_window, sent_window + 1):
            nsid = sid + d
            if 0 <= nsid <= max_sid:
                expanded.add(nsid)

    selected = sorted(expanded)
    lines = []
    for sid in selected:
        sent_text = " ".join(doc["sents"][sid])
        lines.append(f"[S{sid}] {sent_text}")

    return "\n".join(lines)


def build_prompt(
    doc: dict,
    h_idx: int,
    t_idx: int,
    relation_set: List[str],
    selected_rules: List[dict],
    local_context: str,
) -> str:
    h_name = get_entity_name(doc, h_idx)
    t_name = get_entity_name(doc, t_idx)

    if selected_rules:
        rules_text = "\n".join(
            [
                (
                    f"{i + 1}. {rule['rule_text']} "
                    f"(conf={rule['confidence']}, sup={rule['support']}, tc={rule['type_consistency']})"
                )
                for i, rule in enumerate(selected_rules)
            ]
        )
    else:
        rules_text = "None"

    return f"""
You are a document-level relation extraction assistant. Your task is to determine the relations between the given entity pair based on the local evidence text and the provided logical rules.

Entity pair: ({h_name}, {t_name})

Local evidence text:
{local_context}

Reference logical rules:
{rules_text}

Candidate relation set:
{relation_set}

Output only one JSON object in the following format:
{{"relations": ["relation_id_1", "relation_id_2"]}}

If no relation exists between the entity pair, output:
{{"relations": []}}

Do not output explanations, markdown, or any extra text outside the JSON object.
""".strip()

def main():
    parser = argparse.ArgumentParser(description="Build rule-constrained prompts for CR-LLM stage-3.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--candidate_path", type=str, required=True)
    parser.add_argument("--rules_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_rules", type=int, default=5)
    parser.add_argument("--sent_window", type=int, default=1)
    args = parser.parse_args()

    docs = read_json(args.data_path)
    train_docs = read_json(args.train_path)
    candidates = read_json(args.candidate_path)
    rules = load_rules(args.rules_path)

    relation_set = collect_relation_set(train_docs)

    title2doc = {d["title"]: d for d in docs}
    title2cands = defaultdict(list)
    for item in candidates:
        title2cands[item["title"]].append(item)

    outputs = []

    for title, cand_list in title2cands.items():
        if title not in title2doc:
            continue
        doc = title2doc[title]
        pair2gold = build_gold_pair_relations(doc)

        for cand in cand_list:
            h_idx = int(cand["h_idx"])
            t_idx = int(cand["t_idx"])
            h_type = get_entity_type(doc, h_idx)
            t_type = get_entity_type(doc, t_idx)

            selected_rules = select_rules_for_pair(rules, h_type, t_type, args.max_rules)
            local_context = build_local_context(doc, h_idx, t_idx, args.sent_window)
            prompt = build_prompt(doc, h_idx, t_idx, relation_set, selected_rules, local_context)

            outputs.append(
                {
                    "title": title,
                    "h_idx": h_idx,
                    "t_idx": t_idx,
                    "head": get_entity_name(doc, h_idx),
                    "tail": get_entity_name(doc, t_idx),
                    "head_type": h_type,
                    "tail_type": t_type,
                    "score": cand.get("score", None),
                    "gold_relations": pair2gold.get((h_idx, t_idx), []),
                    "prompt": prompt,
                }
            )

    write_json(outputs, args.output_path)
    print(f"Built {len(outputs)} prompts -> {args.output_path}")


if __name__ == "__main__":
    main()
