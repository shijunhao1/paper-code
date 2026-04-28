from __future__ import annotations

import argparse
import json
import re
import time
from typing import Dict, List

from cr_llm_exp.common import dedup_predictions, read_json, write_json


def extract_relations_from_text(text: str, relation_set: List[str]) -> List[str]:
    text = text.strip()

    # Preferred format: {"relations": ["P17", ...]}
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        chunk = m.group(0)
        try:
            data = json.loads(chunk)
            rels = data.get("relations", [])
            if isinstance(rels, list):
                return [r for r in rels if r in relation_set]
        except json.JSONDecodeError:
            pass

    # Fallback: match relation ids from raw text
    return sorted({r for r in relation_set if r in text})


def run_mock_backend(item: dict, relation_set: List[str], use_gold_for_debug: bool) -> Dict[str, object]:
    if use_gold_for_debug:
        rels = [r for r in item.get("gold_relations", []) if r in relation_set]
    else:
        rels = []

    return {
        "raw_output": json.dumps({"relations": rels}, ensure_ascii=False),
        "relations": rels,
    }


def run_openai_backend(
    item: dict,
    relation_set: List[str],
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> Dict[str, object]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Please `pip install openai` before using --backend openai") from exc

    client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=item["prompt"],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    raw_text = getattr(resp, "output_text", "") or ""
    rels = extract_relations_from_text(raw_text, relation_set)

    return {
        "raw_output": raw_text,
        "relations": rels,
    }


def main():
    parser = argparse.ArgumentParser(description="Run stage-3 LLM inference for CR-LLM.")
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--relation_set_path", type=str, required=True, help="Path to JSON list of relation IDs.")
    parser.add_argument("--output_structured_path", type=str, required=True)
    parser.add_argument("--output_raw_path", type=str, required=True)
    parser.add_argument("--backend", type=str, choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_output_tokens", type=int, default=256)
    parser.add_argument("--sleep_sec", type=float, default=0.0)
    parser.add_argument("--mock_use_gold_for_debug", action="store_true")
    args = parser.parse_args()

    prompts = read_json(args.prompt_path)
    relation_set = read_json(args.relation_set_path)

    if not isinstance(relation_set, list):
        raise ValueError("relation_set_path must be a JSON list, e.g. ['P17','P159']")

    raw_results = []
    structured_predictions = []

    for idx, item in enumerate(prompts):
        if args.backend == "mock":
            out = run_mock_backend(item, relation_set, args.mock_use_gold_for_debug)
        else:
            out = run_openai_backend(
                item,
                relation_set,
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )

        raw_results.append(
            {
                "idx": idx,
                "title": item["title"],
                "h_idx": item["h_idx"],
                "t_idx": item["t_idx"],
                "raw_output": out["raw_output"],
                "relations": out["relations"],
            }
        )

        for r in out["relations"]:
            structured_predictions.append(
                {
                    "title": item["title"],
                    "h_idx": int(item["h_idx"]),
                    "t_idx": int(item["t_idx"]),
                    "r": r,
                    "evidence": [],
                }
            )

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    structured_predictions = dedup_predictions(structured_predictions)

    write_json(structured_predictions, args.output_structured_path)
    write_json(raw_results, args.output_raw_path)

    print(f"Raw outputs: {len(raw_results)} -> {args.output_raw_path}")
    print(f"Structured predictions: {len(structured_predictions)} -> {args.output_structured_path}")


if __name__ == "__main__":
    main()
