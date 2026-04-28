import argparse
import json
import os
import random

import torch
from tqdm import tqdm

from src.data_loader import CodexDataLoader
from src.structure_pruner import StructurePruner

S2P_INSTRUCTION = (
    "Given some paths and a triple from a knowledge graph. "
    "The path serves as the context information of the triple. "
    "Please determine the correctness of the triple and response True or False."
)


def extract_triple_text(raw_input):
    if not raw_input:
        return ""
    lines = [ln.strip() for ln in raw_input.strip().split("\n") if ln.strip()]
    return lines[-1] if lines else raw_input.strip()


def normalize_label(item):
    raw = str(item.get("output", "True")).strip().lower()
    return "True" if raw == "true" else "False"


def process_dataset(input_file, output_file, pruner, top_k, max_hops, max_paths, strategy, min_support_score):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    enhanced_data = []
    stats = {"True": 0, "False": 0, "no_path": 0}

    print(f"Processing {input_file} | strategy={strategy} | top_k={top_k} | max_hops={max_hops}")

    for item in tqdm(data):
        h_id, r_id, t_id = item["embedding_ids"]
        label = normalize_label(item)
        stats[label] += 1

        candidates = pruner.get_candidate_paths(h_id, r_id, t_id, max_hops=max_hops, max_paths=max_paths)
        selected_paths = pruner.select_paths(
            candidates=candidates,
            top_k=top_k,
            strategy=strategy,
            min_score=min_support_score if strategy == "s2p" else None,
        )

        if selected_paths:
            path_context = pruner.linearize_paths(selected_paths, include_score=(strategy == "s2p"))
            max_score = selected_paths[0].get("score", 0.0)
        else:
            stats["no_path"] += 1
            path_context = "No specific reasoning paths found."
            max_score = 0.0

        triple_text = extract_triple_text(item.get("input", ""))
        s2p_input_content = f"{path_context}\n\nThe input triple: {triple_text}"

        enhanced_data.append(
            {
                "instruction": S2P_INSTRUCTION,
                "input": s2p_input_content,
                "output": label,
                "embedding_ids": [h_id, r_id, t_id],
                "max_structural_support": round(float(max_score), 6),
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(enhanced_data)} samples to {output_file}")
    print(f"Stats: {stats}")


def main():
    parser = argparse.ArgumentParser(description="Generate ablation data with different path strategies.")
    parser.add_argument("--strategy", type=str, required=True, choices=["s2p", "random", "shortest"])
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--max_paths", type=int, default=200)
    parser.add_argument("--min_support_score", type=float, default=None)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--raw_train", type=str, default="CoDeX-S-train.json")
    parser.add_argument("--raw_test", type=str, default="CoDeX-S-test.json")
    parser.add_argument("--output_train", type=str, required=True)
    parser.add_argument("--output_test", type=str, required=True)

    parser.add_argument("--kge_path", type=str, default="data/CoDeX-S-rotate.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    loader = CodexDataLoader(args.data_dir)
    loader.load_data()

    pruner = StructurePruner(
        graph=loader.graph,
        kge_path=args.kge_path,
        id2entity=loader.id2entity,
        id2relation=loader.id2relation,
        device=device,
    )

    train_input = os.path.join(args.data_dir, args.raw_train)
    test_input = os.path.join(args.data_dir, args.raw_test)

    process_dataset(
        input_file=train_input,
        output_file=args.output_train,
        pruner=pruner,
        top_k=args.k,
        max_hops=args.max_hops,
        max_paths=args.max_paths,
        strategy=args.strategy,
        min_support_score=args.min_support_score,
    )

    process_dataset(
        input_file=test_input,
        output_file=args.output_test,
        pruner=pruner,
        top_k=args.k,
        max_hops=args.max_hops,
        max_paths=args.max_paths,
        strategy=args.strategy,
        min_support_score=args.min_support_score,
    )


if __name__ == "__main__":
    main()
