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
    if lines:
        return lines[-1]
    return raw_input.strip()


def normalize_label(item):
    raw = str(item.get("output", "True")).strip()
    return "True" if raw.lower() == "true" else "False"


def process_dataset(input_path, output_path, pruner, top_k, max_hops, max_paths, min_support_score, shuffle):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    enhanced_data = []
    stats = {"True": 0, "False": 0, "no_path": 0}

    print(f"Processing {input_path} (max_hops={max_hops}, top_k={top_k})...")

    for item in tqdm(data):
        h_id, r_id, t_id = item["embedding_ids"]
        label = normalize_label(item)
        stats[label] += 1

        path_context, max_score, selected_paths = pruner.get_pruned_context(
            h_id=h_id,
            r_id=r_id,
            t_id=t_id,
            top_k=top_k,
            max_hops=max_hops,
            max_paths=max_paths,
            min_support_score=min_support_score,
            return_paths=True,
        )

        if not selected_paths:
            stats["no_path"] += 1
            path_context = "No specific reasoning paths found."

        triple_text = extract_triple_text(item.get("input", ""))
        input_text = f"{path_context}\n\nThe input triple: {triple_text}"

        enhanced_data.append(
            {
                "instruction": S2P_INSTRUCTION,
                "input": input_text,
                "output": label,
                "embedding_ids": [h_id, r_id, t_id],
                "max_structural_support": round(float(max_score), 6),
            }
        )

    if shuffle:
        random.shuffle(enhanced_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(enhanced_data)} samples to {output_path}")
    print(f"Stats: {stats}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate S2P-enhanced train/test files.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--kge_path", type=str, default="data/CoDeX-S-rotate.pth")

    parser.add_argument("--train_file", type=str, default="CoDeX-S-train.json")
    parser.add_argument("--test_file", type=str, default="CoDeX-S-test.json")

    parser.add_argument("--output_train", type=str, default="CoDeX-S-s2p-train.json")
    parser.add_argument("--output_test", type=str, default="CoDeX-S-s2p-test.json")

    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--max_paths", type=int, default=200)
    parser.add_argument("--min_support_score", type=float, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser


def main():
    parser = build_arg_parser()
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

    train_in = os.path.join(args.data_dir, args.train_file)
    test_in = os.path.join(args.data_dir, args.test_file)
    train_out = os.path.join(args.data_dir, args.output_train)
    test_out = os.path.join(args.data_dir, args.output_test)

    process_dataset(
        input_path=train_in,
        output_path=train_out,
        pruner=pruner,
        top_k=args.top_k,
        max_hops=args.max_hops,
        max_paths=args.max_paths,
        min_support_score=args.min_support_score,
        shuffle=args.shuffle,
    )

    process_dataset(
        input_path=test_in,
        output_path=test_out,
        pruner=pruner,
        top_k=args.top_k,
        max_hops=args.max_hops,
        max_paths=args.max_paths,
        min_support_score=args.min_support_score,
        shuffle=False,
    )


if __name__ == "__main__":
    main()
