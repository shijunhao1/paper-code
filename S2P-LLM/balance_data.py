import argparse
import json
import random


def normalize_label(sample):
    label = str(sample.get("output", "")).strip().lower()
    if label == "true":
        return "True"
    if label == "false":
        return "False"
    return None


def main():
    parser = argparse.ArgumentParser(description="Balance S2P training data to 1:1 positive/negative.")
    parser.add_argument("--input", type=str, default="data/CoDeX-S-s2p-train.json")
    parser.add_argument("--output", type=str, default="data/CoDeX-S-s2p-train-balanced.json")
    parser.add_argument("--target_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    pos_samples = [x for x in data if normalize_label(x) == "True"]
    neg_samples = [x for x in data if normalize_label(x) == "False"]

    if not pos_samples or not neg_samples:
        raise ValueError("Input data must contain both positive and negative samples.")

    max_balanced = min(len(pos_samples), len(neg_samples))
    target_len = max_balanced if args.target_per_class is None else min(args.target_per_class, max_balanced)

    balanced_data = random.sample(pos_samples, target_len) + random.sample(neg_samples, target_len)
    random.shuffle(balanced_data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(balanced_data, f, indent=2, ensure_ascii=False)

    print(f"Input: pos={len(pos_samples)}, neg={len(neg_samples)}")
    print(f"Balanced: pos={target_len}, neg={target_len}, total={len(balanced_data)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
