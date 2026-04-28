from __future__ import annotations

import argparse

from cr_llm_exp.common import collect_relation_set, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Export relation set from a DocRE train file.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    train_docs = read_json(args.train_path)
    relation_set = collect_relation_set(train_docs)
    write_json(relation_set, args.output_path)
    print(f"Relations: {len(relation_set)} -> {args.output_path}")


if __name__ == "__main__":
    main()
