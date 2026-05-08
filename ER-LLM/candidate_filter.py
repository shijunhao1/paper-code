from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from cr_llm_exp.common import build_gold_pair_relations, iter_all_entity_pairs, read_json, write_json


@dataclass
class DocFeature:
    title: str
    input_ids: List[int]
    entity_pos: List[List[Tuple[int, int]]]
    pair_indices: List[Tuple[int, int]]
    pair_labels: List[int]


class FeatureDataset(Dataset):
    def __init__(self, features: Sequence[DocFeature]):
        self.features = list(features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> DocFeature:
        return self.features[idx]


def singleton_collate(features: List[DocFeature]) -> DocFeature:
    # This implementation intentionally uses batch size 1 to keep pair-level logic simple.
    return features[0]


def build_features(data_path: str, tokenizer, max_seq_len: int) -> List[DocFeature]:
    docs = read_json(data_path)
    features: List[DocFeature] = []

    for doc in tqdm(docs, desc=f"Build features: {os.path.basename(data_path)}"):
        vertex_set = doc["vertexSet"]

        entity_start = set()
        entity_end = set()
        for entity in vertex_set:
            for mention in entity:
                sid = mention["sent_id"]
                start, end = mention["pos"]
                entity_start.add((sid, start))
                entity_end.add((sid, end - 1))

        sent_map = []
        wp_tokens: List[str] = []

        for sid, sent in enumerate(doc["sents"]):
            token_map: Dict[int, int] = {}
            for tid, token in enumerate(sent):
                pieces = tokenizer.tokenize(token)
                if (sid, tid) in entity_start:
                    pieces = ["*"] + pieces
                if (sid, tid) in entity_end:
                    pieces = pieces + ["*"]
                token_map[tid] = len(wp_tokens)
                wp_tokens.extend(pieces)
            token_map[len(sent)] = len(wp_tokens)
            sent_map.append(token_map)

        wp_tokens = wp_tokens[: max_seq_len - 2]
        input_ids = tokenizer.convert_tokens_to_ids(wp_tokens)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        entity_pos = []
        for entity in vertex_set:
            mentions = []
            for mention in entity:
                sid = mention["sent_id"]
                start = sent_map[sid][mention["pos"][0]]
                end = sent_map[sid][mention["pos"][1]]
                mentions.append((start, end))
            entity_pos.append(mentions)

        pair2gold = build_gold_pair_relations(doc)
        pair_indices = []
        pair_labels = []
        for h, t in iter_all_entity_pairs(len(vertex_set)):
            pair_indices.append((h, t))
            pair_labels.append(1 if (h, t) in pair2gold else 0)

        features.append(
            DocFeature(
                title=doc["title"],
                input_ids=input_ids,
                entity_pos=entity_pos,
                pair_indices=pair_indices,
                pair_labels=pair_labels,
            )
        )

    return features


class CandidateFilterModel(nn.Module):
    def __init__(self, model_name_or_path: str, pair_hidden_size: int | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.encoder.config.hidden_size
        pair_hidden_size = pair_hidden_size or hidden_size

        self.head_entity_proj = nn.Linear(hidden_size, pair_hidden_size)
        self.tail_entity_proj = nn.Linear(hidden_size, pair_hidden_size)
        self.context_proj = nn.Linear(hidden_size, pair_hidden_size, bias=False)
        self.bilinear = nn.Bilinear(pair_hidden_size, pair_hidden_size, 1)

    def _aggregate_entities(
        self,
        sequence_output: torch.Tensor,
        attention: torch.Tensor,
        entity_pos: List[List[Tuple[int, int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # sequence_output: [L, D], attention: [H, L, L]
        seq_len = sequence_output.size(0)
        num_heads = attention.size(0)
        device = sequence_output.device

        entity_embs = []
        entity_atts = []
        offset = 1  # [CLS] token offset

        for mentions in entity_pos:
            mention_embs = []
            mention_atts = []
            for start, _ in mentions:
                wp_idx = start + offset
                if wp_idx < seq_len:
                    mention_embs.append(sequence_output[wp_idx])
                    mention_atts.append(attention[:, wp_idx])

            if mention_embs:
                e_emb = torch.logsumexp(torch.stack(mention_embs, dim=0), dim=0)
                e_att = torch.stack(mention_atts, dim=0).mean(dim=0)
            else:
                e_emb = torch.zeros(sequence_output.size(-1), device=device)
                e_att = torch.zeros(num_heads, seq_len, device=device)

            entity_embs.append(e_emb)
            entity_atts.append(e_att)

        return torch.stack(entity_embs, dim=0), torch.stack(entity_atts, dim=0)

    def _score_pairs(
        self,
        sequence_output: torch.Tensor,
        entity_embs: torch.Tensor,
        entity_atts: torch.Tensor,
        pair_indices: Sequence[Tuple[int, int]],
    ) -> torch.Tensor:
        pair_tensor = torch.tensor(pair_indices, device=sequence_output.device, dtype=torch.long)
        h_idx = pair_tensor[:, 0]
        t_idx = pair_tensor[:, 1]

        hs = entity_embs[h_idx]
        ts = entity_embs[t_idx]

        h_att = entity_atts[h_idx]
        t_att = entity_atts[t_idx]

        # q(s,o) = sum_k As_k * Ao_k
        pair_att = (h_att * t_att).sum(dim=1)
        pair_att = pair_att / (pair_att.sum(dim=1, keepdim=True) + 1e-6)

        # c(s,o) = H^T q(s,o)
        contexts = torch.matmul(pair_att, sequence_output)

        z_s = torch.tanh(self.head_entity_proj(hs) + self.context_proj(contexts))
        z_o = torch.tanh(self.tail_entity_proj(ts) + self.context_proj(contexts))

        return self.bilinear(z_s, z_o).squeeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_pos: List[List[Tuple[int, int]]],
        pair_indices: Sequence[Tuple[int, int]],
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            output_attentions=True,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state[0]
        attention = outputs.attentions[-1][0]

        entity_embs, entity_atts = self._aggregate_entities(sequence_output, attention, entity_pos)
        logits = self._score_pairs(sequence_output, entity_embs, entity_atts, pair_indices)
        return logits


def compute_prf(tp: int, pred_pos: int, gold_pos: int) -> Dict[str, float]:
    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / gold_pos if gold_pos > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def evaluate_pair_level(
    model: CandidateFilterModel,
    features: Sequence[DocFeature],
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    model.eval()

    tp = 0
    pred_pos = 0
    gold_pos = 0

    for feature in features:
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        logits = model(input_ids, attention_mask, feature.entity_pos, feature.pair_indices)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).long().cpu()
        labels = torch.tensor(feature.pair_labels, dtype=torch.long)

        tp += int(((preds == 1) & (labels == 1)).sum())
        pred_pos += int((preds == 1).sum())
        gold_pos += int((labels == 1).sum())

    out = compute_prf(tp, pred_pos, gold_pos)
    out.update({"tp": tp, "pred_pos": pred_pos, "gold_pos": gold_pos})
    return out


@torch.no_grad()
def dump_candidates(
    model: CandidateFilterModel,
    features: Sequence[DocFeature],
    device: torch.device,
    threshold: float,
    output_path: str,
) -> None:
    model.eval()
    candidates = []

    for feature in tqdm(features, desc=f"Predict candidates -> {os.path.basename(output_path)}"):
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        logits = model(input_ids, attention_mask, feature.entity_pos, feature.pair_indices)
        probs = torch.sigmoid(logits).cpu().tolist()

        for (h_idx, t_idx), score in zip(feature.pair_indices, probs):
            if score > threshold:
                candidates.append(
                    {
                        "title": feature.title,
                        "h_idx": h_idx,
                        "t_idx": t_idx,
                        "score": round(float(score), 6),
                    }
                )

    write_json(candidates, output_path)


def run_train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_features = build_features(args.train_path, tokenizer, args.max_seq_len)
    dev_features = build_features(args.dev_path, tokenizer, args.max_seq_len)
    test_features = build_features(args.test_path, tokenizer, args.max_seq_len) if args.test_path else []

    train_loader = DataLoader(
        FeatureDataset(train_features),
        batch_size=1,
        shuffle=True,
        collate_fn=singleton_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateFilterModel(args.model_name_or_path)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_dev_f1 = -1.0
    best_ckpt = os.path.join(args.output_dir, "candidate_filter_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for feature in pbar:
            input_ids = torch.tensor(feature.input_ids, dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            labels = torch.tensor(feature.pair_labels, dtype=torch.float, device=device)

            logits = model(input_ids, attention_mask, feature.entity_pos, feature.pair_indices)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        dev_metrics = evaluate_pair_level(model, dev_features, device, args.threshold)
        print(
            f"[Epoch {epoch}] train_loss={epoch_loss / max(1, len(train_loader)):.4f} "
            f"dev_f1={dev_metrics['f1']:.4f} "
            f"(P={dev_metrics['precision']:.4f}, R={dev_metrics['recall']:.4f})"
        )

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            torch.save(model.state_dict(), best_ckpt)

    print(f"Best dev F1: {best_dev_f1:.4f}")

    # Save train config for reproducible prediction.
    write_json(
        {
            "model_name_or_path": args.model_name_or_path,
            "max_seq_len": args.max_seq_len,
            "threshold": args.threshold,
        },
        os.path.join(args.output_dir, "candidate_filter_config.json"),
    )

    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    dump_candidates(
        model,
        train_features,
        device,
        args.threshold,
        os.path.join(args.output_dir, "train_candidates.json"),
    )
    dump_candidates(
        model,
        dev_features,
        device,
        args.threshold,
        os.path.join(args.output_dir, "dev_candidates.json"),
    )
    if test_features:
        dump_candidates(
            model,
            test_features,
            device,
            args.threshold,
            os.path.join(args.output_dir, "test_candidates.json"),
        )


def run_predict(args):
    cfg = read_json(args.config_path) if args.config_path else {}

    model_name_or_path = args.model_name_or_path or cfg.get("model_name_or_path", "roberta-large")
    max_seq_len = args.max_seq_len or cfg.get("max_seq_len", 1024)
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 0.5)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    features = build_features(args.data_path, tokenizer, max_seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateFilterModel(model_name_or_path)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    dump_candidates(model, features, device, threshold, args.output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Candidate pair filtering for DocRE (CR-LLM stage-1).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train candidate filter and dump candidate pairs.")
    train_parser.add_argument("--train_path", type=str, required=True)
    train_parser.add_argument("--dev_path", type=str, required=True)
    train_parser.add_argument("--test_path", type=str, default="")
    train_parser.add_argument("--output_dir", type=str, required=True)
    train_parser.add_argument("--model_name_or_path", type=str, default="roberta-large")
    train_parser.add_argument("--max_seq_len", type=int, default=1024)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning_rate", type=float, default=2e-5)
    train_parser.add_argument("--weight_decay", type=float, default=0.01)
    train_parser.add_argument("--max_grad_norm", type=float, default=1.0)
    train_parser.add_argument("--threshold", type=float, default=0.5)

    predict_parser = subparsers.add_parser("predict", help="Run candidate prediction with a trained checkpoint.")
    predict_parser.add_argument("--data_path", type=str, required=True)
    predict_parser.add_argument("--checkpoint_path", type=str, required=True)
    predict_parser.add_argument("--output_path", type=str, required=True)
    predict_parser.add_argument("--config_path", type=str, default="")
    predict_parser.add_argument("--model_name_or_path", type=str, default="")
    predict_parser.add_argument("--max_seq_len", type=int, default=0)
    predict_parser.add_argument("--threshold", type=float, default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
