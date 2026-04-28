import argparse
import json

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import BitsAndBytesConfig, GenerationConfig, LlamaForCausalLM, LlamaTokenizer

S2P_INSTRUCTION = (
    "Given some paths and a triple from a knowledge graph. "
    "The path serves as the context information of the triple. "
    "Please determine the correctness of the triple and response True or False."
)


class Prompter:
    def __init__(self):
        self.template = {
            "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "response_split": "### Response:",
        }

    def generate_prompt(self, instruction, input_text):
        return self.template["prompt_input"].format(instruction=instruction, input=input_text)

    def get_response(self, full_text):
        split_res = full_text.split(self.template["response_split"])
        if len(split_res) > 1:
            return split_res[1].strip()
        return full_text.strip()


def normalize_binary(text):
    lowered = text.strip().lower()
    if lowered.startswith("true") or " true" in f" {lowered}":
        return 1, "True"
    if lowered.startswith("false") or " false" in f" {lowered}":
        return 0, "False"
    return 0, "False"


def should_load_lora(lora_weights):
    if lora_weights is None:
        return False
    cleaned = str(lora_weights).strip().lower()
    return cleaned not in {"", "none", "null"}


def load_model(base_model, lora_weights, device):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else {"": device},
    )

    if should_load_lora(lora_weights):
        print(f"Loading LoRA adapters from {lora_weights}")
        model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
    else:
        print("Running evaluation without LoRA adapters.")

    model.eval()
    return model, tokenizer


def main(
    base_model="models/llama-7b",
    lora_weights="./s2p-codex-output",
    test_data_path="data/CoDeX-S-s2p-test.json",
    output_file="s2p_prediction_results.json",
    batch_size=32,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== S2P Evaluation ===\nBase model: {base_model}\nDevice: {device}")

    model, tokenizer = load_model(base_model, lora_weights, device)
    prompter = Prompter()

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    all_ground_truths = []
    all_prompts = []

    for item in test_data:
        gt_label = 1 if str(item.get("output", "")).strip().lower() == "true" else 0
        all_ground_truths.append(gt_label)

        prompt = prompter.generate_prompt(S2P_INSTRUCTION, item.get("input", ""))
        all_prompts.append(prompt)

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.9,
        num_beams=1,
        do_sample=False,
        max_new_tokens=10,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    all_predictions = []
    detailed_results = []

    total_samples = len(all_prompts)
    for i in tqdm(range(0, total_samples, batch_size), desc="Inferencing"):
        batch_prompts = all_prompts[i : i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, total_samples)))

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
            )

        decoded_outputs = tokenizer.batch_decode(generation_output, skip_special_tokens=True)

        for idx, raw_output in enumerate(decoded_outputs):
            sample_idx = batch_indices[idx]
            sample = test_data[sample_idx]
            gt = all_ground_truths[sample_idx]

            response_text = prompter.get_response(raw_output)
            pred_label, pred_text = normalize_binary(response_text)

            all_predictions.append(pred_label)
            detailed_results.append(
                {
                    "id": sample_idx,
                    "input_context": sample.get("input", ""),
                    "ground_truth": sample.get("output", ""),
                    "model_response": response_text,
                    "prediction": pred_text,
                    "correct": pred_label == gt,
                }
            )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    acc = accuracy_score(all_ground_truths, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_ground_truths,
        all_predictions,
        average="binary",
        zero_division=0,
    )

    print("=" * 40)
    print(f"S2P Results on {total_samples} samples")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("=" * 40)
    print(f"Saved details to {output_file}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate S2P model with optional LoRA.")
    parser.add_argument("--base_model", type=str, default="models/llama-7b")
    parser.add_argument("--lora_weights", type=str, default="./s2p-codex-output")
    parser.add_argument("--test_data_path", type=str, default="data/CoDeX-S-s2p-test.json")
    parser.add_argument("--output_file", type=str, default="s2p_prediction_results.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(
        base_model=args.base_model,
        lora_weights=args.lora_weights,
        test_data_path=args.test_data_path,
        output_file=args.output_file,
        batch_size=args.batch_size,
        device=args.device,
    )
