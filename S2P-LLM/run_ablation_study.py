import json
import os
import subprocess
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASE_MODEL = "models/llama-7b"
DATA_DIR = "data"
RESULTS_DIR = "ablation_results"
K = 2
MAX_HOPS = 3

VARIANTS = {
    "S2P_Full": {"strategy": "s2p", "finetune": True},
    "No_Pruning_Shortest": {"strategy": "shortest", "finetune": True},
    "No_Pruning_Random": {"strategy": "random", "finetune": True},
    "No_Instruction_Tuning": {"strategy": "s2p", "finetune": False},
}

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "ablation"), exist_ok=True)


def run_command(args):
    printable = " ".join(args)
    print(f"\n[EXEC] {printable}")
    subprocess.run(args, check=True)


def calculate_metrics(result_json):
    with open(result_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    y_true = [1 if str(i["ground_truth"]).lower() == "true" else 0 for i in data]
    y_pred = [1 if str(i["prediction"]).lower() == "true" else 0 for i in data]
    return accuracy_score(y_true, y_pred) * 100, f1_score(y_true, y_pred) * 100


def main():
    final_results = []

    for name, config in VARIANTS.items():
        print(f"\n{'#' * 30}\nRunning Variant: {name}\n{'#' * 30}")

        strategy = config["strategy"]
        do_finetune = config["finetune"]

        train_file = os.path.join(DATA_DIR, "ablation", f"train_{name}.json")
        test_file = os.path.join(DATA_DIR, "ablation", f"test_{name}.json")
        lora_dir = os.path.join(RESULTS_DIR, f"lora_{name}")
        pred_file = os.path.join(RESULTS_DIR, f"preds_{name}.json")

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            run_command(
                [
                    sys.executable,
                    "generate_ablation_data.py",
                    "--strategy",
                    strategy,
                    "--k",
                    str(K),
                    "--max_hops",
                    str(MAX_HOPS),
                    "--output_train",
                    train_file,
                    "--output_test",
                    test_file,
                ]
            )
        else:
            print("Ablation data already exists, skip generation.")

        if do_finetune:
            if not os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
                run_command(
                    [
                        sys.executable,
                        "finetune.py",
                        "--base_model",
                        BASE_MODEL,
                        "--data_path",
                        train_file,
                        "--output_dir",
                        lora_dir,
                        "--batch_size",
                        "128",
                        "--micro_batch_size",
                        "16",
                        "--num_epochs",
                        "2",
                        "--learning_rate",
                        "5e-4",
                    ]
                )
            lora_arg = lora_dir
        else:
            print("Skipping fine-tuning for this variant.")
            lora_arg = "None"

        if not os.path.exists(pred_file):
            run_command(
                [
                    sys.executable,
                    "evaluate_batched.py",
                    "--base_model",
                    BASE_MODEL,
                    "--lora_weights",
                    lora_arg,
                    "--test_data_path",
                    test_file,
                    "--output_file",
                    pred_file,
                    "--batch_size",
                    "32",
                ]
            )

        acc, f1 = calculate_metrics(pred_file)
        print(f"Variant {name} -> Acc: {acc:.2f}, F1: {f1:.2f}")

        final_results.append({"Variant": name, "ACC (%)": f"{acc:.2f}", "F1 (%)": f"{f1:.2f}"})

    df = pd.DataFrame(final_results)
    print("\n=== Final Ablation Results (Table 3 reproduction) ===")
    print(df)
    df.to_csv(os.path.join(RESULTS_DIR, "ablation_summary.csv"), index=False)


if __name__ == "__main__":
    main()
