import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

K_VALUES = [1, 2, 3, 4]
BASE_MODEL = "models/llama-7b"
DATA_DIR = "data"
RESULTS_DIR = "study_results"
MAX_HOPS = 3

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "study"), exist_ok=True)


def run_command(args):
    printable = " ".join(args)
    print(f"\n[RUNNING] {printable}")
    subprocess.run(args, check=True)


def calculate_metrics(result_json_path):
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    y_true = []
    y_pred = []

    for item in data:
        truth = 1 if str(item["ground_truth"]).lower() == "true" else 0
        pred = 1 if str(item["prediction"]).lower() == "true" else 0
        y_true.append(truth)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    return acc, f1


def plot_results(results):
    ks = sorted(results.keys())
    accs = [results[k]["acc"] for k in ks]
    f1s = [results[k]["f1"] for k in ks]

    x = np.arange(len(ks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width / 2, accs, width, label="Acc", color="#4472C4")
    rects2 = ax.bar(x + width / 2, f1s, width, label="F1", color="#ED7D31")

    ax.set_ylabel("Score (%)")
    ax.set_title("Impact of retained path number K on S2P performance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PATH-{k}" for k in ks])
    ax.set_ylim(60, 90)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt="%.2f")
    ax.bar_label(rects2, padding=3, fmt="%.2f")

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "figure2_reproduction.png")
    plt.savefig(plot_path)
    print(f"Figure saved to {plot_path}")


def main():
    final_metrics = {}

    for k in K_VALUES:
        print(f"\n{'=' * 20} Start Experiment for K={k} {'=' * 20}")

        train_file = os.path.join(DATA_DIR, "study", f"train_k{k}.json")
        test_file = os.path.join(DATA_DIR, "study", f"test_k{k}.json")
        model_output_dir = os.path.join(RESULTS_DIR, f"lora_s2p_k{k}")
        prediction_file = os.path.join(RESULTS_DIR, f"predictions_k{k}.json")

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            run_command(
                [
                    sys.executable,
                    "generate_enhanced_data.py",
                    "--top_k",
                    str(k),
                    "--max_hops",
                    str(MAX_HOPS),
                    "--output_train",
                    os.path.basename(train_file),
                    "--output_test",
                    os.path.basename(test_file),
                ]
            )
        else:
            print(f"Data for K={k} exists, skipping generation.")

        if not os.path.exists(os.path.join(model_output_dir, "adapter_config.json")):
            run_command(
                [
                    sys.executable,
                    "finetune.py",
                    "--base_model",
                    BASE_MODEL,
                    "--data_path",
                    train_file,
                    "--output_dir",
                    model_output_dir,
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
        else:
            print(f"Model for K={k} exists, skipping training.")

        if not os.path.exists(prediction_file):
            run_command(
                [
                    sys.executable,
                    "evaluate_batched.py",
                    "--base_model",
                    BASE_MODEL,
                    "--lora_weights",
                    model_output_dir,
                    "--test_data_path",
                    test_file,
                    "--output_file",
                    prediction_file,
                    "--batch_size",
                    "32",
                ]
            )
        else:
            print(f"Predictions for K={k} exist, skipping evaluation.")

        acc, f1 = calculate_metrics(prediction_file)
        final_metrics[k] = {"acc": acc, "f1": f1}
        print(f"Result for K={k}: Acc={acc:.2f}, F1={f1:.2f}")

    print(f"\n{'=' * 20} Final Hyperparameter Study Results {'=' * 20}")
    print(json.dumps(final_metrics, indent=2))

    with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    plot_results(final_metrics)


if __name__ == "__main__":
    main()
