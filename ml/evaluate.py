"""
LeafDoctor Test-Set Evaluation Script
Touches the held-out test set strictly once.
Computes:
1. Macro-F1 Score
2. Balanced Accuracy
3. Classification Report (Per-class Precision, Recall, F1, Support)
4. Confusion Matrix (Saved as PNG image)
5. Comprehensive Markdown and JSON reports in ml/reports/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score, accuracy_score

# Ensure local imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from train import PlantVillageDataset


def run_test_evaluation(model: nn.Module, test_loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs inference on held-out test loader and extracts softmax probabilities and labels."""
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(np.argmax(probs, axis=1))
            all_targets.extend(targets.numpy())

    return np.array(all_probs), np.array(all_preds), np.array(all_targets)


def plot_and_save_confusion_matrix(cm: np.ndarray, class_labels: List[str], output_path: Path):
    """Renders and exports a high-resolution confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.set_theme(font="sans-serif", style="white")

    # Shorten class labels for display
    short_labels = [label.replace(" - ", "\n") for label in class_labels]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        cbar=True,
        linewidths=0.5,
        linecolor="#cbd5e1"
    )

    plt.title("LeafDoctor: Held-Out Test Set Confusion Matrix", fontsize=14, fontweight="bold", pad=16)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("True Class", fontsize=11, fontweight="bold", labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def generate_markdown_report(
    macro_f1: float,
    balanced_acc: float,
    acc: float,
    clf_dict: Dict[str, Any],
    class_labels: List[str],
    total_samples: int,
    uncertain_stats: Dict[str, Any],
    output_path: Path,
):
    """Generates a professional Markdown evaluation report for hackathon judges."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 🌿 LeafDoctor Model Evaluation Report",
        "",
        "> **Hackathon Mission 78 | AI/ML Held-Out Test Evaluation**  ",
        "> *Rigorous evaluation performed on 15% isolated test data with anti-leakage deduplication.*",
        "",
        "---",
        "",
        "## 1. Executive Summary & Core Metrics",
        "",
        "| Metric | Score | Standard / Benchmark | Interpretation |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Macro-F1** | **{macro_f1 * 100:.2f}%** | Target: > 85% | Unweighted mean F1 treating all 10 classes equally |",
        f"| **Balanced Accuracy** | **{balanced_acc * 100:.2f}%** | Target: > 85% | Macro-average recall across all classes |",
        f"| **Overall Accuracy** | **{acc * 100:.2f}%** | Overall top-1 rate | Percentage of correct top-1 predictions |",
        f"| **Test Samples Evaluated** | **{total_samples}** | 15% Held-out split | Never seen during training or tuning |",
        "",
        "---",
        "",
        "## 2. Per-Class Precision, Recall & F1-Score",
        "",
        "| Crop | Condition | Precision | Recall | F1-Score | Test Support |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for label in class_labels:
        if label in clf_dict:
            row = clf_dict[label]
            crop, cond = label.split(" - ") if " - " in label else (label, "")
            lines.append(
                f"| {crop} | {cond} | {row['precision'] * 100:.1f}% | {row['recall'] * 100:.1f}% | {row['f1-score'] * 100:.1f}% | {row['support']} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Confusion Matrix",
        "",
        "![Confusion Matrix](confusion_matrix.png)",
        "",
        "---",
        "",
        "## 4. Uncertainty & Quality Guardrail Diagnostics",
        "",
        f"- **Confidence Threshold Applied:** `{uncertain_stats.get('confidence_threshold', 0.70)}`",
        f"- **Margin Threshold Applied:** `{uncertain_stats.get('margin_threshold', 0.18)}`",
        f"- **Confident Predictions:** `{uncertain_stats.get('confident_count', 0)} / {total_samples}` ({uncertain_stats.get('confident_rate', 0):.1f}%)",
        f"- **Flagged as Uncertain:** `{uncertain_stats.get('uncertain_count', 0)} / {total_samples}` ({uncertain_stats.get('uncertain_rate', 0):.1f}%)",
        "",
        "---",
        "",
        "## 5. Anti-Leakage & Reproducibility Statement",
        "",
        "1. **Exact Duplicate Purge:** File-level MD5 hashing eliminated identical copy images.",
        "2. **Grouped Near-Duplicate Isolation:** Perceptual 64-bit dHash clustered visually similar leaf angles; clusters were allocated to a single split only.",
        "3. **Fixed Random Seed:** Split generated with `random_seed=42`, strictly isolating the test set.",
        "4. **No Test Data in Tuning:** Calibration of confidence thresholds was performed solely on `val.csv`.",
        "",
        "---",
        "",
        "## 6. Honest Limitations",
        "- **Dataset Constraints:** PlantVillage images feature homogeneous lab backgrounds. Performance on cluttered outdoor garden backgrounds may be lower.",
        "- **Scope:** Screening is strictly calibrated for the 10 supported classes; unlisted pests or nutrient deficiencies require agronomist consultation.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Held-Out Test Set Evaluation for LeafDoctor")
    parser.add_argument("--test_csv", type=str, default=str(config.DATA_DIR / "processed" / "test.csv"))
    parser.add_argument("--model_path", type=str, default=str(config.MODELS_DIR / "best_model.pt"))
    parser.add_argument("--thresholds_path", type=str, default=str(config.MODELS_DIR / "thresholds.json"))
    parser.add_argument("--class_names_path", type=str, default=str(config.MODELS_DIR / "class_names.json"))

    args = parser.parse_args()

    print("=" * 70)
    print("🌿 LeafDoctor Held-Out Test Evaluation")
    print(f"Test Dataset: {args.test_csv}")
    print(f"Model Path  : {args.model_path}")
    print("=" * 70)

    test_csv_path = Path(args.test_csv)
    model_path = Path(args.model_path)
    thresholds_path = Path(args.thresholds_path)
    class_names_path = Path(args.class_names_path)

    if not test_csv_path.exists():
        print(f"[Error] Test set CSV not found at {test_csv_path}. Run prepare_data.py first.")
        sys.exit(1)

    if not model_path.exists():
        print(f"[Error] Model weights not found at {model_path}. Run train.py first.")
        sys.exit(1)

    # Load class names
    with open(class_names_path, "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    class_labels = [c["label"] for c in classes_data]

    # Load thresholds
    thresholds = {}
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
    conf_thresh = thresholds.get("confidence_threshold", 0.70)
    margin_thresh = thresholds.get("margin_threshold", 0.18)

    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading best checkpoint on {device.upper()}...")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Test transforms (deterministic, no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    test_dataset = PlantVillageDataset(str(test_csv_path), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Evaluating {len(test_dataset)} held-out test samples...")
    probs, preds, targets = run_test_evaluation(model, test_loader, device)

    # Compute Standard Metrics
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(targets, preds))
    accuracy = float(accuracy_score(targets, preds))

    # Detailed classification report
    target_names = [class_labels[i] for i in sorted(list(set(targets).union(set(preds))))]
    clf_dict = classification_report(targets, preds, target_names=target_names, output_dict=True, zero_division=0)
    clf_text = classification_report(targets, preds, target_names=target_names, zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_labels))))

    # Uncertainty Statistics
    top1_probs = np.max(probs, axis=1)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    uncertain_mask = (top1_probs < conf_thresh) | (margins < margin_thresh)
    uncertain_count = int(np.sum(uncertain_mask))
    confident_count = len(targets) - uncertain_count

    uncertain_stats = {
        "confidence_threshold": conf_thresh,
        "margin_threshold": margin_thresh,
        "confident_count": confident_count,
        "confident_rate": round(confident_count / len(targets) * 100, 2),
        "uncertain_count": uncertain_count,
        "uncertain_rate": round(uncertain_count / len(targets) * 100, 2),
    }

    # Save deliverables to reports/
    reports_dir = config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    cm_path = reports_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(cm, class_labels, cm_path)

    metrics_json_path = reports_dir / "test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "macro_f1": round(macro_f1, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "accuracy": round(accuracy, 4),
            "test_samples": len(targets),
            "uncertainty_diagnostics": uncertain_stats,
            "classification_report": clf_dict,
        }, f, indent=2)

    report_md_path = reports_dir / "evaluation_report.md"
    generate_markdown_report(
        macro_f1=macro_f1,
        balanced_acc=balanced_acc,
        acc=accuracy,
        clf_dict=clf_dict,
        class_labels=class_labels,
        total_samples=len(targets),
        uncertain_stats=uncertain_stats,
        output_path=report_md_path,
    )

    print("\n" + "=" * 70)
    print("🎉 Held-Out Test Evaluation Results")
    print(f"Overall Accuracy   : {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy  : {balanced_acc * 100:.2f}%")
    print(f"Macro-F1 Score     : {macro_f1 * 100:.2f}%")
    print("=" * 70)
    print("\nPer-Class Breakdown:")
    print(clf_text)
    print("=" * 70)
    print(f"Confusion Matrix PNG : {cm_path}")
    print(f"JSON Test Metrics    : {metrics_json_path}")
    print(f"Markdown Judge Report: {report_md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
