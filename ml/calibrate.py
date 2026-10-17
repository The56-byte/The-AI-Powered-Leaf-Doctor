"""
LeafDoctor Threshold Calibration Script
Calibrates confidence and margin uncertainty thresholds strictly on the VALIDATION set.
Never touches the held-out test set.
Saves calibrated thresholds to ml/models/thresholds.json.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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


def run_validation_inference(model: nn.Module, val_loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs inference on validation loader and extracts top1, top2, margin, and ground truth."""
    model.eval()
    all_top1_probs = []
    all_margins = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            for i in range(len(probs)):
                sorted_probs = np.sort(probs[i])[::-1]
                top1_p = float(sorted_probs[0])
                top2_p = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
                margin = top1_p - top2_p

                all_top1_probs.append(top1_p)
                all_margins.append(margin)
                all_preds.append(int(np.argmax(probs[i])))
                all_targets.append(int(targets[i]))

    return np.array(all_top1_probs), np.array(all_margins), np.array(all_preds), np.array(all_targets)


def calibrate_thresholds(top1_probs: np.ndarray, margins: np.ndarray, preds: np.ndarray, targets: np.ndarray, target_precision: float = 0.85):
    """
    Sweeps confidence and margin thresholds on the validation set.
    Goal: Select the minimum confidence threshold where precision among confident predictions
    is >= target_precision (or maximizes F1/precision trade-off).
    """
    correct = (preds == targets)
    total_val = len(correct)

    confidence_candidates = np.linspace(0.50, 0.90, 41)
    best_conf_threshold = 0.70
    best_conf_score = -1.0
    conf_curve_data = []

    for thresh in confidence_candidates:
        accepted_mask = top1_probs >= thresh
        accepted_count = int(np.sum(accepted_mask))

        if accepted_count == 0:
            precision = 1.0
        else:
            precision = float(np.sum(correct[accepted_mask]) / accepted_count)

        coverage = float(accepted_count / total_val)
        conf_curve_data.append({"threshold": round(thresh, 2), "precision": round(precision, 4), "coverage": round(coverage, 4)})

        # Harmonic score balancing precision and coverage
        if precision >= target_precision and coverage >= 0.50:
            score = 2 * (precision * coverage) / (precision + coverage + 1e-8)
            if score > best_conf_score:
                best_conf_score = score
                best_conf_threshold = float(thresh)

    # Margin threshold calibration
    margin_candidates = np.linspace(0.10, 0.35, 26)
    best_margin_threshold = 0.18
    best_margin_score = -1.0

    for m_thresh in margin_candidates:
        accepted_mask = margins >= m_thresh
        accepted_count = int(np.sum(accepted_mask))
        if accepted_count == 0:
            precision = 1.0
        else:
            precision = float(np.sum(correct[accepted_mask]) / accepted_count)
        coverage = float(accepted_count / total_val)

        if precision >= target_precision and coverage >= 0.50:
            score = 2 * (precision * coverage) / (precision + coverage + 1e-8)
            if score > best_margin_score:
                best_margin_score = score
                best_margin_threshold = float(m_thresh)

    return round(best_conf_threshold, 2), round(best_margin_threshold, 2), conf_curve_data


def plot_calibration_curves(conf_curve_data: List[Dict[str, float]], chosen_thresh: float, output_path: Path):
    """Plots validation precision and coverage vs confidence threshold."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    threshs = [d["threshold"] for d in conf_curve_data]
    precs = [d["precision"] * 100 for d in conf_curve_data]
    covs = [d["coverage"] * 100 for d in conf_curve_data]

    plt.figure(figsize=(8, 5))
    plt.plot(threshs, precs, label="Precision on Confident Predictions (%)", color="#10b981", linewidth=2.5)
    plt.plot(threshs, covs, label="Coverage (% of accepted samples)", color="#3b82f6", linewidth=2.5, linestyle="--")
    plt.axvline(x=chosen_thresh, color="#ef4444", linestyle=":", label=f"Selected Threshold ({chosen_thresh})", linewidth=2)

    plt.title("LeafDoctor Validation Calibration Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Confidence Threshold", fontsize=11)
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Calibrate Uncertainty Thresholds on Validation Set")
    parser.add_argument("--val_csv", type=str, default=str(config.DATA_DIR / "processed" / "val.csv"))
    parser.add_argument("--model_path", type=str, default=str(config.MODELS_DIR / "best_model.pt"))
    parser.add_argument("--output_json", type=str, default=str(config.MODELS_DIR / "thresholds.json"))
    parser.add_argument("--output_plot", type=str, default=str(config.REPORTS_DIR / "calibration_curve.png"))
    parser.add_argument("--target_precision", type=float, default=0.85)

    args = parser.parse_args()

    print("=" * 70)
    print("🌿 LeafDoctor Threshold Calibration (Validation Set Only)")
    print(f"Validation Dataset: {args.val_csv}")
    print(f"Model Checkpoint  : {args.model_path}")
    print("=" * 70)

    val_csv_path = Path(args.val_csv)
    model_path = Path(args.model_path)

    if not val_csv_path.exists():
        print(f"[Error] Validation dataset not found at {val_csv_path}. Run prepare_data.py first.")
        sys.exit(1)

    if not model_path.exists():
        print(f"[Error] Trained model not found at {model_path}. Run train.py first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device.upper()}...")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Transforms for validation
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    val_dataset = PlantVillageDataset(str(val_csv_path), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"Running inference across {len(val_dataset)} validation samples...")
    top1_probs, margins, preds, targets = run_validation_inference(model, val_loader, device)

    raw_val_acc = float(np.mean(preds == targets))
    print(f"Raw Validation Accuracy before thresholding: {raw_val_acc * 100:.2f}%")

    calibrated_conf, calibrated_margin, curve_data = calibrate_thresholds(
        top1_probs, margins, preds, targets, target_precision=args.target_precision
    )

    # Save calibration plot
    plot_path = Path(args.output_plot)
    plot_calibration_curves(curve_data, calibrated_conf, plot_path)

    # Save to thresholds.json
    output_thresholds = {
        "confidence_threshold": calibrated_conf,
        "margin_threshold": calibrated_margin,
        "blur_threshold": float(config.DEFAULT_BLUR_THRESHOLD),
        "min_brightness": float(config.MIN_BRIGHTNESS),
        "max_brightness": float(config.MAX_BRIGHTNESS),
        "calibrated_dataset": "PlantVillage Validation Set (70/15/15)",
        "validation_samples_calibrated": len(val_dataset),
        "raw_validation_accuracy": round(raw_val_acc, 4),
        "note": "Calibrated empirically using validation set precision-coverage sweep."
    }

    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_thresholds, f, indent=2)

    print("\n" + "=" * 70)
    print("🎉 Threshold Calibration Complete!")
    print(f"Empirically Calibrated Confidence Threshold : {calibrated_conf}")
    print(f"Empirically Calibrated Top-2 Margin Threshold: {calibrated_margin}")
    print(f"Updated Thresholds Configuration             : {out_json_path}")
    print(f"Calibration Curve Chart Saved To             : {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
