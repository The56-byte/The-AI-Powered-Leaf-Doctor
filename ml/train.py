"""
LeafDoctor Model Training Pipeline
Fine-tunes MobileNetV2 using a two-stage transfer learning procedure:
Stage 1: Frozen backbone, trains custom classification head.
Stage 2: Unfreezes top convolutional blocks with low learning rate.
Tracks validation Macro-F1 and saves the best model checkpoint to ml/models/best_model.pt.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

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


class PlantVillageDataset(Dataset):
    """PyTorch Dataset loading images from prepare_data CSV files."""

    def __init__(self, csv_file: str, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = int(row["class_id"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to neutral canvas if an image fails to load
            image = Image.new("RGB", config.IMAGE_SIZE, (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_transforms():
    """
    Data Augmentation: Applied to TRAIN set only.
    Validation set uses strictly deterministic resizing and normalization.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE[0], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    return train_transform, val_transform


def build_model(num_classes: int, device: str) -> nn.Module:
    """Builds MobileNetV2 with custom classification head."""
    print("\n[Model] Loading pretrained MobileNetV2 backbone (ImageNet weights)...")
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    # In_features for MobileNetV2 classifier
    in_features = model.classifier[1].in_features

    # Replace classifier with custom head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model.to(device)


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float, float]:
    """Evaluates loss, accuracy, and macro-F1 on validation set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    total = len(data_loader.dataset)
    val_loss = running_loss / total
    val_acc = accuracy_score(all_targets, all_preds)
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return val_loss, val_acc, val_macro_f1


def train_one_epoch(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="LeafDoctor MobileNetV2 Transfer Learning Trainer")
    parser.add_argument("--train_csv", type=str, default=str(config.DATA_DIR / "processed" / "train.csv"))
    parser.add_argument("--val_csv", type=str, default=str(config.DATA_DIR / "processed" / "val.csv"))
    parser.add_argument("--output_model", type=str, default=str(config.MODELS_DIR / "best_model.pt"))
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs_stage1", type=int, default=config.EPOCHS_STAGE1)
    parser.add_argument("--epochs_stage2", type=int, default=config.EPOCHS_STAGE2)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 70)
    print("🌿 LeafDoctor MobileNetV2 Training Pipeline")
    print(f"Device: {device.upper()} | Seed: {config.RANDOM_SEED}")
    print(f"Train Dataset: {args.train_csv}")
    print(f"Val Dataset  : {args.val_csv}")
    print("=" * 70)

    # Verify input CSVs exist
    if not Path(args.train_csv).exists() or not Path(args.val_csv).exists():
        print(f"\n[Error] Split files not found! Please run 'python prepare_data.py' first.")
        sys.exit(1)

    # Load data
    train_transform, val_transform = get_data_transforms()
    train_dataset = PlantVillageDataset(args.train_csv, transform=train_transform)
    val_dataset = PlantVillageDataset(args.val_csv, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Compute class weights for loss function to handle class distribution
    train_labels = train_dataset.df["class_id"].values
    num_classes = len(np.unique(train_labels))
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total_samples = len(train_labels)
    class_weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(float))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"\nLoaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples across {num_classes} classes.")

    # Build model
    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    history = {
        "stage1": {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []},
        "stage2": {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []},
        "best_val_macro_f1": 0.0,
    }

    best_val_macro_f1 = 0.0
    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STAGE 1: Train Classification Head Only (Backbone Frozen)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: Feature Extraction (Backbone Frozen)")
    print(f"Training new classification head for {args.epochs_stage1} epochs with lr={config.LEARNING_RATE_STAGE1}...")
    print("=" * 70)

    # Freeze all backbone layers
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer_stage1 = torch.optim.Adam(model.classifier.parameters(), lr=config.LEARNING_RATE_STAGE1)

    for epoch in range(1, args.epochs_stage1 + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage1, device)
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["stage1"]["train_loss"].append(round(train_loss, 4))
        history["stage1"]["train_acc"].append(round(train_acc, 4))
        history["stage1"]["val_loss"].append(round(val_loss, 4))
        history["stage1"]["val_acc"].append(round(val_acc, 4))
        history["stage1"]["val_macro_f1"].append(round(val_f1, 4))

        is_best = val_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_f1
            torch.save(model, output_model_path)

        print(f"Epoch [{epoch:02d}/{args.epochs_stage1:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% Macro-F1: {val_f1:.4f} "
              f"{'★ (Best)' if is_best else ''}")

    # =========================================================================
    # STAGE 2: Fine-Tuning (Unfreeze Top Convolutional Blocks)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Fine-Tuning (Unfreezing Top Blocks)")
    print(f"Training for {args.epochs_stage2} epochs with reduced lr={config.LEARNING_RATE_STAGE2}...")
    print("=" * 70)

    # Unfreeze top layers of MobileNetV2 (features[14:] through classifier)
    for i, child in enumerate(model.features.children()):
        if i >= 14:
            for param in child.parameters():
                param.requires_grad = True

    optimizer_stage2 = torch.optim.Adam([
        {"params": model.features[14:].parameters(), "lr": config.LEARNING_RATE_STAGE2},
        {"params": model.classifier.parameters(), "lr": config.LEARNING_RATE_STAGE2 * 5},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=args.epochs_stage2)

    for epoch in range(1, args.epochs_stage2 + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage2, device)
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["stage2"]["train_loss"].append(round(train_loss, 4))
        history["stage2"]["train_acc"].append(round(train_acc, 4))
        history["stage2"]["val_loss"].append(round(val_loss, 4))
        history["stage2"]["val_acc"].append(round(val_acc, 4))
        history["stage2"]["val_macro_f1"].append(round(val_f1, 4))

        is_best = val_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_f1
            torch.save(model, output_model_path)

        print(f"Epoch [{epoch:02d}/{args.epochs_stage2:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% Macro-F1: {val_f1:.4f} "
              f"{'★ (Best)' if is_best else ''}")

    history["best_val_macro_f1"] = round(best_val_macro_f1, 4)

    # Save training history
    history_path = config.REPORTS_DIR / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"🎉 Training Complete!")
    print(f"Best Validation Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Model Checkpoint Saved  : {output_model_path}")
    print(f"Training History Report : {history_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
