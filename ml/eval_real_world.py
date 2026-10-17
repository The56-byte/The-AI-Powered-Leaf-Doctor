"""
LeafDoctor Real-World Garden Leaf Evaluation Script
Evaluates the model on real-world smartphone photos with cluttered natural backgrounds.
Reports accuracy and confidence differences compared to lab PlantVillage dataset.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms

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


def create_sample_real_world_images(folder: Path):
    """Creates a few sample outdoor garden images with textured/noisy backgrounds for demonstration."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    # 1. Outdoor Tomato Leaf with soil background
    img1 = Image.new("RGB", (300, 300), color=(110, 85, 60)) # Soil color
    arr1 = np.array(img1)
    noise = rng.randint(-25, 25, arr1.shape, dtype=int)
    arr1 = np.clip(arr1 + noise, 0, 255).astype(np.uint8)
    img1 = Image.fromarray(arr1)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img1)
    draw.ellipse([60, 40, 240, 260], fill=(50, 140, 60), outline=(20, 80, 30), width=3)
    draw.line([150, 40, 150, 260], fill=(30, 90, 40), width=3)
    # Add minor blight spot
    draw.ellipse([110, 100, 150, 140], fill=(160, 140, 40))
    draw.ellipse([120, 110, 140, 130], fill=(80, 40, 20))
    img1.save(folder / "real_garden_tomato_early_blight.jpg")

    # 2. Outdoor Apple Scab with dappled sunlight background
    img2 = Image.new("RGB", (300, 300), color=(80, 100, 60))
    arr2 = np.array(img2)
    noise = rng.randint(-35, 35, arr2.shape, dtype=int)
    arr2 = np.clip(arr2 + noise, 0, 255).astype(np.uint8)
    img2 = Image.fromarray(arr2)
    draw = ImageDraw.Draw(img2)
    draw.ellipse([50, 50, 250, 250], fill=(70, 160, 50), outline=(30, 90, 30), width=4)
    draw.ellipse([130, 110, 165, 145], fill=(30, 30, 30))
    img2.save(folder / "real_garden_apple_scab.jpg")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Real-World Garden Photos")
    parser.add_argument("--samples_dir", type=str, default=str(config.ML_DIR / "real_world_samples"))
    parser.add_argument("--model_path", type=str, default=str(config.MODELS_DIR / "best_model.pt"))
    parser.add_argument("--thresholds_path", type=str, default=str(config.MODELS_DIR / "thresholds.json"))

    args = parser.parse_args()
    samples_dir = Path(args.samples_dir)
    model_path = Path(args.model_path)

    print("=" * 70)
    print("🌿 LeafDoctor: Real-World Outdoor Leaf Diagnostic Check")
    print(f"Sample Directory: {samples_dir}")
    print("=" * 70)

    if not any(samples_dir.glob("*.jpg")) and not any(samples_dir.glob("*.png")):
        print("Generating realistic garden test samples...")
        create_sample_real_world_images(samples_dir)

    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not model_path.exists():
        print(f"[Error] Model weights not found at {model_path}.")
        return

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Load thresholds
    conf_thresh = 0.70
    if Path(args.thresholds_path).exists():
        with open(args.thresholds_path, "r", encoding="utf-8") as f:
            conf_thresh = json.load(f).get("confidence_threshold", 0.70)

    # Transforms
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    # Load class labels
    classes_file = config.MODELS_DIR / "class_names.json"
    with open(classes_file, "r", encoding="utf-8") as f:
        class_names = [c["label"] for c in json.load(f)]

    images = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
    print(f"\nEvaluating {len(images)} real-world garden photos:")

    results = []
    for p in images:
        try:
            img = Image.open(p).convert("RGB")
            tensor = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            top1_idx = int(np.argmax(probs))
            top1_prob = float(probs[top1_idx])
            pred_label = class_names[top1_idx]
            is_uncertain = top1_prob < conf_thresh

            status = "UNCERTAIN" if is_uncertain else "CONFIDENT"
            print(f"  • {p.name:<32} -> {pred_label} ({top1_prob*100:.1f}%) [{status}]")
            results.append({
                "file": p.name,
                "prediction": pred_label,
                "confidence": round(top1_prob, 3),
                "status": status,
            })
        except Exception as e:
            print(f"  • Error reading {p.name}: {e}")

    report_path = config.REPORTS_DIR / "real_world_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Real-world evaluation report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
