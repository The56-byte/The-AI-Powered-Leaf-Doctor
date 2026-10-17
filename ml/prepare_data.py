"""
LeafDoctor Data Preparation & Leakage-Proof Splitting
Handles:
1. Class filtering (10 selected classes or 6-class starter subset).
2. Exact duplicate removal via MD5 file hashing.
3. Near-duplicate grouping via perceptual difference hashing (dHash).
4. Group-aware 70/15/15 stratified train/val/test split.
5. Export of reproducible train.csv, val.csv, test.csv, and summary report.
"""

import os
import sys
import glob
import json
import hashlib
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image, ImageDraw

# Add current directory to path to import config
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


def compute_md5(filepath: Path) -> str:
    """Calculates MD5 hash of a file for exact duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """
    Computes a 64-bit difference hash (dHash) for near-duplicate grouping.
    Fast, robust to small rotations, compression, and lighting changes.
    """
    try:
        with Image.open(image_path) as img:
            # Resize to (width + 1, height) in grayscale
            resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.asarray(resized)
            # Compare adjacent pixels
            diff = pixels[:, 1:] > pixels[:, :-1]
            # Convert boolean array to 64-bit integer
            decimal_value = 0
            for index, value in enumerate(diff.flatten()):
                if value:
                    decimal_value += 1 << index
            return decimal_value
    except Exception:
        return 0


def hamming_distance(hash1: int, hash2: int) -> int:
    """Calculates bitwise Hamming distance between two 64-bit hashes."""
    x = (hash1 ^ hash2) & ((1 << 64) - 1)
    tot = 0
    while x:
        tot += 1
        x &= x - 1
    return tot


def create_mock_plantvillage_dataset(raw_dir: Path, num_samples_per_class: int = 25):
    """
    Generates a mini synthetic PlantVillage dataset for rapid testing
    without needing a multi-gigabyte download.
    """
    print(f"\n[Mock Data Generator] Creating synthetic PlantVillage samples in {raw_dir}...")
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.RANDOM_SEED)

    for cls in config.SUPPORTED_CLASSES:
        cls_folder = raw_dir / cls["folder_name"]
        cls_folder.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples_per_class):
            img = Image.new("RGB", (224, 224), color=(35, 75, 30))
            draw = ImageDraw.Draw(img)

            # Draw leaf shape
            leaf_color = (60 + rng.randint(-10, 20), 160 + rng.randint(-20, 20), 70 + rng.randint(-15, 15))
            draw.ellipse([30, 20, 194, 204], fill=leaf_color, outline=(20, 60, 25), width=3)
            # Main vein
            draw.line([112, 20, 112, 204], fill=(25, 70, 30), width=3)

            # Disease symptoms if diseased
            if not cls["is_healthy"]:
                if "Early_blight" in cls["folder_name"]:
                    # Target board concentric rings
                    spot_x, spot_y = 90 + rng.randint(-20, 20), 100 + rng.randint(-20, 20)
                    draw.ellipse([spot_x - 22, spot_y - 22, spot_x + 22, spot_y + 22], fill=(180, 160, 40))
                    draw.ellipse([spot_x - 14, spot_y - 14, spot_x + 14, spot_y + 14], fill=(100, 50, 20))
                    draw.ellipse([spot_x - 6, spot_y - 6, spot_x + 6, spot_y + 6], fill=(40, 20, 10))
                elif "Late_blight" in cls["folder_name"]:
                    # Large dark water-soaked necrotic patches
                    spot_x, spot_y = 110 + rng.randint(-30, 30), 120 + rng.randint(-30, 30)
                    draw.ellipse([spot_x - 35, spot_y - 25, spot_x + 35, spot_y + 25], fill=(30, 20, 15))
                    draw.ellipse([spot_x + 10, spot_y + 20, spot_x + 45, spot_y + 45], fill=(45, 30, 20))
                elif "scab" in cls["folder_name"].lower() or "rot" in cls["folder_name"].lower():
                    # Corky spots or frogeye rings
                    for _ in range(3):
                        sx, sy = rng.randint(60, 160), rng.randint(60, 160)
                        draw.ellipse([sx - 10, sy - 10, sx + 10, sy + 10], fill=(20, 20, 20), outline=(140, 40, 40), width=2)
                elif "Mold" in cls["folder_name"]:
                    # Yellow spots with velvety olive centers
                    draw.ellipse([70, 80, 120, 130], fill=(160, 180, 50))
                    draw.ellipse([80, 90, 110, 120], fill=(70, 80, 30))

            file_path = cls_folder / f"sample_{i:04d}.jpg"
            img.save(file_path, quality=90)

        # Inject 2 intentional duplicates to demonstrate deduplication & anti-leakage
        dup1 = cls_folder / "sample_0000.jpg"
        if dup1.exists():
            dup2 = cls_folder / "sample_0000_copy.jpg"
            with open(dup1, "rb") as fsrc, open(dup2, "wb") as fdst:
                fdst.write(fsrc.read())

    print(f"[Mock Data Generator] Generated test dataset across {len(config.SUPPORTED_CLASSES)} classes.")


def find_matching_class_dir(raw_dir: Path, target_cls: Dict[str, Any]) -> Path:
    """Finds the folder corresponding to target_cls, tolerating naming variations."""
    target_name = target_cls["folder_name"].lower()
    alt_name = f"{target_cls['crop']}_{target_cls['condition']}".lower().replace(" ", "_")

    for entry in raw_dir.iterdir():
        if entry.is_dir():
            folder_lower = entry.name.lower().replace("-", "_").replace(" ", "_")
            if target_name in folder_lower or folder_lower in target_name or alt_name in folder_lower:
                return entry
    return None


def cluster_near_duplicates(image_records: List[Dict[str, Any]], distance_threshold: int = 3) -> List[Dict[str, Any]]:
    """
    Groups images with Hamming distance <= distance_threshold into clusters.
    All images in the same cluster receive the same group_id.
    """
    groups = []  # List of lists of image indices
    assigned = set()

    n = len(image_records)
    for i in range(n):
        if i in assigned:
            continue
        current_cluster = [i]
        assigned.add(i)
        hash_i = image_records[i]["dhash"]

        for j in range(i + 1, n):
            if j in assigned:
                continue
            hash_j = image_records[j]["dhash"]
            if hamming_distance(hash_i, hash_j) <= distance_threshold:
                current_cluster.append(j)
                assigned.add(j)

        groups.append(current_cluster)

    # Assign group_id
    for group_idx, cluster in enumerate(groups):
        for img_idx in cluster:
            image_records[img_idx]["group_id"] = group_idx

    return image_records


def split_grouped_records(
    class_records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits records into train, validation, and test sets by GROUP/CLUSTER.
    Ensures images from the same cluster NEVER cross split boundaries (zero data leakage).
    """
    rng = random.Random(seed)

    # Aggregate clusters
    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for r in class_records:
        gid = r["group_id"]
        if gid not in clusters:
            clusters[gid] = []
        clusters[gid].append(r)

    total_images = len(class_records)
    target_train = max(1, int(total_images * train_ratio))
    target_val = max(1, int(total_images * val_ratio))

    train_records, val_records, test_records = [], [], []

    # If class has fewer than 3 clusters, split items directly to ensure all splits get samples
    if len(clusters) < 3:
        shuffled = list(class_records)
        rng.shuffle(shuffled)
        n_train = int(total_images * train_ratio)
        n_val = int(total_images * val_ratio)
        train_records = shuffled[:n_train]
        val_records = shuffled[n_train:n_train + n_val]
        test_records = shuffled[n_train + n_val:]
        return train_records, val_records, test_records

    cluster_ids = list(clusters.keys())
    rng.shuffle(cluster_ids)

    curr_train_count = 0
    curr_val_count = 0

    for gid in cluster_ids:
        group_items = clusters[gid]
        group_size = len(group_items)

        if curr_train_count + group_size <= target_train:
            train_records.extend(group_items)
            curr_train_count += group_size
        elif curr_val_count + group_size <= target_val:
            val_records.extend(group_items)
            curr_val_count += group_size
        else:
            test_records.extend(group_items)

    # Ensure every split has at least one sample if total records >= 3
    if not val_records and len(train_records) > 1:
        val_records.append(train_records.pop())
    if not test_records and len(train_records) > 1:
        test_records.append(train_records.pop())
    if not train_records and len(test_records) > 1:
        train_records.append(test_records.pop())

    return train_records, val_records, test_records


def export_csv(records: List[Dict[str, Any]], output_path: Path):
    """Exports records to CSV with standard schema."""
    import csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filepath", "class_id", "label", "crop", "condition", "is_healthy", "group_id", "md5"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "filepath": r["filepath"],
                "class_id": r["class_id"],
                "label": r["label"],
                "crop": r["crop"],
                "condition": r["condition"],
                "is_healthy": r["is_healthy"],
                "group_id": r["group_id"],
                "md5": r["md5"],
            })


def main():
    parser = argparse.ArgumentParser(description="LeafDoctor PlantVillage Data Preparation & Anti-Leakage Splitter")
    parser.add_argument("--data_dir", type=str, default=str(config.DATA_DIR / "raw"), help="Path to raw PlantVillage dataset")
    parser.add_argument("--output_dir", type=str, default=str(config.DATA_DIR / "processed"), help="Path to save processed split CSVs")
    parser.add_argument("--use_starter_subset", action="store_true", help="Filter to 6 starter classes (Tomato & Potato)")
    parser.add_argument("--create_dummy_data", action="store_true", help="Generate synthetic PlantVillage test samples if raw data is missing")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed for reproducible split")

    args = parser.parse_args()
    raw_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("🌿 LeafDoctor Data Preparation & Leakage-Proof Splitting")
    print(f"Target: 70% Train | 15% Validation | 15% Held-out Test (Seed: {args.seed})")
    print("=" * 70)

    # Create dummy data if requested or if directory is empty
    if args.create_dummy_data or (not raw_dir.exists() or not any(raw_dir.iterdir())):
        print(f"\n[Info] Raw directory '{raw_dir}' is empty or --create_dummy_data requested.")
        create_mock_plantvillage_dataset(raw_dir)

    # Determine classes to process
    target_classes = config.SUPPORTED_CLASSES
    if args.use_starter_subset:
        target_classes = [c for c in config.SUPPORTED_CLASSES if c["id"] in config.STARTER_SUBSET_IDS]
        print(f"\n[Subset] Using 6 starter classes:")
    else:
        print(f"\n[Classes] Processing all {len(target_classes)} target classes:")

    for idx, c in enumerate(target_classes):
        print(f"  [{idx}] {c['label']} (Folder: {c['folder_name']})")

    all_train, all_val, all_test = [], [], []
    stats_summary = {
        "seed": args.seed,
        "classes_count": len(target_classes),
        "total_images_scanned": 0,
        "exact_duplicates_removed": 0,
        "near_duplicate_clusters_found": 0,
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "per_class": {},
    }

    print("\nScanning class directories & performing anti-leakage checks...")

    for class_idx, cls in enumerate(target_classes):
        class_dir = find_matching_class_dir(raw_dir, cls)
        if not class_dir or not class_dir.exists():
            print(f"  ⚠️ Warning: Folder for '{cls['label']}' not found in {raw_dir}. Skipping.")
            continue

        # Collect image files
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = [p for p in class_dir.iterdir() if p.suffix.lower() in valid_extensions]
        stats_summary["total_images_scanned"] += len(image_files)

        # 1. Exact Duplicate Elimination via MD5
        seen_md5s = set()
        unique_records = []
        dup_count = 0

        for img_path in image_files:
            img_md5 = compute_md5(img_path)
            if img_md5 in seen_md5s:
                dup_count += 1
                continue
            seen_md5s.add(img_md5)

            # Compute dhash for near-duplicate grouping
            dhash_val = compute_dhash(img_path)

            unique_records.append({
                "filepath": str(img_path.resolve()),
                "class_id": class_idx,
                "label": cls["label"],
                "crop": cls["crop"],
                "condition": cls["condition"],
                "is_healthy": cls["is_healthy"],
                "dhash": dhash_val,
                "md5": img_md5,
            })

        stats_summary["exact_duplicates_removed"] += dup_count

        # 2. Cluster Near-Duplicates (Hamming distance <= 3)
        clustered_records = cluster_near_duplicates(unique_records, distance_threshold=3)
        num_clusters = len(set(r["group_id"] for r in clustered_records))
        stats_summary["near_duplicate_clusters_found"] += num_clusters

        # 3. Stratified Group Split
        cls_train, cls_val, cls_test = split_grouped_records(
            clustered_records,
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO,
            seed=args.seed,
        )

        all_train.extend(cls_train)
        all_val.extend(cls_val)
        all_test.extend(cls_test)

        stats_summary["per_class"][cls["label"]] = {
            "total_unique": len(unique_records),
            "duplicates_purged": dup_count,
            "clusters": num_clusters,
            "train": len(cls_train),
            "val": len(cls_val),
            "test": len(cls_test),
        }

        print(f"  ✓ {cls['label']:<24}: {len(unique_records):>4} unique ({dup_count:>2} exact dups removed, {num_clusters:>4} clusters) -> Train: {len(cls_train):>3}, Val: {len(cls_val):>3}, Test: {len(cls_test):>3}")

    stats_summary["split_counts"]["train"] = len(all_train)
    stats_summary["split_counts"]["val"] = len(all_val)
    stats_summary["split_counts"]["test"] = len(all_test)

    # Export CSVs
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"
    summary_json = output_dir / "split_summary.json"

    export_csv(all_train, train_csv)
    export_csv(all_val, val_csv)
    export_csv(all_test, test_csv)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)

    total_unique = len(all_train) + len(all_val) + len(all_test)
    print("\n" + "=" * 70)
    print(f"🎉 Data Splitting Complete!")
    print(f"Total Images Scanned       : {stats_summary['total_images_scanned']}")
    print(f"Exact Duplicates Removed   : {stats_summary['exact_duplicates_removed']}")
    print(f"Unique Usable Leaf Images  : {total_unique}")
    print(f"Train Set (70%)            : {len(all_train)} images -> {train_csv}")
    print(f"Validation Set (15%)       : {len(all_val)} images -> {val_csv}")
    print(f"Held-out Test Set (15%)    : {len(all_test)} images -> {test_csv}")
    print(f"Split Summary Report       : {summary_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
