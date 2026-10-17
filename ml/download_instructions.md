# 📥 PlantVillage Dataset Download & Setup Guide

This guide explains how to download the **PlantVillage** dataset, place it in the correct directory, and run the automated deduplication and data-splitting pipeline.

---

## 1. Where to Put the Dataset

Place your downloaded dataset in:
```
leafdoctor/ml/data/raw/
```

### Expected Folder Structure
Inside `leafdoctor/ml/data/raw/`, you will have folder names corresponding to our supported classes (or the full PlantVillage extraction; `prepare_data.py` will automatically filter only our target classes):

```
leafdoctor/
└── ml/
    └── data/
        └── raw/
            ├── Tomato___healthy/
            ├── Tomato___Early_blight/
            ├── Tomato___Late_blight/
            ├── Tomato___Leaf_Mold/
            ├── Potato___healthy/
            ├── Potato___Early_blight/
            ├── Potato___Late_blight/
            ├── Apple___healthy/
            ├── Apple___Apple_scab/
            └── Apple___Black_rot/
```

*(Note: `prepare_data.py` is flexible and automatically recognizes common folder naming variations such as `Tomato_healthy`, `Tomato___healthy`, or `tomato-healthy`.)*

---

## 2. Recommended Download Sources

### Option A: Kaggle (Fastest & Most Reliable)
1. Visit the Kaggle PlantVillage page:
   - **URL:** [https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease) (or search for `PlantVillage dataset` on Kaggle)
2. Click **Download (approx. 800MB–1.5GB)**.
3. Extract the ZIP file.
4. Copy the class folders directly into `leafdoctor/ml/data/raw/`.

Alternatively, using the Kaggle CLI:
```bash
kaggle datasets download -d emmarex/plantdisease
# Unzip contents into leafdoctor/ml/data/raw/
```

---

### Option B: Official GitHub Repository
You can clone or download raw folders directly from the open-source PlantVillage repository:
- **Repository:** [https://github.com/spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- Look in `raw/segmented/` or `raw/color/`.

---

### Option C: Quick Offline Testing (Dummy Mode)
If you want to test the entire training and evaluation pipeline immediately without downloading gigabytes of images:
```bash
cd leafdoctor/ml
python prepare_data.py --create_dummy_data
```
This generates a mini PlantVillage test set (20 synthetic leaf samples per class, with realistic variations, duplicates, and labels) so you can verify data splitting, training, and evaluation locally.

---

## 3. Running Data Preparation

Once the images are in `leafdoctor/ml/data/raw/`, execute:

```bash
cd leafdoctor/ml
python prepare_data.py
```

### What `prepare_data.py` does:
1. **Filters** only the 10 selected classes (or 6 if using `--use_starter_subset`).
2. **Deduplicates** exact image copies using MD5 hashing (crucial judge requirement).
3. **Groups** near-identical shots of the same leaf into clusters via perceptual hashing so they never straddle splits.
4. **Splits** data into:
   - **Train:** 70%
   - **Validation:** 15%
   - **Held-out Test:** 15% (Strictly isolated with `random_seed=42`)
5. **Outputs** reproducible CSV files:
   - `leafdoctor/ml/data/processed/train.csv`
   - `leafdoctor/ml/data/processed/val.csv`
   - `leafdoctor/ml/data/processed/test.csv`
   - `leafdoctor/ml/data/processed/split_summary.json`
