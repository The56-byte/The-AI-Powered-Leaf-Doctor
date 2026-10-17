"""
LeafDoctor ML Configuration File
Central configuration for classes, datasets, model hyperparameters, and uncertainty thresholds.
"""

import os
from pathlib import Path

# Base Paths
ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent
MODELS_DIR = ML_DIR / "models"
REPORTS_DIR = ML_DIR / "reports"
DATA_DIR = ML_DIR / "data"

# Reproducibility
RANDOM_SEED = 42

# Image Processing Specifications
IMAGE_SIZE = (224, 224)
INPUT_CHANNELS = 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Split Ratios (Held-out Test Set)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Supported Classes: 10 Target Categories across 3 crops
# Format: PlantVillage folder name -> Human-friendly display label
SUPPORTED_CLASSES = [
    {
        "id": "tomato_healthy",
        "folder_name": "Tomato___healthy",
        "crop": "Tomato",
        "condition": "Healthy",
        "label": "Tomato - Healthy",
        "is_healthy": True,
    },
    {
        "id": "tomato_early_blight",
        "folder_name": "Tomato___Early_blight",
        "crop": "Tomato",
        "condition": "Early Blight",
        "label": "Tomato - Early Blight",
        "is_healthy": False,
    },
    {
        "id": "tomato_late_blight",
        "folder_name": "Tomato___Late_blight",
        "crop": "Tomato",
        "condition": "Late Blight",
        "label": "Tomato - Late Blight",
        "is_healthy": False,
    },
    {
        "id": "tomato_leaf_mold",
        "folder_name": "Tomato___Leaf_Mold",
        "crop": "Tomato",
        "condition": "Leaf Mold",
        "label": "Tomato - Leaf Mold",
        "is_healthy": False,
    },
    {
        "id": "potato_healthy",
        "folder_name": "Potato___healthy",
        "crop": "Potato",
        "condition": "Healthy",
        "label": "Potato - Healthy",
        "is_healthy": True,
    },
    {
        "id": "potato_early_blight",
        "folder_name": "Potato___Early_blight",
        "crop": "Potato",
        "condition": "Early Blight",
        "label": "Potato - Early Blight",
        "is_healthy": False,
    },
    {
        "id": "potato_late_blight",
        "folder_name": "Potato___Late_blight",
        "crop": "Potato",
        "condition": "Late Blight",
        "label": "Potato - Late Blight",
        "is_healthy": False,
    },
    {
        "id": "apple_healthy",
        "folder_name": "Apple___healthy",
        "crop": "Apple",
        "condition": "Healthy",
        "label": "Apple - Healthy",
        "is_healthy": True,
    },
    {
        "id": "apple_scab",
        "folder_name": "Apple___Apple_scab",
        "crop": "Apple",
        "condition": "Apple Scab",
        "label": "Apple - Apple Scab",
        "is_healthy": False,
    },
    {
        "id": "apple_black_rot",
        "folder_name": "Apple___Black_rot",
        "crop": "Apple",
        "condition": "Black Rot",
        "label": "Apple - Black Rot",
        "is_healthy": False,
    },
]

# Quick 6-class starter subset toggle for rapid baseline training
STARTER_SUBSET_IDS = [
    "tomato_healthy",
    "tomato_early_blight",
    "tomato_late_blight",
    "potato_healthy",
    "potato_early_blight",
    "potato_late_blight",
]

# Default Uncertainty & Quality Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.70  # Below this -> status = "uncertain"
DEFAULT_MARGIN_THRESHOLD = 0.18      # If (top1 - top2) < margin -> status = "uncertain" (ambiguous)
DEFAULT_BLUR_THRESHOLD = 70.0        # OpenCV Laplacian variance below this -> status = "low_quality"
MIN_BRIGHTNESS = 30.0                # Mean pixel intensity below this -> too dark
MAX_BRIGHTNESS = 245.0               # Mean pixel intensity above this -> overexposed

# Model Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 10
EARLY_STOPPING_PATIENCE = 4
BACKBONE_NAME = "mobilenet_v2"
