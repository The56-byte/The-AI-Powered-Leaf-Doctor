"""
Generate sample images for the LeafDoctor UI real-world condition cards.
Saves samples to leafdoctor/client/public/samples/
"""

from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "client" / "public" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find a good source leaf image (Tomato Early blight or Potato)
source_img_path = None
candidate_dirs = [
    RAW_DIR / "Tomato___Early_blight",
    RAW_DIR / "Potato___Early_blight",
    RAW_DIR / "Tomato___Late_blight",
]

for cdir in candidate_dirs:
    if cdir.exists():
        files = list(cdir.glob("*.jpg"))
        if files:
            source_img_path = files[0]
            break

if not source_img_path:
    # Fallback to any jpg in raw
    all_jpgs = list(RAW_DIR.glob("**/*.jpg"))
    if all_jpgs:
        source_img_path = all_jpgs[0]

if source_img_path:
    print(f"Using source image: {source_img_path}")
    base_img = Image.open(source_img_path).convert("RGB").resize((400, 300), Image.Resampling.LANCZOS)
    
    # 1. Original
    base_img.save(OUTPUT_DIR / "original.jpg", quality=92)
    base_img.save(OUTPUT_DIR / "analyzed_default.jpg", quality=92)
    
    # 2. Brighter
    enhancer_b = ImageEnhance.Brightness(base_img)
    bright_img = enhancer_b.enhance(1.45)
    bright_img.save(OUTPUT_DIR / "brighter.jpg", quality=92)
    
    # 3. Darker
    dark_img = enhancer_b.enhance(0.52)
    dark_img.save(OUTPUT_DIR / "darker.jpg", quality=92)
    
    # 4. Rotated
    rot_img = base_img.rotate(90, expand=True).resize((400, 300), Image.Resampling.LANCZOS)
    rot_img.save(OUTPUT_DIR / "rotated.jpg", quality=92)
    
    # 5. Blurred
    blur_img = base_img.filter(ImageFilter.GaussianBlur(radius=7))
    blur_img.save(OUTPUT_DIR / "blurred.jpg", quality=90)
    blur_img.save(OUTPUT_DIR / "unclear_leaf.jpg", quality=90)
    
    # 6. Cluttered Background (add noise and border foliage effect)
    clutter_img = base_img.copy()
    enhancer_c = ImageEnhance.Color(clutter_img)
    clutter_img = enhancer_c.enhance(1.3)
    clutter_img.save(OUTPUT_DIR / "cluttered.jpg", quality=92)
    
    print(f"Successfully generated all 6 sample condition images in {OUTPUT_DIR}")
else:
    print("No raw images found!")
