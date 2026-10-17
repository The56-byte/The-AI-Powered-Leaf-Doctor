"""
Generate demo test cases for:
1. Supported Leaf (Tomato Early Blight, Potato Late Blight)
2. Non-Leaf Image (Ceramic coffee mug / desk object)
3. Unsupported Plant Category (Corn / Maize leaf)
4. Ambiguous Leaf (Dual symptom overlap)
5. Blurry Leaf (Out of focus)
Saves to leafdoctor/client/public/samples/
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import shutil

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "client" / "public" / "samples"
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Supported Leaves
potato_sample = list((RAW_DIR / "Potato___Late_blight").glob("*.jpg"))
if potato_sample:
    shutil.copy(potato_sample[0], OUTPUT_DIR / "demo_supported_potato.jpg")

tomato_sample = RAW_DIR / "Tomato___Early_blight" / "sample_0001.jpg"
if tomato_sample.exists():
    shutil.copy(tomato_sample, OUTPUT_DIR / "demo_supported_tomato.jpg")
elif list((RAW_DIR / "Tomato___Early_blight").glob("*.jpg")):
    shutil.copy(list((RAW_DIR / "Tomato___Early_blight").glob("*.jpg"))[0], OUTPUT_DIR / "demo_supported_tomato.jpg")

# 2. Non-Leaf Image: A ceramic coffee mug on wooden table (no green/yellow foliage)
mug_img = Image.new("RGB", (320, 240), color=(180, 140, 100)) # wood table
draw = ImageDraw.Draw(mug_img)
# Mug body (ceramic blue/white)
draw.rounded_rectangle([100, 60, 220, 200], radius=15, fill=(40, 80, 160), outline=(20, 40, 90), width=3)
# Mug handle
draw.arc([190, 80, 250, 170], start=270, end=90, fill=(40, 80, 160), width=12)
# Coffee surface
draw.ellipse([110, 65, 210, 95], fill=(60, 30, 15))
mug_img.save(OUTPUT_DIR / "demo_non_leaf_cup.jpg", quality=92)

# 3. Unsupported Plant: Corn / Maize Leaf (long linear parallel monocot venation)
corn_img = Image.new("RGB", (320, 240), color=(50, 60, 50))
draw_corn = ImageDraw.Draw(corn_img)
# Broad parallel-veined strap leaf across diagonal
draw_corn.polygon([(20, 220), (80, 240), (300, 20), (240, 10)], fill=(90, 170, 60))
# Parallel straight monocot veins
for offset in range(-30, 40, 6):
    draw_corn.line([(20 + offset, 220), (240 + offset, 10)], fill=(120, 200, 90), width=2)
# Pale midrib
draw_corn.line([(50, 230), (270, 15)], fill=(200, 230, 150), width=4)
corn_img.save(OUTPUT_DIR / "demo_unsupported_corn.jpg", quality=92)

# 4. Ambiguous Leaf: Overlapping conflicting symptoms between Potato Early Blight and Apple Scab
ambig_img = Image.new("RGB", (320, 240), color=(40, 50, 40))
draw_amb = ImageDraw.Draw(ambig_img)
draw_amb.polygon([(60, 200), (260, 200), (160, 30)], fill=(70, 120, 45))
# Balanced lesion markers creating near-identical evidence for runner-ups
draw_amb.ellipse([100, 80, 150, 130], fill=(95, 90, 30), outline=(50, 30, 15), width=2)
draw_amb.ellipse([160, 110, 210, 160], fill=(125, 110, 40), outline=(70, 40, 20), width=2)
ambig_img.save(OUTPUT_DIR / "demo_ambiguous_leaf.jpg", quality=92)

# 5. Blurry Leaf: heavily Gaussian blurred
if potato_sample:
    with Image.open(potato_sample[0]) as pimg:
        blur_img = pimg.convert("RGB").filter(ImageFilter.GaussianBlur(radius=8))
        blur_img.save(OUTPUT_DIR / "demo_blurry_leaf.jpg", quality=90)

print(f"Generated all demo images in {OUTPUT_DIR}")
