"""
Unit Tests for LeafDoctor ML Inference Engine & Uncertainty Guardrails.
Tests quality verification (blur, brightness), classification contract,
and Responsible AI refusal mechanisms.
"""

import io
import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image, ImageDraw

# Add model_service to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "model_service"))

from model_service.inference import ModelInferenceEngine


class TestLeafDoctorInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize the inference engine once for tests."""
        cls.engine = ModelInferenceEngine()

    def test_class_catalog_integrity(self):
        """Verify engine loads 10 supported classes with correct schema."""
        self.assertEqual(len(self.engine.classes), 10)
        for cls_item in self.engine.classes:
            self.assertIn("index", cls_item)
            self.assertIn("label", cls_item)
            self.assertIn("crop", cls_item)
            self.assertIn("condition", cls_item)

    def test_thresholds_loaded(self):
        """Verify calibration thresholds are configured properly."""
        thresholds = self.engine.thresholds
        self.assertIn("confidence_threshold", thresholds)
        self.assertIn("margin_threshold", thresholds)
        self.assertGreater(thresholds["confidence_threshold"], 0.25)
        self.assertGreaterEqual(thresholds["margin_threshold"], 0.04)

    def test_blur_detection(self):
        """Ensure blurred/flat images trigger low quality flag."""
        # A completely flat/uniform image has 0 Laplacian variance
        flat_img = np.full((224, 224, 3), 120, dtype=np.uint8)
        ok, reason, metrics = self.engine.assess_image_quality(flat_img)

        self.assertFalse(ok, "Flat/blurred image should fail quality check")
        self.assertIn("blur", reason.lower())
        self.assertLess(metrics["blur_score"], self.engine.thresholds["blur_threshold"])

    def test_darkness_detection(self):
        """Ensure severely underexposed photos trigger quality rejection."""
        # Almost pitch-black image
        dark_img = np.full((224, 224, 3), 10, dtype=np.uint8)
        ok, reason, metrics = self.engine.assess_image_quality(dark_img)

        self.assertFalse(ok, "Dark image should fail quality check")
        self.assertIn("dark", reason.lower())
        self.assertLess(metrics["brightness"], self.engine.thresholds["min_brightness"])

    def test_overexposure_detection(self):
        """Ensure overexposed washed-out photos trigger quality rejection."""
        # Extremely bright white image
        bright_img = np.full((224, 224, 3), 250, dtype=np.uint8)
        ok, reason, metrics = self.engine.assess_image_quality(bright_img)

        self.assertFalse(ok, "Overexposed image should fail quality check")
        self.assertTrue("overexposed" in reason.lower() or "bright" in reason.lower())
        self.assertGreater(metrics["brightness"], self.engine.thresholds["max_brightness"])

    def test_high_quality_synthetic_image_passes_quality(self):
        """Ensure sharp, well-lit textured images pass image quality check."""
        # Generate image with high-contrast sharp grid patterns
        img = Image.new("RGB", (256, 256), color=(60, 150, 60))
        draw = ImageDraw.Draw(img)
        for i in range(0, 256, 16):
            draw.line([(i, 0), (i, 256)], fill=(20, 80, 20), width=2)
            draw.line([(0, i), (256, i)], fill=(180, 220, 80), width=2)

        img_np = np.array(img)
        ok, reason, metrics = self.engine.assess_image_quality(img_np)
        self.assertTrue(ok, f"Sharp patterned image should pass, got: {reason}")
        self.assertGreater(metrics["blur_score"], self.engine.thresholds["blur_threshold"])

    def test_inference_contract_and_disclaimer(self):
        """Ensure prediction response follows strict LeafDoctor API schema."""
        # Create a valid JPEG
        img = Image.new("RGB", (224, 224), color=(40, 140, 40))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(50, 50), (180, 180)], fill=(90, 180, 70), outline=(20, 70, 20))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        result = self.engine.predict(image_bytes)

        # Check required fields
        self.assertIn("status", result)
        self.assertIn("prediction", result)
        self.assertIn("confidence", result)
        self.assertIn("top3", result)
        self.assertIn("uncertain", result)
        self.assertIn("disclaimer", result)
        self.assertIn("quality", result)

        # Check confidence range
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        # Check top3 structure
        self.assertIsInstance(result["top3"], list)
        self.assertLessEqual(len(result["top3"]), 3)
        if len(result["top3"]) > 1:
            # Top-3 should be descending
            first_conf = result["top3"][0][1] if isinstance(result["top3"][0], list) else result["top3"][0]["confidence"]
            second_conf = result["top3"][1][1] if isinstance(result["top3"][1], list) else result["top3"][1]["confidence"]
            self.assertGreaterEqual(first_conf, second_conf)

        # Check responsible AI disclaimer is present
        self.assertTrue(len(result["disclaimer"]) > 10)
        self.assertIn("indication", result["disclaimer"].lower())


if __name__ == "__main__":
    unittest.main()
