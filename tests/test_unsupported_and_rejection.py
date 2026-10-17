"""
Comprehensive Test Suite for Unsupported Input Detection & Reliable Predictions.
Deliverable #6: Tests supported leaves, non-leaf images, unsupported plant classes,
and ambiguous/low-quality leaves against the live API gateway.
"""

import unittest
import requests
from pathlib import Path

GATEWAY_URL = "http://localhost:5000"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "client" / "public" / "samples"


class TestUnsupportedInputAndReliability(unittest.TestCase):
    def test_01_supported_prediction_success(self):
        """
        Deliverable #7: Successful supported prediction.
        Verifies Potato Late Blight is recognized confidently with is_supported=True.
        """
        img_path = SAMPLES_DIR / "demo_supported_potato.jpg"
        self.assertTrue(img_path.exists(), "Sample supported image must exist")

        with open(img_path, "rb") as f:
            files = {"image": ("potato_leaf.jpg", f, "image/jpeg")}
            resp = requests.post(f"{GATEWAY_URL}/api/predict", files=files, timeout=10)

        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "ok")
        self.assertFalse(data["uncertain"])
        self.assertTrue(data.get("is_supported", False))
        self.assertGreater(data["confidence"], 0.35)
        self.assertIn("Potato", data["prediction"])
        self.assertIn("disclaimer", data)
        self.assertIn("explanation", data)

    def test_02_non_leaf_rejection(self):
        """
        Deliverable #1 & #3: Non-leaf image detection.
        Verifies everyday non-leaf objects (ceramic mug) are rejected with non_leaf status.
        """
        img_path = SAMPLES_DIR / "demo_non_leaf_cup.jpg"
        self.assertTrue(img_path.exists(), "Sample non-leaf image must exist")

        with open(img_path, "rb") as f:
            files = {"image": ("coffee_mug.jpg", f, "image/jpeg")}
            resp = requests.post(f"{GATEWAY_URL}/api/predict", files=files, timeout=10)

        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "non_leaf")
        self.assertTrue(data["uncertain"])
        self.assertFalse(data.get("is_supported", True))
        self.assertEqual(data.get("rejection_category"), "non_leaf")
        self.assertIn("Non-Leaf", data["message"])
        self.assertIn("explanation", data)
        self.assertIn("limitation", data)

    def test_03_unsupported_plant_category_rejection(self):
        """
        Deliverable #1 & #3: Unsupported plant category detection.
        Verifies out-of-scope crops (e.g. corn leaf) trigger unsupported/uncertain status.
        """
        img_path = SAMPLES_DIR / "demo_unsupported_corn.jpg"
        self.assertTrue(img_path.exists(), "Sample unsupported crop image must exist")

        with open(img_path, "rb") as f:
            files = {"image": ("corn_leaf.jpg", f, "image/jpeg")}
            resp = requests.post(f"{GATEWAY_URL}/api/predict", files=files, timeout=10)

        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Must flag uncertain and reject from supported scope
        self.assertTrue(data["uncertain"])
        self.assertFalse(data.get("is_supported", True))
        self.assertIn(data["status"], ["unsupported_plant", "uncertain", "ambiguous"])
        self.assertIn("limitation", data)
        self.assertIn("Tomato, Potato", data["limitation"])

    def test_04_ambiguous_or_unclear_leaf_rejection(self):
        """
        Deliverable #1, #2 & #5: Highly ambiguous leaf symptom rejection.
        Verifies leaf with tied/conflicting lesion markers is flagged with explanation.
        """
        img_path = SAMPLES_DIR / "demo_ambiguous_leaf.jpg"
        self.assertTrue(img_path.exists(), "Sample ambiguous image must exist")

        with open(img_path, "rb") as f:
            files = {"image": ("ambiguous_leaf.jpg", f, "image/jpeg")}
            resp = requests.post(f"{GATEWAY_URL}/api/predict", files=files, timeout=10)

        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertTrue(data["uncertain"])
        self.assertFalse(data.get("is_supported", True))
        self.assertIn("explanation", data)
        self.assertIn("limitation", data)

    def test_05_blurry_image_quality_rejection(self):
        """
        Deliverable #2 & #5: Blur quality rejection rule.
        Verifies blurry image is rejected with Laplacian explanation and retake guidance.
        """
        img_path = SAMPLES_DIR / "demo_blurry_leaf.jpg"
        self.assertTrue(img_path.exists(), "Sample blurry image must exist")

        with open(img_path, "rb") as f:
            files = {"image": ("blurry_leaf.jpg", f, "image/jpeg")}
            resp = requests.post(f"{GATEWAY_URL}/api/predict", files=files, timeout=10)

        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "low_quality")
        self.assertTrue(data["uncertain"])
        self.assertEqual(data.get("rejection_category"), "blur")
        self.assertIn("blurry", data["message"].lower())
        self.assertLess(data["quality"]["blur_score"], 65.0)


if __name__ == "__main__":
    unittest.main()
