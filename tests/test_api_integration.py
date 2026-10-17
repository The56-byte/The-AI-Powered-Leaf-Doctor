"""
End-to-End Integration Tests for LeafDoctor Gateway & ML Services.
Tests live HTTP endpoints on port 5000 (Express Gateway) and port 8000 (FastAPI).
"""

import io
import unittest
import requests
from PIL import Image, ImageDraw


GATEWAY_URL = "http://localhost:5000"
FASTAPI_URL = "http://localhost:8000"


class TestLeafDoctorAPIIntegration(unittest.TestCase):
    def test_01_gateway_health(self):
        """Test Express gateway /api/health returns 200 and healthy status."""
        resp = requests.get(f"{GATEWAY_URL}/api/health", timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("gateway", data)
        self.assertIn("model_service", data)
        self.assertEqual(data["gateway"]["status"], "ok")
        self.assertEqual(data["model_service"]["status"], "ok")

    def test_02_supported_classes(self):
        """Test /api/classes returns the 10 cataloged crops and diseases."""
        resp = requests.get(f"{GATEWAY_URL}/api/classes", timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("classes", data)
        self.assertEqual(len(data["classes"]), 10)
        # Check presence of Apple, Potato, Tomato
        crops = {c["crop"] for c in data["classes"]}
        self.assertEqual(crops, {"Apple", "Potato", "Tomato"})

    def test_03_test_metrics_endpoint(self):
        """Test /api/metrics returns macro-F1, balanced accuracy, and sample count."""
        resp = requests.get(f"{GATEWAY_URL}/api/metrics", timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("macro_f1", data)
        self.assertIn("balanced_accuracy", data)
        self.assertIn("test_samples", data)
        self.assertGreater(data["test_samples"], 0)

    def test_04_reject_invalid_mime_type(self):
        """Test gateway strictly rejects non-image files (e.g. text/plain) with 400."""
        files = {
            "image": ("test.txt", io.BytesIO(b"This is not a leaf photo."), "text/plain")
        }
        resp = requests.post(f"{GATEWAY_URL}/api/predict", files=files, timeout=5)
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertEqual(data["status"], "invalid_file")
        self.assertIn("Invalid file type", data["message"])

    def test_05_reject_undersized_image(self):
        """Test gateway rejects tiny images (< 100x100 pixels) with 400."""
        img = Image.new("RGB", (50, 50), color=(100, 200, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        files = {"image": ("tiny.jpg", buf, "image/jpeg")}
        resp = requests.post(f"{GATEWAY_URL}/api/predict", files=files, timeout=5)
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertEqual(data["status"], "invalid_file")
        self.assertTrue("resolution" in data["message"].lower() or "too small" in data["message"].lower())

    def test_06_predict_valid_leaf_image(self):
        """Test gateway end-to-end prediction with a well-formed leaf photo."""
        img = Image.new("RGB", (256, 256), color=(45, 125, 45))
        draw = ImageDraw.Draw(img)
        # Draw veins and leaf structure
        draw.polygon([(128, 20), (220, 128), (128, 236), (36, 128)], fill=(60, 160, 60))
        draw.line([(128, 20), (128, 236)], fill=(30, 90, 30), width=3)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        files = {"image": ("leaf.jpg", buf, "image/jpeg")}
        resp = requests.post(f"{GATEWAY_URL}/api/predict", files=files, timeout=10)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        self.assertIn("top3", data)
        self.assertIn("uncertain", data)
        self.assertIn("disclaimer", data)
        self.assertTrue(len(data["disclaimer"]) > 10)


if __name__ == "__main__":
    unittest.main()
