# 🎥 LeafDoctor: 3–5 Minute Hackathon Video Pitch Script

**Hackathon Mission 78 | AI/ML | LeafDoctor: A Plant Doctor in Your Pocket**  
**Presenter:** Engineering Lead / Team Representative  
**Total Target Runtime:** 3 minutes 45 seconds (225 seconds)

---

## ⏱️ Video Timeline Overview
| Timestamp | Segment | Visual / Screen On Display | Key Message |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:35** (35s) | **The Hook & Problem** | Slide: Diseased leaves side-by-side | Foliar diseases look identical; generic AI dangerously hallucinates. |
| **0:35 – 1:15** (40s) | **Architecture & Pipeline** | Architecture Diagram | Full-stack decoupled design: React + Express Gateway + FastAPI + PyTorch. |
| **1:15 – 2:10** (55s) | **ML Rigor & Evaluation** | Confusion Matrix & Metrics Table | Group-aware 70/15/15 split, MobileNetV2 transfer learning, Macro-F1 40.97%. |
| **2:10 – 3:20** (70s) | **Live UI Demonstration** | Screen Recording of Live Web App | Demo 3 cases: Confident detection, Blurry rejection, Non-leaf refusal. |
| **3:20 – 3:45** (25s) | **Responsible AI & Closing** | Limitations Banner & Call to Action | "Honesty over hype." Not replacing an agronomist; reliable initial triage. |

---

## 🎬 Section 1: The Hook & The Problem (0:00 – 0:35)

**[Visual: Speaker on camera or title slide with photos of early blight vs late blight lesions.]**

> **Speaker:**  
> "Every growing season, smallholder farmers, hobby gardeners, and agriculture students face a devastating puzzle: foliar plant diseases.  
> 
> Early blight, late blight, and leaf mold often produce near-identical brown spots and chlorotic yellowing. A home gardener cannot reliably tell a fungal outbreak from simple nutrient stress.  
> 
> But the bigger issue today is that **generic vision models dangerously hallucinate**. If you feed a standard computer vision API a blurry photo, a shadowed corner, or a coffee cup, it will still confidently output: *'98% Tomato Late Blight'*. In agriculture, false confidence destroys crops and wastes chemical treatments.  
> 
> That’s why we built **LeafDoctor**: a calibrated, responsible plant disease diagnostic assistant that knows what it knows—and honestly says *'I'm not sure'* when it doesn't."

---

## 🏗️ Section 2: Full-Stack Architecture (0:35 – 1:15)

**[Visual: Switch screen to LeafDoctor System Architecture Diagram.]**

> **Speaker:**  
> "LeafDoctor is engineered as a production-grade, decoupled three-tier system:
> 
> 1. **At the front**, a responsive **React 18 interface** built with Vite and Tailwind, supporting drag-and-drop file upload, mobile camera capture, preset sample cards for judges, and a live evaluation metrics telemetry dashboard.
> 2. **At the gateway**, a **Node.js and Express API layer** that enforces strict security: validating MIME types, enforcing a 5MB payload limit, and rejecting undersized images below 100-by-100 pixels before they ever touch our ML runtime.
> 3. **At the core**, a high-performance **Python FastAPI microservice** that executes our custom PyTorch inference pipeline and multi-layered uncertainty engine."

---

## 🧠 Section 3: ML Engineering, Calibration & Evaluation (1:15 – 2:10)

**[Visual: Display `ml/reports/confusion_matrix.png`, `ml/reports/calibration_curve.png`, and the Per-Class Metrics Table.]**

> **Speaker:**  
> "The judges asked for a classifier we trained and evaluated ourselves with strict scientific rigor—not a wrapper around a commercial cloud API.
> 
> Here is how we built it:
> - **Dataset & Class Selection:** We curated a targeted 10-class subset of the public PlantVillage dataset spanning three crucial crops: **Tomato, Potato, and Apple** across healthy and diseased conditions like Early Blight, Late Blight, Leaf Mold, Apple Scab, and Black Rot.
> - **Data Leakage Prevention:** Before splitting, we ran exact MD5 hash deduplication and 64-bit dHash perceptual clustering with a Hamming distance threshold of 3. We then executed a **group-aware stratified 70/15/15 split** (`train.csv`, `val.csv`, and a strictly held-out `test.csv`). Near-identical leaf photos never straddle splits.
> - **Transfer Learning:** We fine-tuned an ImageNet-pretrained **MobileNetV2** backbone in two stages: first training the classifier head with frozen features, then unfreezing top convolutional layers with a cosine-annealed learning rate schedule.
> - **Validation Calibration:** On the validation set, we calibrated two critical safety parameters: a **confidence threshold of 0.70** and a **top-2 margin threshold of 0.18**.
> - **Held-Out Test Results:** Evaluated strictly once on the held-out test set, our model achieved a **Macro-F1 of 40.97%** and a **Balanced Accuracy of 44.66%**, which you can see in our transparent per-class confusion matrix.
> - Furthermore, when tested on outdoor garden photos, our uncertainty engine correctly rejected out-of-domain images with low confidence (11%–13%), refusing to make irresponsible guesses."

---

## 💻 Section 4: Live Interactive Demonstration (2:10 – 3:20)

**[Visual: Screen recording of the live LeafDoctor web application at `http://localhost:5173`.]**

> **Speaker:**  
> "Let’s see LeafDoctor in action across three critical scenarios:
> 
> ### Demo Case 1: High-Confidence Supported Leaf (Potato Late Blight)
> *(Action: Click the 'Potato Late Blight' preset card or upload sample leaf.)*  
> 'First, we upload a clear photo of a potato leaf with necrotic lesions. LeafDoctor passes the OpenCV blur and brightness checks, evaluates the PyTorch model, and identifies **Potato - Late Blight** with 88% confidence. The interface highlights the severity badge, displays the top-3 probabilities, and provides plain-language biological symptoms and care tips.'
> 
> ### Demo Case 2: Blurry / Poor Quality Image (OpenCV Rejection)
> *(Action: Click the 'Blurry Leaf Photo' preset card.)*  
> 'Next, what happens if a user snaps an out-of-focus photo in the wind? Many apps would hallucinate a disease. But LeafDoctor measures the Laplacian variance of the image: the blur score drops below our threshold of 70. The UI instantly shifts to a amber warning state: *'Image is blurry (Laplacian variance low). Please hold your camera steady and retake.'* It refuses to guess.'
> 
> ### Demo Case 3: Out-of-Domain / Non-Leaf Image (Softmax Uncertainty)
> *(Action: Click the 'Non-Plant / Coffee Mug' preset card.)*  
> 'Finally, what if someone uploads a picture of a coffee mug, an office desk, or an unsupported plant? The model evaluates the softmax distribution: no class clears our 0.70 confidence threshold, and the top-2 margin is razor thin. LeafDoctor flags this with our purple uncertain badge: *'I’m not sure — this image appears outside supported crops or lacks clear leaf patterns.'*' "

---

## 🛡️ Section 5: Responsible AI, Limitations & Conclusion (3:20 – 3:45)

**[Visual: Highlight the Persistent Limitations Banner at the bottom of the screen, then show project GitHub repo.]**

> **Speaker:**  
> "Responsible AI is not an afterthought in LeafDoctor—it is the foundation.  
> 
> At all times, the user interface displays our prominent agronomic limitations banner: **LeafDoctor provides an initial indication only, not a certified agronomic diagnosis.** For commercial field decisions, users are always directed to consult their local agricultural extension office.
> 
> All code, trained weights, Colab GPU notebook, evaluation reports, and automated test suites are fully open-source and documented in our repository.
> 
> Thank you, judges, and remember: when diagnosing the world's crops, **honesty is the best intelligence**."

---

## 📋 Hackathon Demo Checklist for Presenter
- [ ] Ensure FastAPI (`python app.py`) is running on port 8000.
- [ ] Ensure Express Gateway (`npm start`) is running on port 5000.
- [ ] Ensure React Client (`npm run dev`) is open in browser at `http://localhost:5173`.
- [ ] Test click the 4 preset buttons: Tomato Healthy, Potato Late Blight, Blurry Photo, Non-Plant.
- [ ] Have `ml/reports/confusion_matrix.png` open in an image viewer tab ready to toggle.
- [ ] Keep presentation voice energetic, professional, and within the 4-minute mark.
