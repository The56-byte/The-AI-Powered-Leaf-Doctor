"""
LeafDoctor FastAPI Model Service
Exposes /health, /classes, /thresholds, and /predict endpoints.
"""

import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import engine

app = FastAPI(
    title="LeafDoctor Model Service",
    description="Inference & Uncertainty Estimation Microservice for Plant Disease Classification",
    version="0.1.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "LeafDoctor Model Service",
        "status": "running",
        "docs": "/docs",
        "endpoints": ["/health", "/classes", "/thresholds", "/predict"],
    }


@app.get("/health")
def health_check():
    """Liveness and readiness check."""
    return {
        "status": "ok",
        "service": "leafdoctor-model-service",
        "model_loaded": engine.model is not None,
        "classes_count": len(engine.classes),
        "device": engine.device,
        "version": "0.1.0-phase0",
    }


@app.get("/classes")
def get_classes():
    """Returns the list of supported crops and conditions."""
    return {
        "count": len(engine.classes),
        "classes": engine.classes,
    }


@app.get("/thresholds")
def get_thresholds():
    """Returns calibrated uncertainty and quality thresholds."""
    return engine.thresholds


@app.post("/predict")
async def predict_leaf(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, verifies format,
    runs quality inspection and uncertainty-calibrated prediction.
    """
    # Verify content type
    if file.content_type and not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={
                "status": "invalid_file",
                "prediction": "Unknown",
                "confidence": 0.0,
                "top3": [],
                "uncertain": True,
                "message": f"Unsupported media type: '{file.content_type}'. Please upload a JPG or PNG image.",
                "disclaimer": "Initial indication only, not a professional diagnosis.",
            },
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # Max 5MB limit
        if len(contents) > 5 * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={
                    "status": "invalid_file",
                    "prediction": "Unknown",
                    "confidence": 0.0,
                    "top3": [],
                    "uncertain": True,
                    "message": "File exceeds the 5MB size limit. Please upload a compressed image.",
                    "disclaimer": "Initial indication only, not a professional diagnosis.",
                },
            )

        result = engine.predict(contents)
        return result

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "prediction": "Error",
                "confidence": 0.0,
                "top3": [],
                "uncertain": True,
                "message": f"Inference processing failed: {str(e)}",
                "disclaimer": "Initial indication only, not a professional diagnosis.",
            },
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting LeafDoctor Model Service on http://{host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False)
