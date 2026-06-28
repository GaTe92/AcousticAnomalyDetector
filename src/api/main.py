import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException

from src.inference.detector import AnomalyDetector

ARTIFACTS = Path(__file__).resolve().parents[2] / "models"
detector = AnomalyDetector(ARTIFACTS)

app = FastAPI(
    title="Acoustic Anomaly Detector",
    description="Detects anomalies in industrials machine sounds.",
    version="0.1.0",
)

@app.get("/health")
def health():
    """Simple liveness check - useful for deployment + quick testing."""
    return {"status": "ok", "threshold": detector.threshold}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept a WAV file, return anomaly score + decision"""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    # Save the upload to a temp file (librosa needs a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = detector.score(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return{
        "filensmr": file.filename,
        "anomaly_score": result["anomaly_score"],
        "is_anomaly": result["is_anomaly"],
        "threshold": result["threshold"],
        "n_windows": result["n_windows"],
    }