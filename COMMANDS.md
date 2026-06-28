# Commands — Cheatsheet

All commands run from the repository root unless noted otherwise.
Requires **Python 3.12** and **Node 22**.

---

## 1. One-time setup

```bash
# clone
git clone <your-repo-url>
cd AcousticAnomalyDetector

# python environment (venv)
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# frontend dependencies
cd frontend && npm install && cd ..
```

> **macOS / Apple Silicon:** use the native arm64 Python
> (`/opt/homebrew/bin/python3.12`) to avoid TensorFlow AVX errors.

---

## 2. Get the data (not included in the repo)

```bash
# 1. download MIMII "fan" (0 dB) from Zenodo:
#    https://zenodo.org/record/3384388
# 2. unzip it, then adjust SOURCE_ROOT inside the script and run:
python src/Data_Separation/setup_data.py
```

Expected output: `train/normal: 860 · test/normal: 151 · test/abnormal: 407`.

---

## 3. (Optional) retrain the model

The trained artifacts are already committed in `models/`, so this is optional.

```bash
# run the notebooks in order (Jupyter or VS Code):
#   notebooks/01_dataset_exploration.ipynb
#   notebooks/02_feature_engineering.ipynb
#   notebooks/03_model_training.ipynb   → writes models/*.npy + autoencoder.keras
jupyter notebook
```

---

## 4. Run the backend

```bash
source .venv/bin/activate
uvicorn src.api.main:app --reload
```

- API docs (interactive): http://localhost:8000/docs
- Health check:           http://localhost:8000/health

---

## 5. Run the frontend

```bash
cd frontend
npm run dev
```

- UI: http://localhost:5173
  (Backend must be running too — both servers at the same time.)

---

## 6. Quick tests (no browser)

```bash
# score one normal + one abnormal file directly via the inference module
python -c "
from pathlib import Path
from src.inference.detector import AnomalyDetector
det = AnomalyDetector(Path('models'))
for label, folder in [('NORMAL','test/normal'), ('ABNORMAL','test/abnormal')]:
    f = sorted(Path(f'data/mimii/fan/id_00/{folder}').glob('*.wav'))[0]
    r = det.score(f)
    print(f'{label:9} score={r[\"anomaly_score\"]:.6f}  is_anomaly={r[\"is_anomaly\"]}')
"
```

```bash
# call the API from the terminal (server must be running)
curl -X POST http://localhost:8000/predict \
  -F "file=@data/mimii/fan/id_00/test/abnormal/00000067.wav"
```

---

## 7. Common pitfalls

| Symptom | Fix |
|---|---|
| `command not found: npm` in IDE terminal | use a login shell / real terminal (PATH not loaded) |
| CORS error in browser console | backend must allow `http://localhost:5173` (already configured) |
| `ModuleNotFoundError` after install | half-installed package — `pip install --force-reinstall <pkg>` |
| TensorFlow AVX error on Mac | use native arm64 Python, not Rosetta x86 |
| frontend loads but "Analyze" fails | is the backend (port 8000) running? |
