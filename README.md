# 🔊 Acoustic Anomaly Detector

> Detect anomalies in industrial machine sounds with a convolutional autoencoder,
> served through a REST API with an explainable diagnosis layer.

Upload a `.wav` recording of a machine (fan, pump, valve…) and the system returns
an anomaly score, a normal/abnormal decision, and a plain-language diagnosis.

---

## What it does

```
 WAV upload ──▶ Mel-spectrogram ──▶ Conv-Autoencoder ──▶ reconstruction error
                                                              │
   plain-text   ◀── Diagnosis layer ◀── anomaly score ◀──  band-weighted scoring
   diagnosis         (rule-based now,        + threshold
                      LLM-ready)
```

The model is trained **only on healthy machine sounds**. Anything it cannot
reconstruct well is flagged as anomalous — an unsupervised approach that needs
no labelled fault data.

---

## Architecture

```
┌────────────┐     HTTP      ┌──────────────────┐
│  Vue 3 +   │  POST /predict│   FastAPI         │
│ TypeScript │ ─────────────▶│   (thin HTTP)     │
│  frontend  │ ◀───────────  │                   │
└────────────┘   JSON result └─────────┬─────────┘
                                        │ calls
                              ┌─────────▼─────────┐
                              │  AnomalyDetector  │  inference layer
                              │  WAV → score      │  (single source of truth)
                              └─────────┬─────────┘
                                        │ uses
                    ┌───────────────────┼───────────────────┐
              ┌─────▼─────┐      ┌───────▼──────┐     ┌───────▼──────┐
              │ autoencoder│     │ norm_params  │     │ band_weights │
              │  .keras    │     │ threshold    │     │              │
              └────────────┘     └──────────────┘     └──────────────┘
                            stored model artifacts
```

| Layer | Technology |
|---|---|
| Feature engineering | LibROSA, Mel-spectrogram (128 bands) |
| Model | TensorFlow / Keras — convolutional autoencoder (~12.8k params) |
| Inference | Reusable Python module (`src/inference`) |
| API | FastAPI + Uvicorn |
| Diagnosis | Strategy pattern: rule-based provider (LLM-ready interface) |
| Frontend | Vue 3 + TypeScript + Vite |
| Dataset | [MIMII](https://zenodo.org/record/3384388) — fan, 0 dB |

---

## Results (honest, leakage-free)

Evaluated on held-out test files, with band weights learned on a **separate**
calibration split (no data leakage), averaged over 50 random splits:

| Metric | Value |
|---|---|
| ROC-AUC (per file) | **0.77 ± 0.04** (range 0.70–0.88) |
| F1 (at Youden threshold) | ~0.85 |
| Precision / Recall | 0.86 / 0.84 |

This sits at the **upper end of typical autoencoder baselines** for MIMII fan
at 0 dB. The full reasoning — why aggregation, why band-weighting, why these
numbers — is documented in [`LESSONS.md`](LESSONS.md).

> **A note on honesty:** the shipped model learns its band weights on all test
> data (it is the *deliverable artifact*, not the *evaluation*). The reported
> 0.77 ± 0.04 is the leakage-free number measured on unseen data. The two are
> deliberately kept distinct.

---

## Quick start

Requires Python 3.12 and Node 22.

```bash
# Backend
pip install -r requirements.txt
uvicorn src.api.main:app --reload          # → http://localhost:8000/docs

# Frontend (separate terminal)
cd frontend && npm install && npm run dev   # → http://localhost:5173
```

Upload a `.wav` from `data/mimii/fan/id_00/test/` and analyse.

> The dataset itself is not in the repo (too large). Download MIMII fan from
> Zenodo and run `src/Data_Separation/setup_data.py` to create the splits.

---

## Project layout

```
src/
├── inference/   detector.py   — WAV → score (used by API + scripts)
├── diagnosis/   provider.py   — score → human-readable text
└── api/         main.py       — FastAPI endpoints
models/          trained artifacts (model, norm params, weights, threshold)
notebooks/       01 exploration · 02 features · 03 training & scoring
frontend/        Vue 3 + TS upload UI
```

---

## Quick start
See [`COMMANDS.md`](COMMANDS.md) for the full command reference.

## Related work

Predecessor project: **Bearing Fault Detection** (CNN classifier, CWRU dataset) —
this project extends that signal-processing + CNN foundation into an
unsupervised, deployable, end-to-end product.
