"""
Acoustics Anomaly Detector - Inference module

Connects raw WAV files to the trained CNN and produces an anomaly score.
This is the single source of truth for the WAV -> score pipeline,
used by both the API and any standalone script.
"""

from pathlib import Path
import numpy as np
import librosa
import tensorflow as tf

# Feature config
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 16_000
HOP_SAMPLES = 8_000
N_MELS = 128
FMAX = 8_000

class AnomalyDetector:
    """Loads the trained atrifacts once and scores WAV files on demand."""

    def __init__(self, artifacts_dir: Path):
        artifacts_dir = Path(artifacts_dir)
        self.model = tf.keras.models.load_model(artifacts_dir / "autoencoder.keras")

        norm = np.load(artifacts_dir / "norm_params.npy")
        self.train_min, self.train_max = float(norm[0]), float(norm[1])

        self.band_weights = np.load(artifacts_dir / "band_weights.npy")

    # Feature extraction
    def _extract_windows(self, y: np.ndarray) -> np.ndarray:
        """Audio signal -> array of normalized Mel-Spectogram windows"""
        windows = []
        start = 0
        while start + WINDOW_SAMPLES <= len(y):
            chunk = y[start: start + WINDOW_SAMPLES]
            S = librosa.feature.melspectrogram(
                y=chunk, sr=SAMPLE_RATE, n_mels=N_MELS, fmax=FMAX
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            windows.append(S_dB.astype(np.float32))
            start += HOP_SAMPLES

        if not windows:
            raise ValueError("Audio too short: no full 1s window could be extracted.")

        X = np.stack(windows)
        X = (X - self.train_min) / (self.train_max - self.train_min) # same norm as training
        return X[..., np.newaxis] # add channel dim -> (n, 128, 32, 1)

    # Public API
    """Score a single WAV file. Returns score + per-window detail"""
    def score(self, wav_path: Path) -> dict:
        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        X = self._extract_windows(y)

        X_pred = self.model.predict(X, verbose=0)
        err_per_band = np.mean(np.square(X - X_pred), axis=(2, 3)) # (n, 128)

        # weighted score per window, then aggregate (mean - found by investigation)
        weighted = err_per_band @ self.band_weights # (n,)
        file_score = float(weighted.mean())

        return {
            "anomaly_score": file_score,
            "n_windows": int(len(weighted)),
            "window_scores": weighted.tolist(),
        }