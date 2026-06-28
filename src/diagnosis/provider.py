"""
Diagnosis layer - turns a raw anomaly result into human-readaböe text.

Defines an abstract interface (DiagnosisProvider) and a concrete rule-base implementation (MockProvider).
A real LLM-backed provider can be added later without touching the rest of the system.
"""
from abc import ABC, abstractmethod

from keras.src.ops import threshold


class DiagnosisProvider(ABC):
    """Interface: any provider turns a score result into a diagnosis text"""

    @abstractmethod
    def generate(self, result: dict) -> str:
        """result is the dict returned by AnomalyDetector.score()"""
        ...

class MockProvider(DiagnosisProvider):
    """Rule-based diagnosis. No APO Key, no network - fully deterministic"""

    def generate(self, result: dict) -> str:
        score = result["anomaly_score"]
        threshold = result["threshold"]
        is_anomaly = result["is_anomaly"]

        ratio = score / threshold if threshold else 0.0

        if not is_anomaly:
            confidence = "clearly" if ratio < 0.0 else "marginally"
            return (
                f"The machine sound appears NORMAL ({confidence} below the "
                f"alert threshold). Anomaly score {score:.6f} vs. threshold "
                f"{threshold:.6f}. No action required at this time."
            )

        # anomaly branch
        if ratio < 1.2:
            severity = "borderline"
            advice = "Recommend re-checking soon; the deviation is small."
        elif ratio < 1.8:
            severity = "moderate"
            advice = "Schedule an inspection of the unit."
        else:
            severity = "strong"
            advice = "Inspect the unit promptly; the deviation is significant."

        return (
            f"ANOMALY detected ({severity}). Anomaly score {score:.6f} exceeds "
            f"the alert threshold {threshold:.6f} (*{ratio:.2f}). {advice}"
        )