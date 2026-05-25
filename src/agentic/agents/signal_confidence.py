"""
Signal Confidence — confidence calibration and tracking.

Tracks prediction confidence vs actual outcomes for calibration.
Groups predictions into confidence bins and tracks actual accuracy
per bin. Used to detect over/under-confidence.
"""

from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict


class ConfidenceCalibrator:
    """Tracks prediction confidence vs actual outcomes for calibration.

    Groups predictions into confidence bins and tracks actual accuracy
    per bin. Used to detect over/under-confidence.
    """

    def __init__(self):
        self._buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._calibration_bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def record_outcome(self, confidence: float, was_correct: bool):
        """Record whether a prediction at a given confidence was correct.

        Args:
            confidence: The confidence value (0.0 to 1.0).
            was_correct: Whether the prediction was correct.
        """
        if not isinstance(confidence, (int, float)):
            return
        for bin_edge in self._calibration_bins:
            if confidence <= bin_edge:
                self._buckets[f"bin_{bin_edge}"].append(1 if was_correct else 0)
                break

    def get_report(self) -> Dict:
        """Get calibration report.

        Returns a dict mapping bin names to their stats:
        expected confidence, actual accuracy, sample count, calibration error.
        """
        report = {}
        for bin_name, outcomes in self._buckets.items():
            if len(outcomes) >= 5:
                actual_accuracy = sum(outcomes) / len(outcomes)
                expected = float(bin_name.split("_")[1])
                report[bin_name] = {
                    "expected": expected,
                    "actual": round(actual_accuracy, 3),
                    "samples": len(outcomes),
                    "calibration_error": round(abs(actual_accuracy - expected), 3),
                }
        return report
