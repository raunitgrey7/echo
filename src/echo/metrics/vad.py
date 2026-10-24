"""Frame-level VAD scores against reference speech spans.

Why: WER alone cannot tell whether a miss came from the recogniser or from a segmenter that
dropped a sentence. Frame precision and recall on a 10 ms grid separate the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echo.vad.energy import SpeechRegion, regions_to_mask


@dataclass(frozen=True, slots=True)
class VadScore:
    """Precision, recall and F1 over frames."""

    precision: float
    recall: float
    f1: float
    speech_frames_ref: int
    speech_frames_hyp: int


def score_regions(reference: list[SpeechRegion], hypothesis: list[SpeechRegion], duration_s: float) -> VadScore:
    """Compare two region lists on a 10 ms frame grid."""
    ref = regions_to_mask(reference, duration_s)
    hyp = regions_to_mask(hypothesis, duration_s)
    true_pos = int(np.count_nonzero(ref & hyp))
    precision = true_pos / hyp.sum() if hyp.any() else 0.0
    recall = true_pos / ref.sum() if ref.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return VadScore(float(precision), float(recall), float(f1), int(ref.sum()), int(hyp.sum()))
