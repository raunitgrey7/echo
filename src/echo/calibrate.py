"""Calibrate the speaker clustering threshold on a train/test split of the corpus.

Why: the clustering has one free parameter, the KL distance below which two regions are the same
voice. Choosing it on the whole benchmark and then reporting benchmark accuracy would be circular,
so this command sweeps the threshold on the recordings of five scripts and reports the other five
separately. REPORT.md quotes this output; EVALUATION.md names the setting as corpus-specific.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from echo.audio.io import read_wav
from echo.bench import PROJECT_ROOT
from echo.corpus.build import RecordingManifest, load_dataset
from echo.diarization.change import SpeakerConfig, agglomerate, region_frames
from echo.metrics.diarization import LabelledSpan, purity
from echo.vad.energy import SpeechRegion, detect_speech

TRAIN_SCRIPTS = frozenset({"photosynthesis", "newton", "binary_search", "course_staff", "research_weekly"})
THRESHOLDS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)


def prepare(
    manifests: list[RecordingManifest], data_dir: Path
) -> list[tuple[RecordingManifest, list[SpeechRegion], list[np.ndarray]]]:
    """VAD regions and per-region frame features for every recording, computed once."""
    prepared = []
    for manifest in manifests:
        audio = read_wav(data_dir / manifest.path)
        regions = detect_speech(audio)
        prepared.append((manifest, regions, region_frames(audio, regions)))
    return prepared


def evaluate(
    items: Sequence[tuple[RecordingManifest, list[SpeechRegion], list[np.ndarray]]], threshold: float
) -> tuple[float, float]:
    """Speaker-count accuracy and mean purity at one threshold."""
    config = SpeakerConfig(distance_threshold=threshold)
    correct: list[float] = []
    purities: list[float] = []
    for manifest, regions, blocks in items:
        labels, _ = agglomerate(blocks, config)
        reference = [LabelledSpan(SpeechRegion(u.start, u.end), u.speaker) for u in manifest.utterances]
        hypothesis = [LabelledSpan(r, f"s{label}") for r, label in zip(regions, labels, strict=True)]
        correct.append(float(len(set(labels.tolist())) == manifest.n_speakers))
        purities.append(purity(reference, hypothesis))
    return float(np.mean(correct)), float(np.mean(purities))


def main(argv: Sequence[str] | None = None) -> int:
    """Print the sweep as a Markdown table."""
    parser = argparse.ArgumentParser(prog="python -m echo.calibrate")
    parser.add_argument("--profile", default="full")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args(argv)
    prepared = prepare(load_dataset(args.profile, args.data_dir), args.data_dir)
    train = [p for p in prepared if p[0].script_id in TRAIN_SCRIPTS]
    test = [p for p in prepared if p[0].script_id not in TRAIN_SCRIPTS]
    print(f"train: {len(train)} recordings, test: {len(test)} recordings (split by script id)\n")
    print("| threshold | train count acc | train purity | test count acc | test purity |")
    print("|---|---|---|---|---|")
    for threshold in THRESHOLDS:
        acc_train, pur_train = evaluate(train, threshold)
        acc_test, pur_test = evaluate(test, threshold)
        print(f"| {threshold:.1f} | {acc_train:.3f} | {pur_train:.3f} | {acc_test:.3f} | {pur_test:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
