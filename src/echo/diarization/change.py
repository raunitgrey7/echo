"""Speaker change detection by agglomerative clustering of per-region voice distributions.

Why: a meeting transcript without speaker turns is hard to read, but a full diarization stack
needs trained embeddings. Here each VAD region is described by the frames it contains: pitch in
semitones (the dominant cue on this corpus, see pitch.py) plus MFCC 1 to 12, taken over voiced
frames so silence and fricatives do not dilute the voice. A region pair is compared by the
symmetric KL divergence between diagonal Gaussians fitted to their frames, which weighs a mean
difference by how variable each coefficient is within the region. Clusters are merged greedily,
closest first, with the Gaussian re-estimated from the pooled frames after every merge, until
the closest pair is further apart than a threshold. The number of speakers therefore falls out
of the data. The threshold was calibrated on half of the benchmark corpus and tested on the other
half; EVALUATION.md reports both numbers and names this as a corpus-specific setting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echo.audio.io import Audio
from echo.diarization.mfcc import mfcc
from echo.diarization.pitch import frame_f0
from echo.vad.energy import SpeechRegion


@dataclass(frozen=True, slots=True)
class SpeakerConfig:
    """Clustering parameters."""

    n_mfcc: int = 13
    pitch_weight: float = 1.0
    distance_threshold: float = 3.0
    min_region_s: float = 0.4
    min_voiced_frames: int = 20
    max_speakers: int = 8


DEFAULT_SPEAKER = SpeakerConfig()
HOP_S = 0.01


@dataclass(frozen=True, slots=True)
class SpeakerResult:
    """Labels per region (ints from 0) and the derived change points."""

    labels: tuple[int, ...]
    n_speakers: int
    changes: tuple[float, ...]
    closest_distance: float


def region_frames(
    audio: Audio, regions: list[SpeechRegion], config: SpeakerConfig = DEFAULT_SPEAKER
) -> list[np.ndarray]:
    """Per region, a (frames, 1 + n_mfcc - 1) matrix of semitone pitch and MFCC 1.. over voiced frames."""
    feats = mfcc(audio.samples, audio.sample_rate, n_mfcc=config.n_mfcc)
    f0 = frame_f0(audio.samples, audio.sample_rate)
    n = min(feats.shape[0], f0.shape[0])
    feats, f0 = feats[:n], f0[:n]
    semitones = 12.0 * np.log2(np.maximum(f0, 1.0)) * config.pitch_weight
    blocks: list[np.ndarray] = []
    for region in regions:
        lo = min(int(region.start / HOP_S), max(n - 1, 0))
        hi = max(lo + 1, min(int(region.end / HOP_S), n))
        voiced = f0[lo:hi] > 0
        pitch = semitones[lo:hi]
        if voiced.sum() < config.min_voiced_frames:
            pitch = np.full(hi - lo, float(np.median(semitones[f0 > 0])) if (f0 > 0).any() else 0.0)
            voiced = np.ones(hi - lo, dtype=bool)
        blocks.append(np.column_stack([pitch[voiced], feats[lo:hi, 1:][voiced]]))
    return blocks


def gaussian_stats(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean, variance (floored) and frame count."""
    return frames.mean(axis=0), frames.var(axis=0) + 1e-3, frames.shape[0]


def symmetric_kl(a: tuple[np.ndarray, np.ndarray, int], b: tuple[np.ndarray, np.ndarray, int]) -> float:
    """Symmetric KL divergence between two diagonal Gaussians, summed over dimensions."""
    mean_a, var_a, _ = a
    mean_b, var_b, _ = b
    diff = np.square(mean_a - mean_b)
    return float(0.5 * np.sum(var_a / var_b + var_b / var_a - 2.0 + diff * (1.0 / var_a + 1.0 / var_b)))


def agglomerate(blocks: list[np.ndarray], config: SpeakerConfig = DEFAULT_SPEAKER) -> tuple[np.ndarray, float]:
    """Merge the closest clusters while their divergence is below the threshold; returns labels and the final gap."""
    members: list[list[int]] = [[i] for i in range(len(blocks))]
    stats = [gaussian_stats(b) for b in blocks]
    closest = float("inf")
    while len(members) > 1:
        best, pair = float("inf"), (0, 1)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                distance = symmetric_kl(stats[i], stats[j])
                if distance < best:
                    best, pair = distance, (i, j)
        closest = best
        if best >= config.distance_threshold and len(members) <= config.max_speakers:
            break
        i, j = pair
        members[i].extend(members.pop(j))
        stats.pop(j)
        stats[i] = gaussian_stats(np.vstack([blocks[k] for k in members[i]]))
    labels = np.zeros(len(blocks), dtype=int)
    for cluster, indices in enumerate(members):
        labels[indices] = cluster
    return labels, closest


def relabel_by_first_appearance(labels: np.ndarray) -> tuple[int, ...]:
    """Rename clusters so the first region is speaker 0, the next new one speaker 1, and so on."""
    mapping: dict[int, int] = {}
    out: list[int] = []
    for label in labels.tolist():
        if label not in mapping:
            mapping[label] = len(mapping)
        out.append(mapping[label])
    return tuple(out)


def change_points(regions: list[SpeechRegion], labels: tuple[int, ...]) -> tuple[float, ...]:
    """Midpoint of each gap between consecutive regions whose labels differ."""
    return tuple(
        (regions[i - 1].end + regions[i].start) / 2.0 for i in range(1, len(regions)) if labels[i] != labels[i - 1]
    )


def label_speakers(audio: Audio, regions: list[SpeechRegion], config: SpeakerConfig = DEFAULT_SPEAKER) -> SpeakerResult:
    """Cluster the regions of one recording into speakers; short regions inherit their predecessor."""
    usable = [r for r in regions if r.duration >= config.min_region_s]
    if not usable:
        return SpeakerResult(tuple(0 for _ in regions), 1 if regions else 0, (), float("inf"))
    raw, closest = agglomerate(region_frames(audio, usable, config), config)
    by_region = {id(r): int(label) for r, label in zip(usable, raw.tolist(), strict=True)}
    full = np.array([by_region.get(id(r), -1) for r in regions])
    for i in range(full.shape[0]):
        if full[i] < 0:
            full[i] = full[i - 1] if i > 0 else 0
    labels = relabel_by_first_appearance(full)
    return SpeakerResult(labels, len(set(labels)), change_points(regions, labels), closest)
