"""Benchmark harness: build the data, run the pipeline, score every product, write the results.

Why: every number in the README must come from one command with a fixed seed. The harness builds
(or reuses) the synthetic corpus, runs exactly the pipeline the API serves on every recording,
scores each product against the manifest labels and writes ``benchmarks/results.json`` plus a
Markdown table. ``--check`` turns the same run into a quality gate by comparing dotted metric
paths against ``benchmarks/thresholds.json``. Rows produced with the mock backend are labelled as
such in the metadata and are never a measurement of recognition.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from echo import __version__
from echo.asr.base import Transcriber
from echo.asr.mock import MockTranscriber, audio_digest
from echo.asr.registry import env_threads, resolve_transcriber
from echo.asr.whisper import WhisperTranscriber
from echo.audio.io import read_wav
from echo.corpus.build import NOISE_VOICES, RecordingManifest, build_dataset, load_dataset, oracle_for
from echo.corpus.scripts import lecture_by_id
from echo.metrics.diarization import LabelledSpan, purity, score_changes
from echo.metrics.products import key_sentence_recall, score_actions, score_query
from echo.metrics.vad import score_regions
from echo.metrics.wer import char_counts, word_counts
from echo.pipeline import Analysis, analyze
from echo.products.keywords import keyword_hits
from echo.vad.energy import SpeechRegion

CHANGE_TOLERANCE_S = 0.75
NOISE_SCRIPTS = ("photosynthesis", "newton")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def score_recording(manifest: RecordingManifest, analysis: Analysis) -> dict[str, Any]:
    """Flat dict of every metric for one recording."""
    row: dict[str, Any] = {
        "id": manifest.id,
        "kind": manifest.kind,
        "axis": manifest.axis,
        "script_id": manifest.script_id,
        "voices": list(manifest.voices),
        "rate_wpm": manifest.rate_wpm,
        "condition": manifest.condition["name"],
        "duration_s": manifest.duration_s,
    }
    row.update(_asr_scores(manifest, analysis))
    row.update(_vad_scores(manifest, analysis))
    row.update(_speaker_scores(manifest, analysis))
    row.update(_product_scores(manifest, analysis))
    row.update(_timing_scores(analysis))
    return row


def _asr_scores(manifest: RecordingManifest, analysis: Analysis) -> dict[str, Any]:
    words = word_counts(manifest.reference_text, analysis.transcript.text)
    chars = char_counts(manifest.reference_text, analysis.transcript.text)
    return {
        "ref_words": words.reference_length,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "wer": words.rate,
        "ref_chars": chars.reference_length,
        "char_errors": chars.errors,
        "cer": chars.rate,
    }


def _vad_scores(manifest: RecordingManifest, analysis: Analysis) -> dict[str, Any]:
    reference = [SpeechRegion(u.start, u.end) for u in manifest.utterances]
    score = score_regions(reference, list(analysis.regions), manifest.duration_s)
    return {"vad_precision": score.precision, "vad_recall": score.recall, "vad_f1": score.f1}


def _speaker_scores(manifest: RecordingManifest, analysis: Analysis) -> dict[str, Any]:
    changes = score_changes(manifest.speaker_changes, analysis.speakers.changes, CHANGE_TOLERANCE_S)
    reference = [LabelledSpan(SpeechRegion(u.start, u.end), u.speaker) for u in manifest.utterances]
    hypothesis = [
        LabelledSpan(region, f"s{label}")
        for region, label in zip(analysis.regions, analysis.speakers.labels, strict=True)
    ]
    hits = round(changes.recall * changes.n_reference)
    return {
        "n_speakers_ref": manifest.n_speakers,
        "n_speakers_hyp": analysis.speakers.n_speakers,
        "speaker_count_correct": analysis.speakers.n_speakers == manifest.n_speakers,
        "change_hits": hits,
        "change_n_ref": changes.n_reference,
        "change_n_hyp": changes.n_hypothesis,
        "change_f1": changes.f1,
        "purity": purity(reference, hypothesis),
    }


def _product_scores(manifest: RecordingManifest, analysis: Analysis) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if manifest.kind == "lecture":
        lecture = lecture_by_id(manifest.script_id)
        gold_sentences = [lecture.sentences[i] for i in manifest.key_sentences]
        recall = key_sentence_recall(analysis.notes.summary, gold_sentences)
        out.update(
            keyword_hits=keyword_hits(list(analysis.keywords), manifest.key_terms),
            keyword_total=len(manifest.key_terms),
            key_sentence_hits=recall.hits,
            key_sentence_total=recall.total,
        )
    else:
        confusion, owners = score_actions(analysis.actions, manifest.utterances)
        out.update(
            action_tp=confusion.tp,
            action_fp=confusion.fp,
            action_fn=confusion.fn,
            owner_hits=owners.hits,
            owner_total=owners.total,
        )
    index = analysis.search_index()
    ranks = [
        score_query(index.search(str(q["text"]), 5), manifest.utterances[int(q["answer_index"])])
        for q in manifest.queries
    ]
    out["search_ranks"] = ranks
    return out


def _timing_scores(analysis: Analysis) -> dict[str, Any]:
    timings = analysis.timings_s
    products = timings.get("keywords", 0.0) + timings.get("notes", 0.0) + timings.get("actions", 0.0)
    return {
        "asr_s": analysis.asr.elapsed_s,
        "vad_s": timings.get("vad", 0.0),
        "speakers_s": timings.get("speakers", 0.0),
        "products_s": products,
        "total_s": analysis.total_elapsed_s,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pool(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a group of recordings; error rates are pooled over reference length, not averaged."""
    if not rows:
        return {"n_recordings": 0}
    audio_s = sum(r["duration_s"] for r in rows)
    ref_words = sum(r["ref_words"] for r in rows)
    ref_chars = sum(r["ref_chars"] for r in rows)
    out: dict[str, Any] = {
        "n_recordings": len(rows),
        "audio_s": audio_s,
        "wer": _safe_div(sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in rows), ref_words),
        "substitution_rate": _safe_div(sum(r["substitutions"] for r in rows), ref_words),
        "deletion_rate": _safe_div(sum(r["deletions"] for r in rows), ref_words),
        "insertion_rate": _safe_div(sum(r["insertions"] for r in rows), ref_words),
        "cer": _safe_div(sum(r["char_errors"] for r in rows), ref_chars),
        "vad_precision": _mean([r["vad_precision"] for r in rows]),
        "vad_recall": _mean([r["vad_recall"] for r in rows]),
        "vad_f1": _mean([r["vad_f1"] for r in rows]),
        "speaker_count_accuracy": _mean([float(r["speaker_count_correct"]) for r in rows]),
        "purity": _mean([r["purity"] for r in rows]),
    }
    out.update(_pool_changes(rows))
    out.update(_pool_products(rows))
    out.update(_pool_latency(rows, audio_s))
    return out


def _pool_changes(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    hits = sum(r["change_hits"] for r in rows)
    n_ref = sum(r["change_n_ref"] for r in rows)
    n_hyp = sum(r["change_n_hyp"] for r in rows)
    precision = _safe_div(hits, n_hyp) if n_hyp else (1.0 if n_ref == 0 else 0.0)
    recall = _safe_div(hits, n_ref) if n_ref else (1.0 if n_hyp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"change_precision": precision, "change_recall": recall, "change_f1": f1, "false_changes": n_hyp - hits}


def _pool_products(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    lectures = [r for r in rows if r["kind"] == "lecture"]
    meetings = [r for r in rows if r["kind"] == "meeting"]
    if lectures:
        out["keyword_recall_at_10"] = _safe_div(
            sum(r["keyword_hits"] for r in lectures), sum(r["keyword_total"] for r in lectures)
        )
        out["key_sentence_recall"] = _safe_div(
            sum(r["key_sentence_hits"] for r in lectures), sum(r["key_sentence_total"] for r in lectures)
        )
    if meetings:
        tp = sum(r["action_tp"] for r in meetings)
        fp = sum(r["action_fp"] for r in meetings)
        fn = sum(r["action_fn"] for r in meetings)
        precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
        out["action_precision"] = precision
        out["action_recall"] = recall
        out["action_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out["owner_accuracy"] = _safe_div(
            sum(r["owner_hits"] for r in meetings), sum(r["owner_total"] for r in meetings)
        )
    ranks = [rank for r in rows for rank in r["search_ranks"]]
    if ranks:
        out["search_hit_at_1"] = _safe_div(sum(1 for k in ranks if k == 1), len(ranks))
        out["search_hit_at_3"] = _safe_div(sum(1 for k in ranks if 1 <= k <= 3), len(ranks))
        out["search_mrr"] = _safe_div(sum(1.0 / k for k in ranks if k > 0), len(ranks))
        out["search_queries"] = len(ranks)
    return out


def _pool_latency(rows: Sequence[dict[str, Any]], audio_s: float) -> dict[str, float]:
    minutes = audio_s / 60.0
    return {
        "asr_s_per_audio_min": _safe_div(sum(r["asr_s"] for r in rows), minutes),
        "vad_s_per_audio_min": _safe_div(sum(r["vad_s"] for r in rows), minutes),
        "speakers_s_per_audio_min": _safe_div(sum(r["speakers_s"] for r in rows), minutes),
        "products_s_per_audio_min": _safe_div(sum(r["products_s"] for r in rows), minutes),
        "pipeline_s_per_audio_min": _safe_div(sum(r["total_s"] for r in rows), minutes),
        "asr_rtf": _safe_div(sum(r["asr_s"] for r in rows), audio_s),
        "pipeline_rtf": _safe_div(sum(r["total_s"] for r in rows), audio_s),
    }


def group_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The report's groupings: voice, condition (with a matched clean baseline), rate, meetings."""
    voice_axis = [r for r in rows if r["axis"] == "voice"]
    baseline = [r for r in voice_axis if r["script_id"] in NOISE_SCRIPTS and r["voices"][0] in NOISE_VOICES]
    by_condition = {"clean": pool(baseline)}
    for name in sorted({r["condition"] for r in rows if r["axis"] == "noise"}, key=_condition_order):
        by_condition[name] = pool([r for r in rows if r["axis"] == "noise" and r["condition"] == name])
    by_rate = {"175": pool(baseline)}
    for rate in sorted({r["rate_wpm"] for r in rows if r["axis"] == "rate"}):
        by_rate[str(rate)] = pool([r for r in rows if r["axis"] == "rate" and r["rate_wpm"] == rate])
    meetings = [r for r in rows if r["axis"] == "meeting"]
    return {
        "overall": pool(rows),
        "voice_axis": {
            "overall": pool(voice_axis),
            **{
                voice: pool([r for r in voice_axis if r["voices"][0] == voice])
                for voice in sorted({r["voices"][0] for r in voice_axis})
            },
        },
        "condition_axis": by_condition,
        "rate_axis": by_rate,
        "meetings": {
            "overall": pool(meetings),
            **{
                name: pool([r for r in meetings if r["condition"] == name])
                for name in sorted({r["condition"] for r in meetings})
            },
        },
        "lectures_clean": pool([r for r in rows if r["kind"] == "lecture" and r["condition"] == "clean"]),
    }


def _condition_order(name: str) -> tuple[int, float]:
    family = {"white": 0, "pink": 1, "babble": 2, "reverb": 3}.get(name.split("_")[0], 9)
    digits = "".join(ch for ch in name.split("_", 1)[-1] if ch.isdigit() or ch == ".")
    value = float(digits) if digits else 0.0
    return (family, -value if family < 3 else value)


def git_commit() -> str | None:
    """Short commit hash when run inside the repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def make_transcriber(backend: str, model_size: str, manifests: list[RecordingManifest], data_dir: Path) -> Transcriber:
    """Resolve the backend; the mock gets the manifest oracle so its output is labelled and deterministic."""
    table = oracle_for(manifests, data_dir) if backend in ("mock", "auto") else {}
    return resolve_transcriber(
        backend, model_size, oracle=lambda audio: table.get(audio_digest(audio)), mock_error_rate=0.0
    )


def run(
    profile: str, backend: str, model_size: str, data_dir: Path, out_dir: Path, limit: int | None = None
) -> dict[str, Any]:
    """Build data if needed, analyse every recording and write results.json and results.md."""
    started = time.perf_counter()
    manifests = (
        build_dataset(profile, data_dir)
        if not (data_dir / f"manifest_{profile}.json").exists()
        else load_dataset(profile, data_dir)
    )
    if limit:
        manifests = manifests[:limit]
    transcriber = make_transcriber(backend, model_size, manifests, data_dir)
    load_s = _warm_up(transcriber)
    rows: list[dict[str, Any]] = []
    for index, manifest in enumerate(manifests, start=1):
        analysis = analyze(read_wav(data_dir / manifest.path), transcriber)
        rows.append(score_recording(manifest, analysis))
        rtf = rows[-1]["total_s"] / manifest.duration_s
        print(f"[{index:3d}/{len(manifests)}] {manifest.id:48s} WER={rows[-1]['wer']:.3f} RTF={rtf:.3f}", flush=True)
    results = {
        "meta": _meta(profile, transcriber, manifests, load_s, time.perf_counter() - started),
        "groups": group_rows(rows),
        "recordings": rows,
    }
    write_results(results, out_dir)
    return results


def _warm_up(transcriber: Transcriber) -> float:
    started = time.perf_counter()
    if isinstance(transcriber, WhisperTranscriber):
        transcriber.warm_up()
    return time.perf_counter() - started


def _meta(
    profile: str, transcriber: Transcriber, manifests: list[RecordingManifest], load_s: float, wall_s: float
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "echo_version": __version__,
        "git_commit": git_commit(),
        "profile": profile,
        "backend": transcriber.name,
        "model": transcriber.model,
        "is_mock": isinstance(transcriber, MockTranscriber),
        "cpu_threads": env_threads(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "n_recordings": len(manifests),
        "total_audio_s": sum(m.duration_s for m in manifests),
        "model_load_s": load_s,
        "wall_s": wall_s,
        "seeds": "per-recording seed = first 4 bytes of sha256(recording id); mock seed 0",
    }


def write_results(results: dict[str, Any], out_dir: Path) -> None:
    """results.json and results.md side by side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (out_dir / "results.md").write_text(render_markdown(results), encoding="utf-8")


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(results: dict[str, Any]) -> str:
    """Human-readable tables for the README and REPORT."""
    meta, groups = results["meta"], results["groups"]
    parts = [
        f"# echo benchmark ({meta['profile']} profile)",
        "",
        f"Backend: `{meta['backend']}` model `{meta['model']}`, {meta['n_recordings']} recordings, "
        f"{meta['total_audio_s'] / 60:.1f} min of audio, {meta['cpu_threads']} CPU threads, {meta['platform']}. "
        f"Run at {meta['timestamp_utc']}" + (f", commit `{meta['git_commit']}`." if meta.get("git_commit") else "."),
        "",
    ]
    if meta["is_mock"]:
        parts += [
            "**Mock backend: the ASR rows reproduce the reference text and are not a recognition measurement.**",
            "",
        ]
    parts += ["## Per voice (clean lectures)", "", _voice_table(groups["voice_axis"]), ""]
    parts += [
        "## Per acoustic condition (two lectures, three voices)",
        "",
        _condition_table(groups["condition_axis"]),
        "",
    ]
    parts += ["## Per speaking rate", "", _rate_table(groups["rate_axis"]), ""]
    parts += ["## Meetings (speaker turns, action items, search)", "", _meeting_table(groups["meetings"]), ""]
    parts += ["## Lecture products (clean)", "", _lecture_products(groups["lectures_clean"]), ""]
    parts += ["## Latency (all recordings)", "", _latency_table(groups["overall"], meta), ""]
    return "\n".join(parts)


def _voice_table(group: dict[str, Any]) -> str:
    rows = [
        [
            name,
            str(g["n_recordings"]),
            _pct(g["wer"]),
            _pct(g["cer"]),
            _pct(g["substitution_rate"]),
            _pct(g["deletion_rate"]),
            _pct(g["insertion_rate"]),
            f"{g['vad_f1']:.3f}",
            f"{g['asr_rtf']:.3f}",
        ]
        for name, g in group.items()
        if g.get("n_recordings")
    ]
    return _table(["Voice", "Recordings", "WER", "CER", "Sub", "Del", "Ins", "VAD F1", "ASR RTF"], rows)


def _condition_table(group: dict[str, Any]) -> str:
    rows = [
        [
            name,
            str(g["n_recordings"]),
            _pct(g["wer"]),
            _pct(g["cer"]),
            f"{g['vad_f1']:.3f}",
            f"{g['vad_precision']:.3f}",
            f"{g['vad_recall']:.3f}",
            _pct(g["speaker_count_accuracy"]),
        ]
        for name, g in group.items()
        if g.get("n_recordings")
    ]
    return _table(["Condition", "Recordings", "WER", "CER", "VAD F1", "VAD P", "VAD R", "1 speaker found"], rows)


def _rate_table(group: dict[str, Any]) -> str:
    rows = [
        [f"{name} wpm", str(g["n_recordings"]), _pct(g["wer"]), _pct(g["cer"]), f"{g['vad_f1']:.3f}"]
        for name, g in group.items()
        if g.get("n_recordings")
    ]
    return _table(["Rate", "Recordings", "WER", "CER", "VAD F1"], rows)


def _meeting_table(group: dict[str, Any]) -> str:
    rows = [
        [
            name,
            str(g["n_recordings"]),
            _pct(g["wer"]),
            _pct(g["speaker_count_accuracy"]),
            f"{g['change_f1']:.3f}",
            f"{g['purity']:.3f}",
            f"{g.get('action_precision', 0):.2f}/{g.get('action_recall', 0):.2f}/{g.get('action_f1', 0):.2f}",
            _pct(g.get("owner_accuracy", 0)),
            _pct(g.get("search_hit_at_1", 0)),
            f"{g.get('search_mrr', 0):.3f}",
        ]
        for name, g in group.items()
        if g.get("n_recordings")
    ]
    return _table(
        [
            "Set",
            "Recordings",
            "WER",
            "Speaker count",
            "Change F1",
            "Purity",
            "Actions P/R/F1",
            "Owner acc",
            "Search hit@1",
            "Search MRR",
        ],
        rows,
    )


def _lecture_products(group: dict[str, Any]) -> str:
    if not group.get("n_recordings"):
        return "(no clean lectures in this profile)"
    rows = [
        [
            str(group["n_recordings"]),
            _pct(group["keyword_recall_at_10"]),
            _pct(group["key_sentence_recall"]),
            _pct(group["search_hit_at_1"]),
            f"{group['search_mrr']:.3f}",
        ]
    ]
    return _table(["Recordings", "Keyword recall@10", "Key-sentence recall", "Search hit@1", "Search MRR"], rows)


def _latency_table(group: dict[str, Any], meta: dict[str, Any]) -> str:
    rows = [
        ["ASR", f"{group['asr_s_per_audio_min']:.2f}", f"{group['asr_rtf']:.3f}"],
        ["VAD", f"{group['vad_s_per_audio_min']:.3f}", ""],
        ["Speakers", f"{group['speakers_s_per_audio_min']:.3f}", ""],
        ["Products", f"{group['products_s_per_audio_min']:.3f}", ""],
        ["Full pipeline", f"{group['pipeline_s_per_audio_min']:.2f}", f"{group['pipeline_rtf']:.3f}"],
        ["Model load (once)", f"{meta['model_load_s']:.2f} s", ""],
    ]
    return _table(["Stage", "Seconds per audio minute", "RTF"], rows)


def lookup(results: dict[str, Any], dotted: str) -> Any:
    """Resolve ``groups.voice_axis.overall.wer`` style paths."""
    node: Any = results
    for key in dotted.split("."):
        node = node[key]
    return node


def check_thresholds(results: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """Return one message per violated threshold; an empty list means the gate passes."""
    section = thresholds.get(results["meta"]["backend"], {})
    failures: list[str] = []
    for path, rule in section.items():
        try:
            value = lookup(results, path)
        except KeyError:
            failures.append(f"{path}: missing from results")
            continue
        if "max" in rule and value > rule["max"]:
            failures.append(f"{path}: {value:.4f} exceeds max {rule['max']}")
        if "min" in rule and value < rule["min"]:
            failures.append(f"{path}: {value:.4f} below min {rule['min']}")
    return failures


def build_parser() -> argparse.ArgumentParser:
    """CLI options."""
    parser = argparse.ArgumentParser(prog="python -m echo.bench", description="Run the echo benchmark.")
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--backend", choices=("auto", "whisper", "mock"), default="auto")
    parser.add_argument("--model", default=None, help="whisper model size (default from ECHO_WHISPER_MODEL or base)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="only the first N recordings (development)")
    parser.add_argument("--check", type=Path, default=None, help="thresholds JSON; exit 1 when a threshold fails")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    model_size = args.model or "base"
    results = run(args.profile, args.backend, model_size, args.data_dir, args.out, args.limit)
    print(render_markdown(results))
    if args.check:
        failures = check_thresholds(results, json.loads(args.check.read_text(encoding="utf-8")))
        for failure in failures:
            print(f"THRESHOLD FAILED: {failure}", file=sys.stderr)
        print("quality gate:", "FAILED" if failures else "passed")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
