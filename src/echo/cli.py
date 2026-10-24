"""Command line entry points: transcribe, analyse, synthesise, build data, benchmark, serve.

Why: every capability should be reachable without the HTTP layer so it can be scripted and
tested, and so the benchmark and the API provably run the same pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from echo import __version__
from echo.asr.registry import resolve_transcriber
from echo.audio.io import Audio, read_wav, write_wav
from echo.audio.synth import list_voices, synthesize
from echo.corpus.build import build_dataset
from echo.pipeline import analysis_to_dict, analyze
from echo.products.transcript import to_srt, to_text, to_vtt


def build_parser() -> argparse.ArgumentParser:
    """All subcommands."""
    parser = argparse.ArgumentParser(prog="echo", description="Speech intelligence for lectures and meetings.")
    parser.add_argument("--version", action="version", version=f"echo {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    transcribe = sub.add_parser("transcribe", help="print a transcript for a WAV file")
    transcribe.add_argument("file", type=Path)
    transcribe.add_argument("--format", choices=("txt", "srt", "vtt", "json"), default="txt")
    _backend_args(transcribe)

    analyse = sub.add_parser("analyze", help="write every product for a WAV file into a directory")
    analyse.add_argument("file", type=Path)
    analyse.add_argument("--out", type=Path, required=True)
    _backend_args(analyse)

    synth = sub.add_parser("synth", help="synthesise speech with macOS say into a 16 kHz WAV")
    synth.add_argument("text")
    synth.add_argument("--voice", default="Samantha")
    synth.add_argument("--rate", type=int, default=175)
    synth.add_argument("-o", "--out", type=Path, required=True)

    sub.add_parser("voices", help="list English system voices")

    data = sub.add_parser("build-data", help="build the benchmark corpus")
    data.add_argument("--profile", choices=("quick", "full"), default="full")
    data.add_argument("--data-dir", type=Path, default=Path("data"))

    bench = sub.add_parser("bench", help="run the benchmark (same flags as python -m echo.bench)")
    bench.add_argument("args", nargs=argparse.REMAINDER)

    serve = sub.add_parser("serve", help="start the API and dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("auto", "whisper", "mock"), default=None)
    parser.add_argument("--model", default=None, help="whisper model size")


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Transcribe one file to stdout."""
    analysis = analyze(read_wav(args.file), resolve_transcriber(args.backend, args.model))
    renderers = {"txt": to_text, "srt": to_srt, "vtt": to_vtt}
    if args.format == "json":
        print(json.dumps(analysis_to_dict(analysis)["transcript"], indent=1))
    else:
        print(renderers[args.format](analysis.transcript))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Write transcript formats, notes, actions, keywords and the full analysis JSON."""
    analysis = analyze(read_wav(args.file), resolve_transcriber(args.backend, args.model))
    args.out.mkdir(parents=True, exist_ok=True)
    payload = analysis_to_dict(analysis)
    (args.out / "analysis.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    (args.out / "transcript.txt").write_text(to_text(analysis.transcript), encoding="utf-8")
    (args.out / "transcript.srt").write_text(to_srt(analysis.transcript), encoding="utf-8")
    (args.out / "transcript.vtt").write_text(to_vtt(analysis.transcript), encoding="utf-8")
    (args.out / "notes.md").write_text(analysis.notes_markdown(args.file.stem), encoding="utf-8")
    (args.out / "actions.json").write_text(json.dumps(payload["actions"], indent=1), encoding="utf-8")
    (args.out / "keywords.json").write_text(json.dumps(payload["keywords"], indent=1), encoding="utf-8")
    summary = f"{analysis.duration:.1f} s of audio, RTF {analysis.real_time_factor:.3f}, backend {analysis.asr.backend}"
    print(f"wrote {args.out} ({summary})")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    """Synthesise one utterance."""
    audio: Audio = synthesize(args.text, args.voice, args.rate)
    write_wav(args.out, audio)
    print(f"wrote {args.out} ({audio.duration:.2f} s)")
    return 0


def cmd_voices(_: argparse.Namespace) -> int:
    """List voices."""
    for voice in list_voices():
        print(f"{voice.name:28s} {voice.locale}")
    return 0


def cmd_build_data(args: argparse.Namespace) -> int:
    """Build the corpus."""
    manifests = build_dataset(args.profile, args.data_dir)
    print(f"built {len(manifests)} recordings, {sum(m.duration_s for m in manifests) / 60:.1f} min of audio")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Delegate to the benchmark module."""
    from echo.bench import main as bench_main

    return bench_main(args.args)


def cmd_serve(args: argparse.Namespace) -> int:
    """Run uvicorn."""
    import uvicorn

    uvicorn.run("echo.api.app:app", host=args.host, port=args.port, log_level="warning")
    return 0


COMMANDS = {
    "transcribe": cmd_transcribe,
    "analyze": cmd_analyze,
    "synth": cmd_synth,
    "voices": cmd_voices,
    "build-data": cmd_build_data,
    "bench": cmd_bench,
    "serve": cmd_serve,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
