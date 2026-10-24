# echo

Speech intelligence for lectures and meetings, running entirely on a laptop CPU. Give it a
recording and it returns a timestamped transcript with speaker turns, a keyword timeline,
extractive notes, action items with owners and deadlines, and a search index that jumps to the
moment a question was answered. Every quality number below was measured by a command in this
repository on a synthetic corpus with certain ground truth.

## What and why

Recorded lectures and meetings are long, linear and hard to come back to. Students with hearing or
attention difficulties need captions and notes they can trust; teams need the commitments that were
made, not the whole hour. Cloud transcription services solve this with an upload and a bill; echo
does it offline, deterministically, with a measured error budget.

The hard part of building this honestly is evaluation: real meeting corpora are licensed and small.
echo sidesteps that with macOS `say`: scripted lectures and meetings are synthesised in several
voices and speaking rates, degraded with noise and reverberation of known level, and every label
(text, speech spans, speakers, action items, key terms, queries) is exact by construction. The
benchmark therefore reports WER per voice and per SNR, speaker-turn accuracy, action-item F1 and
search rank, not a single headline number.

## Architecture

```
WAV ──> audio.io ──> vad.energy ──┬──> asr (faster-whisper | mock) ──┐
        (16 kHz mono)             └──> diarization (pitch+MFCC, KL) ─┼──> transcript with speakers
                                                                      │
                     keywords (timeline) <── notes (TextRank) <── actions (cues) <── search (BM25)
                                                                      │
                                     cli.py  /  api.app + dashboard  /  bench.py (scores vs labels)
```

One function, `echo.pipeline.analyze`, runs every stage for the CLI, the API and the benchmark, so
the benchmark measures exactly what the API serves. `docs/ARCHITECTURE.md` describes each module
and why it is shaped the way it is.

## Quickstart

```bash
make setup                                     # uv sync, Python 3.13
make test                                      # 126 tests on the deterministic mock backend
make bench                                     # build the corpus with say, run whisper base, write benchmarks/results.{json,md}
make serve                                     # dashboard and API on http://127.0.0.1:8000
uv run echo analyze talk.wav --out out/        # transcript (txt, srt, vtt), notes.md, actions.json, keywords.json
```

`make bench` needs macOS (`say`, `afconvert`) and the faster-whisper `base` weights in the Hugging
Face cache (`ECHO_ALLOW_DOWNLOAD=1` lets it fetch them once, about 145 MB). Everything else runs on
Linux too; without cached weights the pipeline falls back to a deterministic mock and says so.

## Measured results

All numbers below are copied from `benchmarks/results.json`, written by `make bench` at 2026-10-24T23:56:49+00:00 (commit `a50fd4d`): faster-whisper `base` int8, 4 CPU threads, macOS-26.5.1-arm64-arm-64bit-Mach-O, Python 3.13.17, 110 recordings, 136.0 minutes of synthetic audio, 455 s wall time. WER and CER are pooled over reference length; VAD scores are frame level against the clean sentence spans; see `docs/EVALUATION.md` for every definition and `REPORT.md` for the command outputs.

Headline: 2.8% WER on clean synthetic lectures across five English voices, 7.4% at 10 dB white noise, 39.4% at 0 dB; speaker purity 0.989 and action-item F1 0.965 on three-speaker meetings; the full pipeline runs at 0.050 times real time (3.0 s per audio minute) on CPU.

### Per voice (clean lectures, 175 wpm)

| Voice | Recordings | WER | CER | Sub | Del | Ins | VAD F1 | ASR RTF |
|---|---|---|---|---|---|---|---|---|
| overall | 30 | 2.8% | 1.0% | 1.7% | 0.6% | 0.6% | 0.979 | 0.047 |
| Daniel | 6 | 3.1% | 1.0% | 1.8% | 0.5% | 0.8% | 0.979 | 0.049 |
| Karen | 6 | 3.5% | 1.3% | 2.0% | 0.8% | 0.8% | 0.979 | 0.048 |
| Moira | 6 | 2.9% | 1.0% | 1.6% | 0.6% | 0.7% | 0.980 | 0.044 |
| Rishi | 6 | 3.1% | 0.9% | 1.8% | 0.5% | 0.8% | 0.980 | 0.047 |
| Samantha | 6 | 1.5% | 0.5% | 1.0% | 0.5% | 0.0% | 0.977 | 0.047 |

### Per acoustic condition (two lectures, three voices)

| Condition | Recordings | WER | CER | VAD F1 | VAD P | VAD R | 1 speaker found |
|---|---|---|---|---|---|---|---|
| clean | 6 | 2.5% | 0.6% | 0.981 | 0.963 | 1.000 | 100.0% |
| white_20db | 6 | 4.0% | 1.4% | 0.988 | 0.983 | 0.994 | 100.0% |
| white_10db | 6 | 7.4% | 3.2% | 0.983 | 0.989 | 0.977 | 50.0% |
| white_5db | 6 | 14.2% | 7.0% | 0.975 | 0.991 | 0.960 | 50.0% |
| white_0db | 6 | 39.4% | 22.8% | 0.969 | 0.992 | 0.949 | 33.3% |
| pink_10db | 6 | 4.9% | 1.6% | 0.988 | 0.987 | 0.989 | 83.3% |
| pink_5db | 6 | 6.4% | 2.7% | 0.981 | 0.990 | 0.972 | 50.0% |
| babble_10db | 6 | 7.6% | 2.9% | 0.983 | 0.985 | 0.982 | 66.7% |
| babble_5db | 6 | 14.4% | 7.1% | 0.961 | 0.943 | 0.982 | 66.7% |
| reverb_rt60_0.4s | 6 | 4.3% | 1.5% | 0.978 | 0.958 | 1.000 | 66.7% |
| reverb_rt60_0.8s | 6 | 7.3% | 3.2% | 0.982 | 0.965 | 1.000 | 66.7% |

### Per speaking rate

| Rate | Recordings | WER | CER | VAD F1 |
|---|---|---|---|---|
| 175 wpm | 6 | 2.5% | 0.6% | 0.981 |
| 140 wpm | 6 | 2.7% | 0.8% | 0.983 |
| 220 wpm | 6 | 2.6% | 0.8% | 0.976 |

### Meetings (speaker turns, action items, search)

| Set | Recordings | WER | Speaker count | Change F1 | Purity | Actions P/R/F1 | Owner acc | Search hit@1 | Search MRR |
|---|---|---|---|---|---|---|---|---|---|
| overall | 8 | 4.1% | 75.0% | 0.994 | 0.989 | 0.95/0.98/0.96 | 88.2% | 75.0% | 0.833 |
| clean | 4 | 1.5% | 75.0% | 0.989 | 0.978 | 0.95/0.95/0.95 | 88.2% | 83.3% | 0.917 |
| white_10db | 4 | 6.6% | 75.0% | 1.000 | 1.000 | 0.95/1.00/0.98 | 88.2% | 66.7% | 0.750 |

### Lecture products (clean)

| Recordings | Keyword recall@10 | Key-sentence recall | Search hit@1 | Search MRR |
|---|---|---|---|---|
| 42 | 52.1% | 54.0% | 82.5% | 0.902 |

### Latency (all recordings)

| Stage | Seconds per audio minute | RTF |
|---|---|---|
| ASR | 2.86 | 0.048 |
| VAD | 0.014 |  |
| Speakers | 0.140 |  |
| Products | 0.001 |  |
| Full pipeline | 3.01 | 0.050 |
| Model load (once) | 0.20 s |  |

## API

| Method and path | Purpose |
|---|---|
| `GET /health` | Status, backend, model, whether an API key is required |
| `GET /metrics` | Prometheus text format: request counts and latency, jobs, audio seconds |
| `POST /v1/jobs` | Multipart `file` (PCM WAV, any sample rate) and optional `title`; returns the full analysis |
| `POST /v1/demo` | Analyse a synthetic three-speaker meeting (uses `say`, or the bundled clip) |
| `GET /v1/jobs`, `GET /v1/jobs/{id}` | List and fetch jobs |
| `GET /v1/jobs/{id}/audio` | WAV for the player |
| `GET /v1/jobs/{id}/transcript?format=txt|srt|vtt|json` | Transcript exports |
| `GET /v1/jobs/{id}/notes` | Notes as Markdown |
| `GET /v1/jobs/{id}/search?q=...&k=5` | BM25 hits with timestamps |

Every response carries an `X-Request-ID` header (propagated when the client sends one); each request
is logged as one JSON line. Set `APP_API_KEY` to require the `X-API-Key` header on `/v1/*`. Uploads
over `ECHO_MAX_UPLOAD_MB` (50) answer 413, non-PCM files 415, audio over `ECHO_MAX_AUDIO_S` (3600)
422. Schemas are typed with Pydantic and served at `/docs`.

The dashboard at `/` is plain HTML and JavaScript (no build step): upload or run the demo, then
click any segment, keyword marker, note, action item or search hit to seek the player; the current
segment and word are highlighted while the audio plays.

## Configuration

See the table in `docs/ARCHITECTURE.md`. The important ones: `ECHO_ASR_BACKEND` (`auto`, `whisper`,
`mock`), `ECHO_WHISPER_MODEL` (`base` by default), `ECHO_CPU_THREADS` (4), `APP_API_KEY`.

## Docker

```bash
make docker && docker run --rm -p 8000:8000 -e ECHO_ALLOW_DOWNLOAD=1 echo:latest
```

Multi-stage image, non-root user, health check. Mount `~/.cache/huggingface` to
`/home/echo/.cache/huggingface` to reuse downloaded weights; without weights and without download
permission the service runs the mock backend and reports it in `/health`.

## Limitations

All of these are measured, not guessed:

- **Noise floor of the recogniser.** WER climbs from 2.5% clean to 14.2% at 5 dB and 39.4% at 0 dB white noise (single recordings reach 57%); babble at 5 dB costs 14.4%. Insertions grow fastest, which is whisper filling noise with words.
- **Speaker counting breaks in noise for single speakers.** On the 60 noise-axis lectures the clustering returned more than one speaker in 22 cases (speaker count accuracy 50% at 10 dB white noise, 33% at 0 dB) while purity stayed at 1.000, so turns are never mixed but a noisy lecture can be shown as two voices. Meetings: 6 of 8 recordings got the exact count (change-point F1 0.994, purity 0.989).
- **Keyword extraction without a background corpus** covers 52.1% of the expert key terms in its top 10, and the notes keep 54.0% of the sentences the scripts mark essential. The keyword prior constants were tuned on the same six scripts (`docs/EVALUATION.md`), so that number is in-sample on the text side.
- **Synthetic speech is easier than people.** `say` voices are consistent in pitch and timbre and never overlap, hesitate or trail off. Every number here is an upper bound for human recordings, and there is no human-recorded validation set yet.
- **The speaker threshold was calibrated on half of this corpus** (count accuracy on the held-out half is in `REPORT.md`); it is a setting for these voices, not a universal constant.
- **Owners need correct names.** Owner accuracy is 88.2%; the misses are names the recogniser altered ("Mei" heard as "May").
- **Search** answers 82.5% of lecture queries at rank 1 (MRR 0.902) and 75.0% of meeting queries (MRR 0.833); the misses are queries whose words occur in a neighbouring sentence.
- **English only, whole-file processing, synchronous API.** No streaming, no language detection, and the corpus builder needs macOS for `say`.
- **The mock backend is plumbing, not recognition**; its rows are flagged in `results.json` and never appear in the tables above.

## Roadmap

- Replace the one-transcript keyword prior with an IDF table from a public-domain corpus, which the
  in-sample sweep suggests is the largest single quality lever left.
- Noise-robust speaker features (a denoising front end or learned embeddings) so noisy single-speaker
  lectures stop being split; today purity stays high but the speaker count does not.
- Streaming: the VAD already produces regions incrementally; whisper would need windowed decoding.
- Optional LLM polish of the extractive notes behind a flag, never a dependency for tests or benchmarks.
- A human-recorded validation set to check that the synthetic numbers transfer.

## Development

```bash
make lint          # ruff check and format check
make test          # pytest
make bench-quick   # 7-recording profile used as the CI gate
uv run python -m echo.calibrate   # speaker threshold sweep, train/test split by script
```

Licensed under MIT.
