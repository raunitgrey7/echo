# REPORT: what was built, every measured number, and what is still missing

Project: `echo`, speech intelligence for lectures and meetings. Built on an
Apple Silicon Mac (15 cores, 24 GB), Python 3.13.17 via uv, no GPU, no network at runtime.

## 1. What was built

| Area | Delivered |
|---|---|
| Synthesis harness | `echo.audio.synth`: macOS `say` per (voice, rate, text), `afconvert` to 16 kHz mono PCM, numpy AIFF decoder as fallback, content-addressed cache under `data/cache/`. |
| Degradations | `echo.audio.degrade`: white, pink and babble noise at a target SNR measured on speech-active frames; Schroeder reverberation parameterised by RT60; every draw seeded. |
| Corpus | 6 lecture scripts (10 sentences each) and 4 meeting scripts (11 to 13 turns, 3 speakers) with key terms, key sentences, action-item labels (owner, deadline) and queries. 110 recordings, 136.0 minutes, manifests with SHA-256 checksums. |
| ASR | `faster-whisper` adapter (CPU int8, beam 5, temperature 0, word timestamps, numpy input so no ffmpeg) plus a deterministic oracle-backed mock for tests and CI. |
| VAD | Band-limited energy detector with percentile floor, adaptive onset margin, hysteresis, merge and pad rules. |
| Speaker change detection | Per-region pitch (autocorrelation) and MFCC frames, diagonal Gaussians, symmetric KL distance, agglomerative merging with re-estimation, number of speakers from a calibrated threshold. |
| Products | Timestamped transcript (txt, SRT, VTT, JSON) with speaker labels; keyword timeline; extractive notes (TextRank, TextTiling sections); action items with owner and deadline; BM25 search over sentences. |
| Benchmark | `python -m echo.bench`: builds the corpus, runs the pipeline, scores every product, writes `benchmarks/results.{json,md}`, gates on `benchmarks/thresholds.json`. |
| API | FastAPI with Pydantic schemas, `/health`, `/metrics` (Prometheus), JSON logs, request ids, API key header when `APP_API_KEY` is set, upload and length validation. |
| Dashboard | Vanilla HTML/JS at `/`: upload or demo, player synced to transcript (segment and word highlighting), keyword and speaker timeline, notes, action items, search, SRT/VTT/notes downloads. |
| Packaging | `Makefile` (setup, lint, test, bench, serve), multi-stage non-root `Dockerfile`, GitHub Actions CI (lint and tests on Linux and macOS; benchmark gate on macOS). |
| Docs | `README.md`, `docs/ARCHITECTURE.md`, `docs/EVALUATION.md`, this report. |

## 2. Measured numbers

Command: `make bench` (equivalent to `uv run python -m echo.bench --profile full --backend auto --model base --check benchmarks/thresholds.json`).
Run: 2026-10-24T23:56:49+00:00, commit `a50fd4d`, faster-whisper `base-int8`, 4 CPU threads,
macOS-26.5.1-arm64-arm-64bit-Mach-O, Python 3.13.17, 110 recordings, 136.0 min of audio, wall time 454.7 s,
model load 0.20 s. Output file: `benchmarks/results.json`; tables: `benchmarks/results.md`.
The quality gate passed.

### 2.1 WER and CER per voice (clean lectures, 175 wpm, 30 recordings)

| Voice | Recordings | WER | CER | Sub | Del | Ins | VAD F1 | ASR RTF |
|---|---|---|---|---|---|---|---|---|
| overall | 30 | 2.8% | 1.0% | 1.7% | 0.6% | 0.6% | 0.979 | 0.042 |
| Daniel (en_GB) | 6 | 3.1% | 1.0% | 1.8% | 0.5% | 0.8% | 0.979 | 0.042 |
| Karen (en_AU) | 6 | 3.5% | 1.3% | 2.0% | 0.8% | 0.8% | 0.979 | 0.042 |
| Moira (en_IE) | 6 | 2.9% | 1.0% | 1.6% | 0.6% | 0.7% | 0.980 | 0.039 |
| Rishi (en_IN) | 6 | 3.1% | 0.9% | 1.8% | 0.5% | 0.8% | 0.980 | 0.045 |
| Samantha (en_US) | 6 | 1.5% | 0.5% | 1.0% | 0.5% | 0.0% | 0.977 | 0.041 |

### 2.2 Per acoustic condition (lectures photosynthesis and newton; voices Samantha, Daniel, Rishi; 6 recordings each)

| Condition | WER | CER | VAD F1 | VAD P | VAD R | Single speaker found |
|---|---|---|---|---|---|---|
| clean | 2.5% | 0.6% | 0.981 | 0.963 | 1.000 | 100.0% |
| white 20 dB | 4.0% | 1.4% | 0.988 | 0.983 | 0.994 | 100.0% |
| white 10 dB | 7.4% | 3.2% | 0.983 | 0.989 | 0.977 | 50.0% |
| white 5 dB | 14.2% | 7.0% | 0.975 | 0.991 | 0.960 | 50.0% |
| white 0 dB | 39.4% | 22.8% | 0.969 | 0.992 | 0.949 | 33.3% |
| pink 10 dB | 4.9% | 1.6% | 0.988 | 0.987 | 0.989 | 83.3% |
| pink 5 dB | 6.4% | 2.7% | 0.981 | 0.990 | 0.972 | 50.0% |
| babble 10 dB | 7.6% | 2.9% | 0.983 | 0.985 | 0.982 | 66.7% |
| babble 5 dB | 14.4% | 7.1% | 0.961 | 0.943 | 0.982 | 66.7% |
| reverb RT60 0.4 s | 4.3% | 1.5% | 0.978 | 0.958 | 1.000 | 66.7% |
| reverb RT60 0.8 s | 7.3% | 3.2% | 0.982 | 0.965 | 1.000 | 66.7% |

Purity is 1.000 in every condition: noise splits a single voice into extra clusters but never mixes
voices. Worst single recordings: `lec_photosynthesis_rishi_white_0db` WER 57.4% (70 substitutions,
2 deletions, 29 insertions over 176 words) and `lec_photosynthesis_samantha_white_0db` WER 56.8%.

### 2.3 Per speaking rate (same two lectures and three voices)

| Rate | Recordings | WER | CER | VAD F1 |
|---|---|---|---|---|
| 175 wpm | 6 | 2.5% | 0.6% | 0.981 |
| 140 wpm | 6 | 2.7% | 0.8% | 0.983 |
| 220 wpm | 6 | 2.6% | 0.8% | 0.976 |

### 2.4 Meetings (4 scripts, 3 speakers each, clean and white 10 dB)

| Set | Recordings | WER | Speaker count | Change F1 | Purity | Actions P/R/F1 | Owner acc | Search hit@1 | Search MRR |
|---|---|---|---|---|---|---|---|---|---|
| overall | 8 | 4.1% | 75.0% | 0.994 | 0.989 | 0.95/0.98/0.96 | 88.2% | 75.0% | 0.833 |
| clean | 4 | 1.5% | 75.0% | 0.989 | 0.978 | 0.95/0.95/0.95 | 88.2% | 83.3% | 0.917 |
| white 10 dB | 4 | 6.6% | 75.0% | 1.000 | 1.000 | 0.95/1.00/0.98 | 88.2% | 66.7% | 0.750 |

Per recording: action items TP/FP/FN were 6/0/0, 6/0/0, 4/1/1, 5/1/0, 5/0/0, 5/0/0, 5/0/0, 5/0/0
(41 of 42 labelled actions found, 2 false alarms, both "we need to ..." sentences the script does
not label). Speaker count was wrong in `meet_incident_review` (4 found, purity 0.911) and
`meet_course_staff_white_10db` (4 found, purity 1.000). Owner misses are name recognition
("Mei" heard as "May").

### 2.5 Lecture products (42 clean lecture recordings, 126 queries)

| Keyword recall@10 | Key-sentence recall | Search hit@1 | Search hit@3 | Search MRR |
|---|---|---|---|---|
| 52.1% | 54.0% | 82.5% | 99.2% | 0.902 |

By script (keywords hit/total, key sentences hit/total): photosynthesis 54/110 and 21/33, newton
66/110 and 21/33, binary_search 20/50 and 10/15, supply_demand 30/50 and 5/15, water_cycle 30/50
and 6/15, french_revolution 19/50 and 5/15.

### 2.6 Latency (all 110 recordings, 4 threads)

| Stage | Seconds per audio minute | RTF |
|---|---|---|
| ASR (faster-whisper base int8) | 2.86 | 0.048 |
| VAD | 0.014 | |
| Speaker clustering | 0.140 | |
| Keywords, notes, actions | 0.001 | |
| Full pipeline | 3.01 | 0.050 |
| Model load (once) | 0.20 s | |

### 2.7 Speaker threshold calibration

Command: `uv run python -m echo.calibrate` (train: scripts photosynthesis, newton, binary_search,
course_staff, research_weekly; test: the other five scripts).

| threshold | train count acc | train purity | test count acc | test purity |
|---|---|---|---|---|
| 1.5 | 0.527 | 1.000 | 0.947 | 0.995 |
| 2.0 | 0.582 | 1.000 | 0.947 | 0.995 |
| 2.5 | 0.692 | 1.000 | 0.947 | 0.995 |
| 3.0 | 0.747 | 1.000 | 0.947 | 0.995 |
| 3.5 | 0.791 | 0.997 | 0.895 | 0.982 |
| 4.0 | 0.791 | 0.993 | 0.842 | 0.970 |
| 5.0 | 0.780 | 0.990 | 0.789 | 0.956 |
| 6.0 | 0.790 | 0.990 | 0.842 | 0.942 |
| 8.0 | 0.824 | 0.987 | 0.789 | 0.942 |

The shipped threshold is 3.0: the largest value that keeps purity at 1.000 on the training half
while the held-out half stays at 0.947 count accuracy. The training half contains all 60 noise-axis
lectures, which is why its count accuracy is lower; the test half is clean lectures and two meetings.

### 2.8 Earlier measurements that shaped the design

- Pairwise speaker features on the quick profile (7 recordings): region-mean MFCCs gave a
  within-voice maximum distance of 5.46 against a between-voice minimum of 2.85 (no separation);
  median pitch alone 1.11 against 0.98; symmetric KL over frame Gaussians with pitch had a within
  90th percentile of 1.67 against a between 5th percentile of 3.75, which is why the shipped
  method uses frame distributions and merges greedily with re-estimation.
- Spectral subtraction before the mel filterbank changed count accuracy from 0.782 to 0.791 on the
  full corpus and was not adopted; dropping pitch cut meeting count accuracy from 0.75 to 0.125.
- Keyword prior sweep on the six reference texts: plain RAKE (three-word phrases) covered 17 of 60
  key terms, single-word RAKE 23, the shipped prior 33.
- Achieved SNR check: `add_noise` lands within 0.01 dB of the requested SNR (unit test with
  hypothesis over SNR in [-5, 30] dB and random seeds).

## 3. Tests

`uv run pytest -q`: 126 tests in 11 files: unit, integration (API through the
ASGI test client, CLI end to end), property-based (hypothesis: PCM round trip, edit distance
invariants, identical-text WER, achieved SNR), one determinism test on the full product set
(`test_same_input_same_output`), adversarial and robustness cases (silence, DC-clipped, NaN, 50-sample
and loud-noise inputs through the whole pipeline; non-WAV uploads; oversize uploads; over-long audio;
wrong API key), and one whisper test that runs when the base model is cached and skips otherwise.

## 4. Command outputs (final run from a clean shell)

Shell: `env -i HOME=$HOME PATH=$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin /bin/zsh -c 'make lint && make test && make bench'`.

`make lint` (tail):

```
uv run ruff check src tests
All checks passed!
uv run ruff format --check src tests
54 files already formatted
```

`make test` (tail):

```
uv run pytest
........................................................................ [ 57%]
......................................................                   [100%]
126 passed in 5.74s
```

`make bench` (tail, full log in the terminal; tables in `benchmarks/results.md`):

```

## Latency (all recordings)

| Stage | Seconds per audio minute | RTF |
|---|---|---|
| ASR | 2.86 | 0.048 |
| VAD | 0.014 |  |
| Speakers | 0.140 |  |
| Products | 0.001 |  |
| Full pipeline | 3.01 | 0.050 |
| Model load (once) | 0.20 s |  |

quality gate: passed
exit 0
```


`make docker` and a container smoke test (`docker run --rm -p 18080:8000 -e ECHO_ASR_BACKEND=mock echo:latest`):

```
docker image ls echo:latest      -> echo:latest 911MB
GET /health                      -> {"status":"ok","version":"0.1.0","backend":"mock","model":"mock-oracle-wer0.00","auth_required":false,"jobs":0,"demo_available":true}
POST /v1/demo                    -> 201 (bundled clip, since the container has no `say`)
docker exec echo-smoke id        -> uid=999(echo) gid=999(echo) groups=999(echo)
```

## 5. Known gaps and limitations

1. Everything is measured on synthetic `say` voices. They are cleaner, steadier and never overlap.
   Real recordings will score worse on every row; there is no human-recorded validation set.
2. Speaker counting over-segments noisy single-speaker lectures (22 of 60 noise-axis lectures were
   split; purity stayed 1.000). Two of eight meetings returned four clusters instead of three.
3. The speaker threshold, the VAD defaults and the keyword prior constants were set on this corpus
   (`docs/EVALUATION.md` lists exactly what was tuned where). Keyword recall@10 of 52.1% is
   therefore an in-sample number on the text side.
4. Keyword recall@10 (52.1%) and key-sentence recall (54.0%) are the weakest products; both need a
   background corpus (IDF) that the offline, dependency-light design does not yet include.
5. WER at 0 dB white noise is 39.4% and up to 57% on single recordings; whisper base is the
   limiting factor and the harness makes that visible rather than hiding it.
6. Owner extraction depends on names being recognised; 88.2% owner accuracy on the meetings.
7. The service processes whole files synchronously and the HTTP request waits for the result: at the
   measured 2.53 s per audio minute (four threads) a one-hour lecture would take about 2.5 minutes.
   No streaming, English only.
8. The numbers in this report come from the local full run. CI on GitHub Actions runs lint and the
   tests on Linux and macOS and the quick benchmark gate on macOS on every push, mirroring the local
   `make lint`, `make test` and `make bench-quick` commands.
9. The Docker image builds the Linux arm64 wheels of ctranslate2 and onnxruntime; it was built
   locally (see section 4) but not run against the whisper weights inside the container.
10. No LLM enhancement was added; the roadmap keeps it optional and behind a flag.
