# Architecture

echo turns one audio file into a set of products: a timestamped transcript with speaker labels, a
keyword timeline, extractive notes, action items and a searchable index. Every entry point (CLI,
HTTP API, benchmark) calls the same `echo.pipeline.analyze` function, so a number measured by the
benchmark describes the code path the API serves.

## Data flow

```
                 +-------------------+
 WAV (any rate)  | audio.io          |  float32 mono 16 kHz
 --------------->| read_wav/resample |------------------------------+
                 +-------------------+                              |
                                                                    v
   +---------------------+      +----------------------+     +--------------------+
   | vad.energy          |      | asr.whisper / mock   |     | diarization.change |
   | band energy, floor, |      | word timestamps,     |     | pitch + MFCC per   |
   | hysteresis, merge   |      | temperature 0        |     | region, KL merge   |
   +----------+----------+      +-----------+----------+     +---------+----------+
              | regions                      | segments + words        | labels per region
              +---------------+--------------+-------------------------+
                              v
                   products.transcript.assign_speakers   (overlap in time)
                              |
          +-------------------+-------------------+-------------------+
          v                   v                   v                   v
   products.keywords   products.notes      products.actions    products.search
   (specificity +      (TextRank, TextTiling (cue weights,      (BM25 over
    cues, timeline)     sections)           owner, deadline)    sentences)
          |                   |                   |                   |
          +-------------------+---------+---------+-------------------+
                                        v
                               pipeline.Analysis
                        (+ per-stage timings, RTF)
                                        |
                 +----------------------+----------------------+
                 v                      v                      v
            cli.py (files)        api.app (JSON, SRT,      bench.py (scores vs
                                  VTT, dashboard)          manifest labels)
```

## Modules

| Module | Responsibility | Why it is shaped this way |
|---|---|---|
| `echo.audio.io` | PCM WAV in and out, resampling, levels | Python 3.13 removed `aifc` and `audioop`; one internal format (16 kHz mono float32) keeps every stage simple. |
| `echo.audio.synth` | macOS `say` harness, `afconvert`, numpy AIFF decoder | Ground truth for free: each (voice, rate, text) triple is a labelled sample; the numpy decoder is the fallback when afconvert is missing. |
| `echo.audio.degrade` | White, pink and babble noise at a target SNR; Schroeder reverberation by RT60 | Degradations with units so results read as curves; every draw is seeded. |
| `echo.audio.features` | Framing, windows, power spectra | One frame grid shared by VAD, MFCC and pitch. |
| `echo.corpus.scripts` | Six lectures and four meetings with key terms, key sentences, action labels and queries | Labels the recogniser cannot have seen. |
| `echo.corpus.build` | Composes recordings from cached utterances, applies conditions, writes manifests with SHA-256 | Reproducible dataset; sentence spans give the VAD and speaker references. |
| `echo.asr.base` | `Transcriber` protocol, `Word`, `Segment`, `TranscriptionResult` | Engine-agnostic boundary. |
| `echo.asr.whisper` | faster-whisper, CPU int8, beam 5, temperature 0, word timestamps | Deterministic output for a given input; numpy input so ffmpeg is not needed. |
| `echo.asr.mock` | Deterministic oracle-backed stand-in | Tests, CI and the demo run without a model; rows are labelled `mock`. |
| `echo.asr.registry` | `auto`, `whisper` or `mock` from `ECHO_ASR_BACKEND` | One documented fallback that is logged. |
| `echo.vad.energy` | Band-limited energy VAD with percentile floor, adaptive onset, hysteresis | Fast, dependency-free, measurable against clean-signal spans. |
| `echo.diarization.mfcc`, `pitch`, `change` | MFCCs, autocorrelation pitch, agglomerative clustering with symmetric KL between diagonal Gaussians | Pitch separates the corpus voices; re-estimating the Gaussian after each merge made the clustering stable; the number of speakers falls out of a distance threshold. |
| `echo.metrics.*` | Text normalisation with number folding, WER/CER with operation counts, VAD frame scores, change-point F1 and purity, product scores | Every product has a label and a score. |
| `echo.products.transcript` | Speaker assignment by overlap, SRT, VTT, text, JSON | The product every other product builds on. |
| `echo.products.keywords` | Specificity-weighted phrases with definitional, capitalisation and opening cues; timestamps per phrase | A one-transcript extractor without a background corpus. |
| `echo.products.notes` | Sentence split across segments, TF-IDF, TextRank, TextTiling sections | Deterministic extractive notes with timestamps. |
| `echo.products.actions` | Weighted cue sum, owner and deadline extraction | Transparent and auditable. |
| `echo.products.search` | BM25 inverted index over sentences | Jump to the moment a question was answered. |
| `echo.pipeline` | Orders the stages, times them, builds `Analysis`, JSON view | Single shared path. |
| `echo.bench` | Builds data, runs the pipeline, scores, writes `benchmarks/results.{json,md}`, checks thresholds | Every published number comes from this command. |
| `echo.api.app`, `middleware`, `schemas`, `static/` | FastAPI routes, request ids, JSON logs, Prometheus, API key, dashboard | Operable service with a player synced to the transcript. |
| `echo.cli` | `transcribe`, `analyze`, `synth`, `voices`, `build-data`, `bench`, `serve` | Everything scriptable without HTTP. |

## Runtime configuration

| Variable | Default | Meaning |
|---|---|---|
| `ECHO_ASR_BACKEND` | `auto` | `auto` uses whisper when its weights are cached, else the mock; `whisper` or `mock` force a backend. |
| `ECHO_WHISPER_MODEL` | `base` | faster-whisper size (`tiny`, `base`, `small`, `medium`). |
| `ECHO_CPU_THREADS` | `4` | CTranslate2 threads; modest by default because other work shares the machine. |
| `ECHO_ALLOW_DOWNLOAD` | `0` | Set to `1` to let faster-whisper download missing weights. |
| `APP_API_KEY` | unset | When set, `/v1/*` requires the `X-API-Key` header. |
| `ECHO_JOB_DIR` | `data/jobs` | Where finished jobs (WAV + JSON) are kept. |
| `ECHO_MAX_UPLOAD_MB` | `50` | Upload size limit (413 above it). |
| `ECHO_MAX_AUDIO_S` | `3600` | Audio length limit (422 above it). |
| `ECHO_LOG_LEVEL` | `INFO` | JSON log level. |
| `HF_HOME` / `HF_HUB_CACHE` | `~/.cache/huggingface` | Where whisper weights live. |

## Determinism

Whisper decodes with beam search at temperature zero and no fallback sampling; the mock seeds its
corruption from the audio checksum; the corpus seeds every recording from the SHA-256 of its id;
TextRank starts from a uniform vector and iterates to a tolerance; all ties in ranking break on
time of first occurrence. The test `test_same_input_same_output` asserts the full product set is
identical across two runs.

## What is deliberately not here

No GPU path, no training, no network access at runtime, no LLM dependency. An LLM could polish
the extractive notes behind a flag; it would never be required for tests or the benchmark.
