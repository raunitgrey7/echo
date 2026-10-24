# Evaluation

Every number in README.md and REPORT.md comes from one command:

```
make bench            # full profile: 110 recordings, 136 minutes of audio, whisper base int8
make bench-quick      # 7 recordings, the CI gate
```

Both write `benchmarks/results.json` (every per-recording score plus pooled groups) and
`benchmarks/results.md` (the tables), then compare dotted metric paths against
`benchmarks/thresholds.json` and exit non-zero on a violation.

## Dataset: synthetic speech with certain labels

There is no licensed corpus. Ground truth comes from macOS `say`: every sentence of six lecture
scripts and every turn of four meeting scripts is synthesised once per (voice, speaking rate) and
cached under `data/cache/`. Recordings are composed from those utterances with seeded silence gaps
(0.7 to 1.4 s between lecture sentences, 0.5 to 1.2 s between meeting turns; both longer than the
pauses `say` puts at commas, so sentence spans are a consistent reference for the VAD), then
degraded, peak-normalised to -1 dBFS and written as 16 kHz mono 16-bit WAV with a JSON manifest.
Each manifest records the SHA-256 of the audio, the reference text, every utterance's speech span
(trimmed at 40 dB below the clip's peak), speaker, action-item label with owner and deadline, the
lecture's key terms and key sentences, and search queries with the index of the answering turn.
`load_dataset` verifies every checksum before a run.

Seeds: each recording's seed is the first four bytes of `sha256(recording id)`, so gaps and noise
never depend on build order. Building the full corpus takes about five minutes on an M-series Mac
and happens automatically on the first `make bench`.

### Grid (full profile, 110 recordings)

| Axis | Recordings | What varies |
|---|---|---|
| voice | 30 | 6 lectures x 5 voices (Samantha en_US, Daniel en_GB, Karen en_AU, Moira en_IE, Rishi en_IN), clean, 175 wpm |
| noise | 60 | 2 lectures x 3 voices x 10 conditions: white 20/10/5/0 dB, pink 10/5 dB, babble 10/5 dB, reverberation RT60 0.4/0.8 s |
| rate | 12 | 2 lectures x 3 voices x 140 and 220 wpm |
| meeting | 8 | 4 meetings x {clean, white 10 dB}, three voices each |

The quick profile keeps 4 clean lectures, 2 noisy ones and 1 meeting.

### Degradation models

- **White noise**: Gaussian, flat spectrum, the electronic noise floor.
- **Pink noise**: 1/f spectrum by frequency-domain shaping, the usual shape of room and ventilation noise.
- **Babble**: four other talkers (voices Tessa, Fred and Aman reading lectures 4 to 6, so the babble never contains the target text) looped from random offsets.
- **SNR** is measured against the speech-active RMS (frames above the 40th energy percentile), so pauses do not inflate the signal level. A test asserts the achieved SNR is within 0.01 dB of the request.
- **Reverberation**: Schroeder model, a direct impulse followed by Gaussian noise decaying as `exp(-ln(1000) t / RT60)`, scaled so the reverberant-to-direct energy ratio is 0 dB at RT60 = 0.5 s and grows with RT60. The wet signal is RMS-matched to the dry one.

## Metrics

| Metric | Definition | Where |
|---|---|---|
| WER, CER | Levenshtein alignment after normalisation; errors pooled over total reference words (characters), not averaged per file. Substitution, deletion and insertion rates are reported separately. | `echo.metrics.wer` |
| Normalisation | NFKC, lowercase, punctuation removed (intra-word apostrophes kept), hyphens to spaces, thousands separators removed, `%` to `percent`, and spoken numbers folded to digits on both sides (`forty` and `40` agree; `twelfth` and `12th` agree). No spelling map: a British spelling against an American one counts as an error, and the scripts use American spelling. | `echo.metrics.text_norm` |
| VAD precision, recall, F1 | Frame level on a 10 ms grid against the sentence spans of the clean synthesis. Pauses inside a sentence count as speech. | `echo.metrics.vad` |
| Speaker count accuracy | Predicted number of clusters equals the number of speakers (1 for lectures, 3 for meetings). | `echo.bench` |
| Change-point F1 | Greedy one-to-one matching of predicted and reference turn boundaries within 0.75 s; reference boundaries are gap midpoints where the speaker changes. | `echo.metrics.diarization` |
| Purity | Duration-weighted fraction of predicted-cluster time that belongs to the cluster's majority true speaker. | `echo.metrics.diarization` |
| Keyword recall@10 | Fraction of the lecture's 10 expert key terms covered by one of the 10 extracted phrases (match on shared stemmed tokens). | `echo.products.keywords.keyword_hits` |
| Key-sentence recall | Fraction of the 3 sentences the script marks essential that appear in the summary with token Jaccard at least 0.5 (so an ASR slip does not break the match). | `echo.metrics.products` |
| Action precision, recall, F1 | Each detected item is mapped to the labelled turn it overlaps most in time; true positive if that turn is labelled an action and not already matched. | `echo.metrics.products` |
| Owner accuracy | Among matched items with a labelled owner: name match, or the sentence's own speaker label when the commitment is first person (the system never learns names). | `echo.metrics.products` |
| Search hit@1, hit@3, MRR | Rank of the first result whose span overlaps the answering turn (0.5 s tolerance), over 3 queries per script. | `echo.metrics.products` |
| Latency | Seconds of processing per minute of audio and real-time factor, per stage, measured with `time.perf_counter` around each stage; model load time measured once. | `echo.pipeline`, `echo.bench` |

## What was tuned on this corpus

Honesty about in-sample settings:

- **Speaker clustering threshold** (`distance_threshold = 3.0`) was swept on the half of the corpus whose scripts are photosynthesis, newton, binary_search, course_staff and research_weekly (91 recordings) and checked on the other half (19 recordings); REPORT.md lists both halves. The pitch weight (1.0, semitones) and the use of voiced frames were chosen from a feature-separation analysis on the quick profile.
- **Keyword prior constants** (length slope 0.5, no phrase bonus, cue boost 3.0, capital boost 1.5, opening boost 2.0) were swept on the six reference lecture texts, which are the same scripts the benchmark uses. The reported keyword recall is therefore in-sample for the text side; the ASR side is not.
- **VAD defaults** (15th percentile floor, 9 dB onset shrinking to a fraction of the dynamic range, 0.5 s minimum silence) were set by inspection on the quick profile.
- **Action cue weights** were written by hand before scoring and not swept; the single change after scoring was raising `we should <verb>` to 1.5 and adding interjections to the vocative name filter.

## What the mock backend measures

`ECHO_ASR_BACKEND=mock` replaces recognition with the manifest text spread over the VAD regions.
Its WER is zero by construction and never appears as a recognition result; it exists so the VAD,
speaker, product and latency plumbing can be gated in CI without a model download. Mock rows are
flagged `is_mock: true` in `results.json` and the Markdown carries a warning line.

## Reproducing

```
make setup                      # uv sync
make bench                      # ~10 minutes on an M-series Mac with 4 threads
ECHO_WHISPER_MODEL=small make bench PROFILE=quick   # another model size
python -m echo.bench --profile full --backend mock  # plumbing only, no recognition
```

The run is deterministic for a fixed machine: whisper decodes greedily at temperature zero and
every random draw is seeded. Latency numbers vary with hardware and load and are reported with the
machine and thread count in `results.json` under `meta`.
