# echo benchmark (full profile)

Backend: `faster-whisper` model `base-int8`, 110 recordings, 136.0 min of audio, 4 CPU threads, macOS-26.5.1-arm64-arm-64bit-Mach-O. Run at 2026-10-24T23:56:49+00:00, commit `a50fd4d`.

## Per voice (clean lectures)

| Voice | Recordings | WER | CER | Sub | Del | Ins | VAD F1 | ASR RTF |
|---|---|---|---|---|---|---|---|---|
| overall | 30 | 2.8% | 1.0% | 1.7% | 0.6% | 0.6% | 0.979 | 0.047 |
| Daniel | 6 | 3.1% | 1.0% | 1.8% | 0.5% | 0.8% | 0.979 | 0.049 |
| Karen | 6 | 3.5% | 1.3% | 2.0% | 0.8% | 0.8% | 0.979 | 0.048 |
| Moira | 6 | 2.9% | 1.0% | 1.6% | 0.6% | 0.7% | 0.980 | 0.044 |
| Rishi | 6 | 3.1% | 0.9% | 1.8% | 0.5% | 0.8% | 0.980 | 0.047 |
| Samantha | 6 | 1.5% | 0.5% | 1.0% | 0.5% | 0.0% | 0.977 | 0.047 |

## Per acoustic condition (two lectures, three voices)

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

## Per speaking rate

| Rate | Recordings | WER | CER | VAD F1 |
|---|---|---|---|---|
| 175 wpm | 6 | 2.5% | 0.6% | 0.981 |
| 140 wpm | 6 | 2.7% | 0.8% | 0.983 |
| 220 wpm | 6 | 2.6% | 0.8% | 0.976 |

## Meetings (speaker turns, action items, search)

| Set | Recordings | WER | Speaker count | Change F1 | Purity | Actions P/R/F1 | Owner acc | Search hit@1 | Search MRR |
|---|---|---|---|---|---|---|---|---|---|
| overall | 8 | 4.1% | 75.0% | 0.994 | 0.989 | 0.95/0.98/0.96 | 88.2% | 75.0% | 0.833 |
| clean | 4 | 1.5% | 75.0% | 0.989 | 0.978 | 0.95/0.95/0.95 | 88.2% | 83.3% | 0.917 |
| white_10db | 4 | 6.6% | 75.0% | 1.000 | 1.000 | 0.95/1.00/0.98 | 88.2% | 66.7% | 0.750 |

## Lecture products (clean)

| Recordings | Keyword recall@10 | Key-sentence recall | Search hit@1 | Search MRR |
|---|---|---|---|---|
| 42 | 52.1% | 54.0% | 82.5% | 0.902 |

## Latency (all recordings)

| Stage | Seconds per audio minute | RTF |
|---|---|---|
| ASR | 2.86 | 0.048 |
| VAD | 0.014 |  |
| Speakers | 0.140 |  |
| Products | 0.001 |  |
| Full pipeline | 3.01 | 0.050 |
| Model load (once) | 0.20 s |  |
