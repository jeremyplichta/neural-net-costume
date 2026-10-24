# Handoff: where the project stands

For a fresh session (or reviewer) picking this up. The training audio is **private and not in
this repo** (see "Data"); everything else is. Last updated 2026-10-24.

## Goal

Wearable neural net for Halloween. T-shirt with 4 bands of 10 addressable LEDs (input,
hidden 1, hidden 2, output) joined by metallic cord. The wearer says one of 9 words (ghost,
boo, spooky, witch, candy, cat, bat, spider, skull) or anything else (`other`); the network
runs live on an ESP32-S3 (Adafruit Sparkle Motion Stick, I2S mic) and the LEDs show each
neuron's activation as brightness. The user wants to read and understand all training and
evaluation code, and needs per-neuron 0..1 values on the device for animation.

## What exists (all tested, 23 tests: `python -m pytest ml/tests tools/tests`)

| Piece | Where | Notes |
|---|---|---|
| Recorder | `tools/record_samples.py`, `tools/recorder.py` | 1 s clips, 16 kHz mono; mic-peak trigger with 0.25 s pre-roll; `<word>[-<speaker>]-<n>.wav`; optional `-s speaker` |
| Clip audit | `tools/audit_clips.py` | flags clipped / weak-signal / late-bump clips; `--move` to `data/quarantine/`, `--restore` undoes |
| Feature extractor (C) | `firmware/lib/nnfeatures/src/` | one C source used by training (ctypes, `ml/features.py`) and firmware. Modes: `mean`, `contours`, `hybrid`, each also `_norm` (volume removed) |
| Forward pass (C) | `firmware/lib/nnforward/src/` | 10-10-10-10: z-score -> sigmoid -> sigmoid -> softmax; returns `act[4][10]` all in 0..1 + winner |
| Training / eval | `ml/train.py`, `ml/evaluate.py`, `ml/nn.py` | plain PyTorch; evaluation runs the **C** forward pass; `--loso` = leave-one-speaker-out + pooled confusion matrix |
| Export | `ml/export_weights.py` | model -> `nn_weights.h` (weights + feature mode + labels) |
| Prototype | `ml/experiments/grid_alignment.py` | numpy/torch grid-input experiments; not firmware code |

Not started: ESP32 firmware (mic capture, calling the two C libs, LED animation), augmentation,
public datasets, the shirt itself.

## Data (private, must be supplied)

~400 clips were recorded by 4 speakers (the owner = untagged files, plus three family members
with `-<tag>-` names), 10 takes per word each. They live in `data/training/` on the owner's
machine and are git-ignored (also `data/features.npz`, `data/quarantine/`, `ml/models/*.npz`).
A fresh checkout has none: ask the owner for the clips, put them in `data/training/`, then
`python ml/extract_dataset.py`. Run `python tools/audit_clips.py` (dry run) first; on the
current data it flags 17 of 400 clips (one speaker's whole last take for all 10 words is heavily
clipped; a few weak-signal and late-bump clips). These were moved to quarantine, leaving 383.
The audit heuristics are untuned guesses; a human should listen to the "late" ones.

## Results (leave-one-speaker-out, 383 clips, mean of 3 seeds, ~4 pt noise)

| Network input | Inputs | Avg accuracy |
|---|---|---|
| `mean` / `mean_norm` | 10 | 39% / 41% |
| `contours` / `contours_norm` | 10 | 35% / 37% |
| `hybrid` / `hybrid_norm` | 10 | 53% / 54% |
| **log-mel grid, 5 time x 4 bands** (prototype) | 20 | **65%** |
| grid 3x4, 4x4, 6x4, 8x4, 10x4 | 12-40 | 62, 61, 62, 60, 57% |
| grid 5x3, 5x5, 5x6, 6x5 | 15-30 | 52, 60, 63, 58% |
| grid 5x10, 10x10, 5x20 | 50-100 | 61, 60, 54% |
| 20-input grid + `hybrid_norm` stacked | 60 | 63% |
| onset/centroid-aligned 5x4 grid (various windows) | 20 | 53-62% (worse than unaligned) |

Hidden layers were 10 wide throughout. Pooled `hybrid_norm` confusion: weakest words are
spider (30%), candy (31%), `other` (41%); common mix-ups spider->skull, cat<->bat, ghost<->witch.
Random-split accuracy (voices seen in training) is ~66%, so unseen voices cost ~12 points.

Conclusions so far: raw coarse spectrogram > hand-built summaries; 12-32 inputs is a plateau;
bigger inputs overfit (only ~290 train clips per fold); volume normalisation gives little;
onset alignment hurt (it helped one speaker, hurt another a lot).

## Decisions made

- Plain PyTorch (not Lightning/Keras); weights exported to C headers; hand-written C forward pass.
- Hidden/output layers stay 10 wide so the shirt shows 4 x 10 neurons; the input layer may be
  larger (about 20) with only some shown on the LEDs.
- One C source of truth for features and forward pass, loaded from Python, so training and
  device cannot drift. A tiny self-written radix-2 FFT, no dependency (swap in esp-dsp if slow).
- Evaluate on held-out *speakers*, not random clips.

## Suggested next steps (rough priority)

1. **Build the `grid` input into the C extractor**: a mode with compile-time T x M (start 5x4),
   configurable mel band count (the filterbank is currently fixed at 10 bands), `_norm` level
   subtraction; make `nn_weights_t`/`nn_forward` take an input size other than 10 (first layer
   becomes IN x 10). Add numpy-reference tests like the existing modes. Keep old modes working.
2. **Augmentation** (noise, time shift, gain, small pitch/speed change) applied to training
   clips only; re-run `--loso`. Biggest expected gain for the speaker-generalisation gap.
3. **More data**, ideally more voices; and public datasets (below).
4. Cheap feature ideas to A/B on the grid: MFCC-style DCT + deltas, PCEN instead of log,
   zero-crossing rate / spectral flux as a few extra inputs. (Skip pitch/F0: it encodes the speaker.)
5. Try hidden width 16-20 and weight decay / dropout; multiple seeds and confidence intervals.
6. ESP32 firmware, then LED animation using `act[4][10]`.

### Public datasets (unverified: the cloud sandbox could not reach huggingface.co)

- Google Speech Commands v2 (`google/speech_commands`, CC BY 4.0 per its loader script): 35 words;
  contains "cat" but, as far as I know, not our other words. Useful mainly for `other`.
- Multilingual Spoken Words Corpus (`MLCommons/ml_spoken_words`, CC BY 4.0): forced-aligned
  1 s clips of many English words from Common Voice. Check whether ghost / boo / spooky /
  witch / candy / bat / spider / skull exist and how many clips each has. If used, credit the dataset.
- Domain mismatch matters (different mics/rooms), so evaluate on the family recordings only
  and treat public clips as extra training data.

## Gotchas and known risks

- `nnf_extract` keeps about 7 KB on the stack (two 512-float FFT buffers + 61x10 energies +
  a 10x10 grid); the default Arduino loop task stack is 8 KB, so make these static or give the
  task a bigger stack before running on the board. Also not thread-safe (lazy static tables).
- `logf`/`powf` per frame and a plain complex FFT: not timed on the S3; expect it to be fine for
  1 s clips but measure. The device must trigger and pre-roll the same way the recorder does.
- Hardware facts (S3, I2S mic on GPIO12/13/14, LED outputs on GPIO21/22) come from web search
  of Adafruit/CNX pages; one source said "ESP32 mini module". Verify against the board.
- Python needs `torch` and a C compiler (`cc`); `ml/features.py` builds into `ml/.build/`.
- Speaker label for untagged clips prints as `jp` in `train.py`. Models are saved per mode to
  `ml/models/<mode>.npz` (ignored).
- Results use few seeds and few test clips; resist reading small differences.

## Review checklist for the next session

- Re-run the tests and `python ml/train.py --loso --seeds 3` on the real data to confirm the table.
- Read `nnfeatures.c` / `nnforward.c` critically (numerics, stack, FFT scaling) and confirm the
  numpy reference tests really are independent of the C code.
- Check the audit heuristics and that no private data is committed (`git ls-files data ml/models`).
