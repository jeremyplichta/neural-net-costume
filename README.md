# Neural Net Costume 🎃

A wearable, *working* neural network for Halloween. A T-shirt carries four bands of
addressable LEDs — one band per layer, 10 "neurons" per band — joined by metallic cord
"synapses". You say a spooky word, and the network lights up as it runs the inference
live on the shirt.

| Layer | Neurons | Colour | Band position |
|---|---|---|---|
| Input | 10 | orange | top |
| Hidden 1 | 10 | green | 4–5" down |
| Hidden 2 | 10 | purple | 4–5" down |
| Output | 10 | blue | bottom |

**Hardware:** Adafruit Sparkle Motion Stick (ESP32, with microphone) running our own
firmware, driving a 100-LED NeoPixel Pebble strand (4" pitch → 40 used), 1 mm metallic
cord, 1" non-roll black elastic for the bands.

## What it recognises

Nine words plus a catch-all, matching the 10 output neurons (order defined in
[`tools/words.py`](tools/words.py)):

| 💬 other | 👻 ghost | 🎃 boo | 💀 spooky | 🧙 witch | 🍬 candy | 🐈‍⬛ cat | 🦇 bat | 🕷️ spider | ☠️ skull |
|---|---|---|---|---|---|---|---|---|---|

`other` = anything else: random speech, background noise, silence. It is the class the
network falls back to, so record plenty of it.

## Pipeline

```
 mic (1 s, 16 kHz) → feature extraction (10 values) → 10 → 10 → 10 → 10 → argmax → word
                          input layer                 hidden hidden  output
```

The same feature extractor and the trained weights must run on the ESP32, so the
network is deliberately tiny (10-10-10-10 ≈ 300 weights) and the features are cheap.

## Repository layout

```
tools/       Python helpers (data collection now; feature extraction later)
  record_samples.py   prompt-and-record training clips
  recorder.py         triggered recording with pre-roll (unit-tested)
  audit_clips.py      flag suspect clips (clipped / weak / late bump); --move quarantines them
  words.py            the vocabulary / emoji / output order
  tests/
data/
  training/           recorded clips: <word>[-<speaker>]-<n>.wav (git-ignored)
  quarantine/         clips set aside by tools/audit_clips.py, with manifest.csv (git-ignored)
ml/          feature extraction, training, evaluation, export (PyTorch)
  features.py         Python wrapper around the firmware's C feature extractor
  extract_dataset.py  wavs -> data/features.npz
  train.py            plain-PyTorch training loop; compares the feature modes
  evaluate.py         accuracy + confusion matrix, using the firmware's C forward pass
  export_weights.py   trained model -> nn_weights.h for the ESP32
  tests/
firmware/
  lib/nnfeatures/     C feature extractor, shared by training AND the ESP32
  lib/nnforward/      C forward pass of the 10-10-10-10 net, shared the same way
  (mic capture, LEDs: planned)
```

## Roadmap

- [x] Repo structure and README
- [x] **Training-data recorder** (`tools/record_samples.py`)
- [ ] Record a dataset (aim for 30+ clips per word, several voices/distances, lots of `other`)
- [x] Audio feature extractor → 10 inputs: one C source (`firmware/lib/nnfeatures`) used by both training (via ctypes) and the ESP32
- [x] Training / evaluation / export scaffold (PyTorch); tested on synthetic data only so far
- [ ] Train on real recordings, compare feature modes, pick one
- [ ] ESP32 firmware: mic → `nnf_extract` → `nn_forward`
- [ ] LED animation: light layers as activations propagate; highlight the winning output
- [ ] Build the shirt (bands, wiring, cord)

## Recording training data

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r tools/requirements.txt     # sounddevice needs PortAudio (macOS: built in; Linux: apt install libportaudio2)
python tools/record_samples.py             # 10 takes of every word, shuffled
```

For each prompt the terminal shows the word and its emoji. Just say it: recording begins
when the mic level crosses a threshold, but a 0.25 s rolling pre-roll buffer is kept so the
start of the word is never chopped. Every clip is exactly 1 s, 16 kHz mono 16-bit, saved as
`data/training/ghost-1.wav`, `ghost-2.wav`, … (numbering continues across sessions).

Useful options: `-w ghost witch` (subset), `-n 25` (takes per word), `-t 4000` (manual
trigger level; default is auto-calibrated from room noise), `-d <device>` /
`--list-devices` (pick a mic), `--no-shuffle`, and `-s <name>` to tag a speaker (files become
`ghost-wife-1.wav`; numbering is per speaker). Without `-s`, files are `ghost-1.wav`.
Tagging speakers lets us hold one voice out as a test set later.

Tips: vary distance, volume and speaker; record `other` with ordinary chatter, music and
silence; run `python tools/audit_clips.py` to find clipped / no-signal / bump-only clips (add `--move`
to quarantine them, `--restore` to undo), or delete bad clips by hand. Run the tests with
`python -m pytest tools/tests`.

## Features: one C implementation for training and the ESP32

The Sparkle Motion Stick is an ESP32-S3 (dual-core Xtensa LX7, 240 MHz) with an I2S MEMS
mic (WS/DATA/BCLK on GPIO12/13/14 per Adafruit's pinout; LED outputs on GPIO21/22; verify
against the board). So the feature code is plain C with no dependencies, compiled for the
board and, for training, into a shared library that `ml/features.py` loads.

Front end (all modes): 1 s of 16 kHz audio -> 61 Hann frames (512, hop 256) -> 10 mel bands
(100-7600 Hz). Three ways to turn that into the 10 input neurons, to be compared on real data:

| Mode | The 10 inputs |
|---|---|
| `mean` | mean log-energy of each band over the clip. No timing information (baseline) |
| `contours` | 5 loudness time-bins + 5 pitch time-bins |
| `hybrid` | 4 loudness bins + 3 pitch bins + 3 2D-DCT coefficients (spectral tilt, bump, and tilt-over-time sweep) |

*Loudness* = mean ln(total band energy) in the bin; *pitch* = energy-weighted mean band index
scaled 0..1 (0.5 when the bin is >20 dB below the loudest, so silence doesn't produce noise).
Append `_norm` (`hybrid_norm`, ...) to subtract the clip's loudest-frame level from the
ln-energy features, removing recording volume (pitch bins and DCT terms are already
volume-independent). Exact definitions are in `firmware/lib/nnfeatures/src/nnfeatures.h`. Values are not
normalised in C; training computes mean/std and the firmware applies them.

Try it: `python ml/features.py --mode hybrid data/training/ghost-1.wav`. Tests
(`python -m pytest ml/tests`) compare every mode to an independent numpy implementation and
check the behaviours we care about (timing in loudness bins, rising vs falling sweeps).

## Training and exporting the network

```bash
pip install -r ml/requirements.txt
python ml/extract_dataset.py                       # data/training/*.wav -> data/features.npz
python ml/train.py                                 # compare mean / contours / hybrid features
python ml/train.py --mode hybrid --holdout wife    # test on a voice it never trained on
python ml/train.py --loso --seeds 3                # every speaker held out in turn, all modes
python ml/evaluate.py ml/models/hybrid.npz         # confusion matrix: which words get mixed up
python ml/export_weights.py ml/models/hybrid.npz   # -> firmware/lib/nnforward/src/nn_weights.h
```

The network is `10 -> 10 -> 10 -> 10`: features are z-scored with the training set's mean and
std, then two sigmoid hidden layers, then softmax over the words (3 weight matrices, 330
parameters). `train.py` is a plain PyTorch loop, short enough to read top to bottom.

**On the ESP32** (`firmware/lib/nnforward`): `nn_forward(&NN_WEIGHTS, features, act)` fills
`act[4][10]`, every value in 0..1 and ready to use as LED brightness: `act[0]` the inputs
(z of -2..+2 mapped to 0..1), `act[1]`, `act[2]` the hidden sigmoids, `act[3]` the softmax
probabilities; it returns the winning output index. `nn_weights.h` also says which feature
mode (`NN_FEATURE_MODE`) to pass to `nnf_extract`. Evaluation in Python runs this same C code,
so the accuracy you see is what the costume will get.

Note: so far this is tested only on synthetic data (easy by construction), so it shows the
pipeline works, not how well the features discriminate real words.

## Results so far

383 clean clips, 4 speakers, 10 words. Test = leave-one-speaker-out (train on three voices,
test on the fourth), 3 seeds; chance is 10%:

| Features | avg accuracy | | Features | avg accuracy |
|---|---|---|---|---|
| `mean` | 39% | | `mean_norm` | 41% |
| `contours` | 35% | | `contours_norm` | 37% |
| `hybrid` | 53% | | `hybrid_norm` | 54% |

Timing information is what helps (`hybrid` >> `mean`); volume normalisation helps little.
Weakest words (pooled hybrid_norm): candy 31%, spider 30%, other 41%; common confusions are
spider->skull, cat<->bat, ghost<->witch. Re-run `ml/train.py --loso` as data grows.
