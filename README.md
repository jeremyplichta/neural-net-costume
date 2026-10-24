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
  words.py            the vocabulary / emoji / output order
  tests/
data/
  training/           recorded clips: <word>-<n>.wav (git-ignored)
ml/          feature extraction + training (PyTorch or Keras)
  features.py         Python wrapper around the firmware's C feature extractor
  tests/
firmware/
  lib/nnfeatures/     C feature extractor, shared by training AND the ESP32
  (mic capture, inference, LEDs: planned)
```

## Roadmap

- [x] Repo structure and README
- [x] **Training-data recorder** (`tools/record_samples.py`)
- [ ] Record a dataset (aim for 30+ clips per word, several voices/distances, lots of `other`)
- [x] Audio feature extractor → 10 inputs: one C source (`firmware/lib/nnfeatures`) used by both training (via ctypes) and the ESP32
- [ ] Train the 10-10-10-10 network (PyTorch or Keras), evaluate, export weights
- [ ] ESP32 firmware: mic → features → inference
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
silence; delete bad clips by hand (just remove the file). Run the tests with
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
Exact definitions are in `firmware/lib/nnfeatures/src/nnfeatures.h`. Values are not
normalised in C; training computes mean/std and the firmware applies them.

Try it: `python ml/features.py --mode hybrid data/training/ghost-1.wav`. Tests
(`python -m pytest ml/tests`) compare every mode to an independent numpy implementation and
check the behaviours we care about (timing in loudness bins, rising vs falling sweeps).
