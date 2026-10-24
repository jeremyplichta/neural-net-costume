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
ml/          training in PyTorch or Keras                         (planned)
firmware/    ESP32 code: mic capture, features, inference, LEDs   (planned)
```

## Roadmap

- [x] Repo structure and README
- [x] **Training-data recorder** (`tools/record_samples.py`)
- [ ] Record a dataset (aim for 30+ clips per word, several voices/distances, lots of `other`)
- [ ] Audio feature extractor → 10 inputs (e.g. mel/band energies; must be portable to ESP32)
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
`--list-devices` (pick a mic), `--no-shuffle`.

Tips: vary distance, volume and speaker; record `other` with ordinary chatter, music and
silence; delete bad clips by hand (just remove the file). Run the tests with
`python -m pytest tools/tests`.
