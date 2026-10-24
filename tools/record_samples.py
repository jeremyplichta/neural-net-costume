#!/usr/bin/env python3
"""Record labelled 1 second training clips for the neural-net costume.

For each prompt the word + emoji is shown; say it and the recorder starts as
soon as the mic level peaks (keeping a short pre-roll so the start isn't cut
off). Clips are saved as data/training/<word>-<n>.wav (16 kHz, mono, 16-bit).

    python tools/record_samples.py                 # 10 takes of every word, shuffled rounds
    python tools/record_samples.py -w ghost witch  # only some words
    python tools/record_samples.py -n 25           # takes per word (this session)
    python tools/record_samples.py --list-devices
"""
import argparse
import queue
import random
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from recorder import SAMPLE_RATE, TriggeredRecorder  # noqa: E402
from words import EMOJI, LABELS  # noqa: E402

BLOCK = 512  # 32 ms at 16 kHz
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "training"


def next_index(out_dir, word):
    pat = re.compile(rf"^{re.escape(word)}-(\d+)\.wav$")
    nums = [int(m.group(1)) for f in out_dir.glob(f"{word}-*.wav") if (m := pat.match(f.name))]
    return max(nums, default=0) + 1


def save_wav(path, clip):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(clip.astype("<i2").tobytes())


def drain(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def calibrate(q, seconds=1.5):
    """Measure ambient noise; trigger at a multiple of its peak."""
    drain(q)
    peaks, end = [], time.time() + seconds
    while time.time() < end:
        try:
            peaks.append(int(np.abs(q.get(timeout=0.2).astype(np.int32)).max()))
        except queue.Empty:
            pass
    ambient = max(peaks, default=0)
    return max(int(ambient * 3), 1500), ambient


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-w", "--words", nargs="+", default=LABELS, choices=LABELS, metavar="WORD",
                    help=f"words to record (default: all of {', '.join(LABELS)})")
    ap.add_argument("-n", "--takes", type=int, default=10, help="takes per word this session")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help="output folder")
    ap.add_argument("-t", "--threshold", type=int, default=None,
                    help="trigger level 0-32767 (default: auto from ambient noise)")
    ap.add_argument("-d", "--device", default=None, help="input device index or name")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--no-shuffle", action="store_true", help="go through words in order")
    args = ap.parse_args()

    import sounddevice as sd  # imported late so --help and tests work without audio libs
    if args.list_devices:
        print(sd.query_devices())
        return
    device = int(args.device) if args.device and args.device.isdigit() else args.device

    args.out.mkdir(parents=True, exist_ok=True)
    q = queue.Queue()
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=BLOCK,
                            device=device, callback=lambda data, *_: q.put(data[:, 0].copy()))
    with stream:
        if args.threshold is None:
            print("Measuring room noise... stay quiet.")
            threshold, ambient = calibrate(q)
            print(f"Ambient peak {ambient}, trigger threshold {threshold} (override with -t).")
        else:
            threshold = args.threshold
        rec = TriggeredRecorder(threshold)

        rounds = []
        for _ in range(args.takes):
            r = list(args.words)
            if not args.no_shuffle:
                random.shuffle(r)
            rounds += r
        total = len(rounds)
        print(f"\n{total} clips to record. Say each word when it appears. Ctrl-C to stop.\n")
        time.sleep(1)

        try:
            for i, word in enumerate(rounds, 1):
                print(f"\033[2J\033[H\n\n    [{i}/{total}]\n\n    {EMOJI[word]}   \033[1m{word.upper()}\033[0m\n")
                time.sleep(0.4)  # let the screen/your own keypress settle before arming
                drain(q)
                rec.reset()
                clip = None
                while clip is None:
                    clip = rec.feed(q.get())
                path = args.out / f"{word}-{next_index(args.out, word)}.wav"
                save_wav(path, clip)
                peak = int(np.abs(clip.astype(np.int32)).max())
                print(f"    saved {path.name}  (peak {peak}{' - CLIPPING' if peak >= 32000 else ''})")
                time.sleep(0.6)
        except KeyboardInterrupt:
            print("\nStopped.")
    print(f"Done. Clips are in {args.out}")


if __name__ == "__main__":
    main()
