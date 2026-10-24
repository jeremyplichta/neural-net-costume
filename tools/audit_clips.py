#!/usr/bin/env python3
"""Find recordings that are probably bad and (optionally) set them aside.

    python tools/audit_clips.py              # dry run: list suspects and why
    python tools/audit_clips.py --move       # move them to data/quarantine/ (+ manifest.csv)
    python tools/audit_clips.py --restore    # put everything in quarantine back

Heuristics, so a human should glance at the list:
  clipped   many samples at full scale (distorted)
  weak      loudest frame is barely above the background (no clear word)
  late      loudest burst is late in the clip; the recorder triggers ~0.25 s in, so a
            louder sound at the end is probably a bump/click, not the word
Quiet-but-clean clips are NOT flagged: that is a volume problem, fixed by normalisation.
"""
import argparse
import csv
import shutil
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRAME, HOP = 512, 256


def read_pcm(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def audit(pcm, max_clipped=100, min_snr_db=15.0, late_ms=500, rate=16000):
    """Return a list of human-readable reasons this clip looks bad (empty = fine)."""
    x = pcm.astype(np.float64)
    reasons = []
    n_clip = int((np.abs(x) >= 32000).sum())
    if n_clip >= max_clipped:
        reasons.append(f"clipped: {n_clip} samples at full scale")
    rms = np.array([np.sqrt((x[i:i + FRAME] ** 2).mean()) for i in range(0, len(x) - FRAME + 1, HOP)])
    db = 20 * np.log10(rms + 1)
    snr = db.max() - np.percentile(db, 10)
    if snr < min_snr_db:
        reasons.append(f"weak: loudest frame only {snr:.0f} dB above background")
    start_ms = int(np.argmax(db > db.max() - 15)) * HOP * 1000 // rate
    if start_ms > late_ms:
        reasons.append(f"late: loudest sound starts at {start_ms} ms")
    return reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "training")
    ap.add_argument("--quarantine", type=Path, default=ROOT / "data" / "quarantine")
    ap.add_argument("--move", action="store_true", help="move suspects to the quarantine folder")
    ap.add_argument("--restore", action="store_true", help="move everything in quarantine back")
    ap.add_argument("--max-clipped", type=int, default=100)
    ap.add_argument("--min-snr", type=float, default=15.0)
    ap.add_argument("--late-ms", type=int, default=500)
    args = ap.parse_args()
    manifest = args.quarantine / "manifest.csv"

    if args.restore:
        wavs = sorted(args.quarantine.glob("*.wav"))
        for f in wavs:
            shutil.move(str(f), args.data / f.name)
        manifest.unlink(missing_ok=True)
        print(f"restored {len(wavs)} clips to {args.data}")
        return

    suspects = []
    for f in sorted(args.data.glob("*.wav")):
        reasons = audit(read_pcm(f), args.max_clipped, args.min_snr, args.late_ms)
        if reasons:
            suspects.append((f, reasons))
            print(f"{f.name:22s} " + "; ".join(reasons))
    total = len(list(args.data.glob("*.wav")))
    print(f"\n{len(suspects)} of {total} clips flagged" + ("" if args.move else " (dry run; --move to quarantine)"))
    if args.move and suspects:
        args.quarantine.mkdir(parents=True, exist_ok=True)
        new = not manifest.exists()
        with open(manifest, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["file", "reasons"])
            for f, reasons in suspects:
                w.writerow([f.name, "; ".join(reasons)])
                shutil.move(str(f), args.quarantine / f.name)
        print(f"moved to {args.quarantine} (manifest.csv lists reasons)")


if __name__ == "__main__":
    main()
