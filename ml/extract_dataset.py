"""Turn data/training/*.wav into a feature file for training.

    python ml/extract_dataset.py                  # -> data/features.npz
Filenames: <word>-<n>.wav or <word>-<speaker>-<n>.wav (see tools/record_samples.py).
The file holds X_mean / X_contours / X_hybrid (one row per clip), y (label index into
tools/words.py), speaker ("" if untagged) and the source file names.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
import features as F  # noqa: E402
from words import LABELS  # noqa: E402


def parse_name(path):
    """'ghost-wife-3.wav' -> ('ghost', 'wife'); 'ghost-3.wav' -> ('ghost', ''); else None."""
    parts = Path(path).stem.split("-")
    if parts[0] in LABELS and parts[-1].isdigit() and len(parts) in (2, 3):
        return parts[0], parts[1] if len(parts) == 3 else ""
    return None


def build(wav_dir):
    X = {m: [] for m in F.MODES}
    y, speaker, files = [], [], []
    for wav in sorted(Path(wav_dir).glob("*.wav")):
        parsed = parse_name(wav)
        if parsed is None:
            print(f"skipping {wav.name}: not <word>[-<speaker>]-<n>.wav")
            continue
        pcm = F.read_wav(wav)
        for m in F.MODES:
            X[m].append(F.extract(pcm, m))
        y.append(LABELS.index(parsed[0]))
        speaker.append(parsed[1])
        files.append(wav.name)
    return {**{f"X_{m}": np.array(v, dtype=np.float32) for m, v in X.items()},
            "y": np.array(y), "speaker": np.array(speaker), "files": np.array(files)}


if __name__ == "__main__":
    root = HERE.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wavs", type=Path, default=root / "data" / "training")
    ap.add_argument("--out", type=Path, default=root / "data" / "features.npz")
    args = ap.parse_args()
    data = build(args.wavs)
    np.savez(args.out, **data)
    print(f"{len(data['y'])} clips -> {args.out}")
    for i, name in enumerate(LABELS):
        print(f"  {name:8s} {int((data['y'] == i).sum())}")
    print("speakers:", sorted(set(data["speaker"])) or "none")
