"""Evaluate a saved model with the same C forward pass the costume runs.

    python ml/evaluate.py ml/models/hybrid.npz                  # whole dataset
    python ml/evaluate.py ml/models/hybrid.npz --speaker wife   # one voice only
(Scores on data the model trained on are optimistic; use a held-out speaker.)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
from nn import CForward  # noqa: E402
from words import LABELS  # noqa: E402


def confusion_matrix(y_true, y_pred, n=len(LABELS)):
    """cm[true, predicted] = number of clips."""
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean()) if len(y_true) else float("nan")


def report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print(f"accuracy {accuracy(y_true, y_pred):.1%} on {len(y_true)} clips\n")
    print("rows = what you said, columns = what the network heard")
    print("          " + " ".join(f"{l[:5]:>5s}" for l in LABELS) + "   recall")
    for i, label in enumerate(LABELS):
        total = cm[i].sum()
        recall = f"{cm[i, i] / total:6.0%}" if total else "     -"
        print(f"{label:9s} " + " ".join(f"{c:5d}" for c in cm[i]) + "  " + recall)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--features", type=Path, default=HERE.parent / "data" / "features.npz")
    ap.add_argument("--speaker", default=None, help="only clips from this speaker ('' = untagged)")
    args = ap.parse_args()
    net = CForward(args.model)
    data = np.load(args.features)
    keep = np.ones(len(data["y"]), bool) if args.speaker is None else data["speaker"] == args.speaker
    X, y = data[f"X_{net.mode}"][keep], data["y"][keep]
    report(y, net.predict(X))
