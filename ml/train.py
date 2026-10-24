"""Train the 10-10-10-10 network on extracted features and compare feature modes.

    python ml/extract_dataset.py                         # once, after recording
    python ml/train.py                                   # compare mean / contours / hybrid
    python ml/train.py --mode hybrid --holdout wife      # test on a voice it never trained on
    python ml/train.py --mode hybrid --seeds 5           # average over random restarts

Without --holdout, a random 20% of clips (per word) is held out. Models are saved to
ml/models/<mode>.npz and can be exported to the ESP32 with ml/export_weights.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
from evaluate import accuracy, report  # noqa: E402
from features import MODES  # noqa: E402
from nn import CForward, make_model, save_model  # noqa: E402


def split(y, speaker, holdout=None, test_frac=0.2, seed=0):
    """Return (train_idx, test_idx): a held-out speaker, or a random per-word split."""
    if holdout is not None:
        test = speaker == holdout
        if not test.any():
            raise SystemExit(f"no clips from speaker {holdout!r}; have {sorted(set(speaker))}")
        return np.where(~test)[0], np.where(test)[0]
    rng = np.random.default_rng(seed)
    test = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        test[idx[:max(1, int(len(idx) * test_frac))]] = True
    return np.where(~test)[0], np.where(test)[0]


def train(X, y, epochs=500, lr=0.01, batch_size=32, seed=0, log=None):
    """Train on (X, y); returns (model, mean, std). X is raw features, normalised here."""
    torch.manual_seed(seed)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6   # statistics of the training set only
    inputs = torch.tensor((X - mean) / std, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.long)

    model = make_model()
    loss_fn = nn.CrossEntropyLoss()                    # softmax + log loss
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(inputs))
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            loss = loss_fn(model(inputs[batch]), labels[batch])  # how wrong are we?
            opt.zero_grad()
            loss.backward()                                      # which way to nudge each weight
            opt.step()                                           # nudge them
        if log and epoch % 100 == 0:
            with torch.no_grad():
                train_acc = (model(inputs).argmax(1) == labels).float().mean().item()
            log(f"  epoch {epoch:4d}  loss {loss.item():.3f}  train acc {train_acc:.1%}")
    return model, mean, std


def run_mode(data, mode, args, verbose):
    """Train (possibly several seeds) on one feature mode; returns (mean acc, std, n_train, n_test)."""
    X, y, speaker = data[f"X_{mode}"], data["y"], data["speaker"]
    accs = []
    for seed in range(args.seeds):
        tr, te = split(y, speaker, args.holdout, seed=seed)
        model, mean, std = train(X[tr], y[tr], args.epochs, args.lr, seed=seed,
                                 log=print if verbose and seed == 0 else None)
        if seed == 0:  # keep this model, and score it with the C code the ESP32 will run
            path = args.out_dir / f"{mode}.npz"
            save_model(path, model, mean, std, mode)
            pred = CForward(path).predict(X[te])
            if verbose:
                report(y[te], pred)
        else:
            with torch.no_grad():
                pred = model(torch.tensor((X[te] - mean) / std, dtype=torch.float32)).argmax(1).numpy()
        accs.append(accuracy(y[te], pred))
    return float(np.mean(accs)), float(np.std(accs)), len(tr), len(te)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=HERE.parent / "data" / "features.npz")
    ap.add_argument("--mode", choices=[*MODES, "all"], default="all")
    ap.add_argument("--holdout", default=None, help="speaker to hold out as the test set ('' = untagged)")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=1, help="random restarts to average")
    ap.add_argument("--out-dir", type=Path, default=HERE / "models")
    args = ap.parse_args()

    data = np.load(args.features)
    modes = list(MODES) if args.mode == "all" else [args.mode]
    results = {}
    for mode in modes:
        print(f"\n=== {mode} ===")
        results[mode] = run_mode(data, mode, args, verbose=len(modes) == 1)
    print("\nmode        test accuracy   (train / test clips)")
    for mode, (mean, std, ntr, nte) in results.items():
        print(f"{mode:10s}  {mean:6.1%} ± {std:4.1%}    ({ntr} / {nte})")
