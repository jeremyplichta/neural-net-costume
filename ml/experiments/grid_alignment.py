"""Prototype: raw time x pitch grid inputs (+ optional onset alignment), leave-one-speaker-out.

Pure numpy/torch re-implementation of the front end (NOT the firmware C code); it exists to
choose a design before it is built into firmware/lib/nnfeatures. Level normalisation matches
the C `_norm` modes: each clip's loudest-frame level (10-band total) is subtracted.

    python ml/experiments/grid_alignment.py --grids 5x4 5x6 10x10
    python ml/experiments/grid_alignment.py --grids 5x4 --align none onset15:38 centroid:50

--align specs: none | onset<dB>:<frames> | centroid:<frames>   (a frame is 16 ms)
  onset15:38  window of 38 frames (0.6 s) starting 2 frames before the first frame within
              15 dB of the loudest; centroid:50 window centred on the energy centroid.
Needs data/features.npz (ml/extract_dataset.py) and the wavs in data/training/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
import features as F  # noqa: E402

FRAME, HOP, FRAMES = 512, 256, 61
WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / FRAME)
nats = lambda db: db * np.log(10) / 10


def mel_filterbank(n, fmin=100.0, fmax=7600.0):
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(mel(fmin), mel(fmax), n + 2))
    hz = np.arange(FRAME // 2 + 1) * F.SAMPLE_RATE / FRAME
    fb = np.zeros((n, len(hz)))
    for b in range(n):
        lo, mid, hi = edges[b:b + 3]
        up, dn = (hz > lo) & (hz <= mid), (hz > mid) & (hz < hi)
        fb[b, up] = (hz[up] - lo) / (mid - lo)
        fb[b, dn] = (hi - hz[dn]) / (hi - mid)
    return fb


FB10 = mel_filterbank(10)


def power_spectrogram(pcm):
    x = pcm.astype(np.float64) / 32768
    return np.array([np.abs(np.fft.rfft(x[s:s + FRAME] * WINDOW)) ** 2
                     for s in range(0, len(x) - FRAME + 1, HOP)])  # (61, 257)


def grid_features(spec, t_bins, m_bands, fb, align="none"):
    """(t_bins x m_bands) mean log-mel grid, level-normalised, flattened."""
    E = spec @ fb.T
    ltot = np.log((spec @ FB10.T).sum(1) + 1e-6)
    level, floor = ltot.max(), np.percentile(ltot, 10)
    start, width = 0, FRAMES
    if align.startswith("onset"):
        db, w = align[5:].split(":")
        width = int(w)
        thr = max(level - nats(float(db)), floor + nats(6))
        start = int(np.argmax(ltot >= thr)) - 2
    elif align.startswith("centroid"):
        width = int(align.split(":")[1])
        wt = np.maximum(ltot - floor - nats(6), 0)
        centre = (np.arange(FRAMES) * wt).sum() / wt.sum() if wt.sum() > 0 else FRAMES // 2
        start = int(round(centre - width / 2))
    pad = np.log(E.min(0) + 1e-6)  # stand-in for "silence" outside the clip
    out = np.zeros((t_bins, m_bands))
    for t in range(t_bins):
        s, e = start + t * width // t_bins, start + (t + 1) * width // t_bins
        rows = [np.log(E[f] + 1e-6) if 0 <= f < FRAMES else pad for f in range(s, max(e, s + 1))]
        out[t] = np.mean(rows, axis=0) - level
    return out.ravel()


def fit_predict(X, y, tr, te, seed, hidden=10, epochs=500):
    torch.manual_seed(seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    a = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
    lab = torch.tensor(y[tr])
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.Sigmoid(),
                        nn.Linear(hidden, hidden), nn.Sigmoid(), nn.Linear(hidden, 10))
    opt, loss_fn = torch.optim.Adam(net.parameters(), lr=0.01), nn.CrossEntropyLoss()
    for _ in range(epochs):
        order = torch.randperm(len(a))
        for i in range(0, len(order), 32):
            b = order[i:i + 32]
            loss = loss_fn(net(a[b]), lab[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(torch.tensor((X[te] - mu) / sd, dtype=torch.float32)).argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", nargs="+", default=["5x4"], help="TxM, e.g. 5x4 (T time slices x M bands)")
    ap.add_argument("--align", nargs="+", default=["none"])
    ap.add_argument("--features", type=Path, default=ROOT / "data" / "features.npz")
    ap.add_argument("--wavs", type=Path, default=ROOT / "data" / "training")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=10)
    args = ap.parse_args()

    d = np.load(args.features)
    y, speaker = d["y"], d["speaker"]
    specs = [power_spectrogram(F.read_wav(args.wavs / f)) for f in d["files"]]
    speakers = sorted(set(speaker))
    print(f"{'config':22s}{'inputs':>7s}" + "".join(f"{(s or 'jp'):>8s}" for s in speakers) + "   average")
    for g in args.grids:
        t_bins, m_bands = map(int, g.split("x"))
        fb = mel_filterbank(m_bands)
        for align in args.align:
            X = np.array([grid_features(s, t_bins, m_bands, fb, align) for s in specs], dtype=np.float32)
            accs = []
            for sp in speakers:
                tr, te = np.where(speaker != sp)[0], np.where(speaker == sp)[0]
                accs.append(np.mean([(fit_predict(X, y, tr, te, k, args.hidden) == y[te]).mean()
                                     for k in range(args.seeds)]))
            print(f"{g + ' ' + align:22s}{X.shape[1]:7d}" + "".join(f"{a:8.1%}" for a in accs)
                  + f"   {np.mean(accs):7.1%}", flush=True)


if __name__ == "__main__":
    main()
