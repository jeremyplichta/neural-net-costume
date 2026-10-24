import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
import export_weights  # noqa: E402
import extract_dataset  # noqa: E402
import features as F  # noqa: E402
import train as T  # noqa: E402
from nn import CForward, make_model, save_model  # noqa: E402
from record_samples import save_wav  # noqa: E402
from words import LABELS  # noqa: E402


def random_saved_model(tmp_path, seed=0):
    torch.manual_seed(seed)
    model = make_model()
    rng = np.random.default_rng(seed)
    mean, std = rng.normal(0, 2, 10), rng.uniform(0.5, 3, 10)
    path = tmp_path / "m.npz"
    save_model(path, model, mean, std, "hybrid")
    return model, mean, std, path


def torch_forward(model, mean, std, x):
    with torch.no_grad():
        z = torch.tensor((x - mean) / std, dtype=torch.float32)
        h1 = torch.sigmoid(model[0](z))
        h2 = torch.sigmoid(model[2](h1))
        return h1.numpy(), h2.numpy(), torch.softmax(model[4](h2), 0).numpy()


def test_c_forward_matches_pytorch(tmp_path):
    model, mean, std, path = random_saved_model(tmp_path)
    net = CForward(path)
    for x in np.random.default_rng(1).normal(0, 3, (20, 10)):
        win, act = net.forward(x)
        h1, h2, out = torch_forward(model, mean, std, x)
        np.testing.assert_allclose(act[1], h1, atol=1e-5)
        np.testing.assert_allclose(act[2], h2, atol=1e-5)
        np.testing.assert_allclose(act[3], out, atol=1e-5)
        assert win == out.argmax()
        assert ((act >= 0) & (act <= 1)).all() and abs(act[3].sum() - 1) < 1e-5


def test_input_display_value_is_clamped_z(tmp_path):
    _, mean, std, path = random_saved_model(tmp_path)
    net = CForward(path)
    x = mean.copy()
    x[0] += 10 * std[0]   # +10 sigma
    x[1] -= 10 * std[1]
    x[2] += 1 * std[2]    # +1 sigma -> 0.75
    act = net.forward(x)[1]
    assert act[0, 0] == 1.0 and act[0, 1] == 0.0 and abs(act[0, 2] - 0.75) < 1e-5
    assert abs(act[0, 3] - 0.5) < 1e-5


def test_exported_header_matches(tmp_path):
    _, mean, std, path = random_saved_model(tmp_path, seed=3)
    src_dir = HERE.parent / "firmware" / "lib" / "nnforward" / "src"
    (tmp_path / "nn_weights.h").write_text(export_weights.header(path))
    (tmp_path / "main.c").write_text('''
#include <stdio.h>
#include "nn_weights.h"
int main(void) {
    float f[NN_N], act[NN_LAYERS][NN_N];
    for (int i = 0; i < NN_N; i++) f[i] = 0.3f * i - 1.0f;
    int w = nn_forward(&NN_WEIGHTS, f, act);
    printf("%d", w);
    for (int i = 0; i < NN_N; i++) printf(" %.6f", act[3][i]);
    printf(" %s\\n", NN_LABELS[w]);
    return NN_FEATURE_MODE == NNF_MODE_HYBRID ? 0 : 1;
}''')
    nnf = HERE.parent / "firmware" / "lib" / "nnfeatures" / "src"
    subprocess.run(["cc", "-Wall", "-I", str(src_dir), "-I", str(nnf), str(tmp_path / "main.c"),
                    str(src_dir / "nnforward.c"), "-o", str(tmp_path / "a.out"), "-lm"], check=True)
    out = subprocess.run([str(tmp_path / "a.out")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    toks = out.stdout.split()
    x = 0.3 * np.arange(10) - 1.0
    win, act = CForward(path).forward(x)
    assert int(toks[0]) == win and toks[-1] == LABELS[win]
    np.testing.assert_allclose([float(t) for t in toks[1:11]], act[3], atol=1e-5)


def fake_dataset(n_per=30, seed=0):
    """10 separable classes in 10-D, 3 speakers; all feature modes share the same X."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (10, 10))
    y = np.repeat(np.arange(10), n_per)
    X = (centers[y] + rng.normal(0, 0.7, (len(y), 10))).astype(np.float32)
    speaker = np.array(["", "wife", "kid"])[np.arange(len(y)) % 3]
    return {f"X_{m}": X for m in F.MODES} | {"y": y, "speaker": speaker}


def args(tmp_path, **kw):
    return Namespace(**{"epochs": 150, "lr": 0.01, "seeds": 1, "holdout": None,
                        "out_dir": tmp_path, **kw})


def test_training_learns_separable_data(tmp_path):
    r = T.run_mode(fake_dataset(), "hybrid", args(tmp_path), verbose=False)
    assert r["acc"] > 0.9 and r["n_train"] + r["n_test"] == 300


def test_holdout_speaker_split_and_training(tmp_path):
    d = fake_dataset()
    tr, te = T.split(d["y"], d["speaker"], holdout="kid")
    assert set(d["speaker"][te]) == {"kid"} and "kid" not in set(d["speaker"][tr])
    r = T.run_mode(d, "mean", args(tmp_path, holdout="kid", seeds=2), verbose=False)
    assert r["acc"] > 0.9 and r["n_test"] == 100 and len(r["pred"]) == 100
    r = T.run_mode(d, "mean", args(tmp_path, seeds=1), verbose=False, holdout="wife")  # per-call override
    assert r["n_test"] == 100


def test_wav_to_features_to_model_end_to_end(tmp_path):
    """Synthetic 'words' = tones of different pitch, written as real wav files."""
    wavs = tmp_path / "wavs"
    wavs.mkdir()
    t = np.arange(F.CLIP_SAMPLES) / F.SAMPLE_RATE
    rng = np.random.default_rng(0)
    for i, word in enumerate(LABELS):
        for n in range(1, 13):
            speaker = "" if n <= 6 else "kid"
            tone = 6000 * np.sin(2 * np.pi * (200 + 600 * i) * t) * (t > 0.1 * (i % 4)) \
                + rng.normal(0, 300, len(t))
            name = f"{word}-{speaker + '-' if speaker else ''}{n}.wav"
            save_wav(wavs / name, tone.astype(np.int16))
    (wavs / "junk.wav").write_bytes(b"")  # unparseable name must be skipped, not crash
    data = extract_dataset.build(wavs)
    assert data["X_hybrid"].shape == (120, 10) and set(data["speaker"]) == {"", "kid"}
    r = T.run_mode(data, "contours", args(tmp_path, epochs=300, holdout="kid"), verbose=False)
    assert r["acc"] > 0.9


def test_parse_name():
    assert extract_dataset.parse_name("ghost-3.wav") == ("ghost", "")
    assert extract_dataset.parse_name("spider-wife-12.wav") == ("spider", "wife")
    assert extract_dataset.parse_name("dragon-1.wav") is None
    assert extract_dataset.parse_name("ghost.wav") is None
