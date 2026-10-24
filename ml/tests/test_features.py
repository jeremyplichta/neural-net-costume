import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features as F  # noqa: E402


def numpy_reference(pcm, n=F.NUM_FEATURES, frame=512, hop=256, fmin=100.0, fmax=7600.0):
    """Independent numpy implementation of the documented pipeline."""
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(mel(fmin), mel(fmax), n + 2))
    hz = np.arange(frame // 2 + 1) * F.SAMPLE_RATE / frame
    fb = np.zeros((n, len(hz)))
    for b in range(n):
        lo, mid, hi = edges[b:b + 3]
        up = (hz > lo) & (hz <= mid)
        dn = (hz > mid) & (hz < hi)
        fb[b, up] = (hz[up] - lo) / (mid - lo)
        fb[b, dn] = (hi - hz[dn]) / (hi - mid)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame)
    x = pcm.astype(np.float64) / 32768
    frames = [np.abs(np.fft.rfft(x[s:s + frame] * win)) ** 2
              for s in range(0, len(x) - frame + 1, hop)]
    return np.log(np.array(frames) @ fb.T + 1e-6).mean(axis=0)


def test_matches_numpy_reference_on_noise():
    rng = np.random.default_rng(1)
    pcm = rng.normal(0, 3000, F.CLIP_SAMPLES).astype(np.int16)
    np.testing.assert_allclose(F.extract(pcm), numpy_reference(pcm), atol=2e-3)


def test_tone_lands_in_expected_band():
    t = np.arange(F.CLIP_SAMPLES) / F.SAMPLE_RATE
    lo = F.extract((8000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16))
    hi = F.extract((8000 * np.sin(2 * np.pi * 4000 * t)).astype(np.int16))
    assert lo.argmax() < hi.argmax()


def test_silence_is_finite_and_wav_roundtrip(tmp_path):
    sil = np.zeros(F.CLIP_SAMPLES, dtype=np.int16)
    assert np.isfinite(F.extract(sil)).all()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from record_samples import save_wav
    save_wav(tmp_path / "x.wav", sil)
    assert (F.read_wav(tmp_path / "x.wav") == sil).all()
