import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features as F  # noqa: E402

N = F.NUM_FEATURES


def mel_energies(pcm, frame=512, hop=256, fmin=100.0, fmax=7600.0):
    """Independent numpy front end: 61 x 10 linear mel-band energies of a 1 s clip."""
    pcm = np.pad(pcm[:F.CLIP_SAMPLES], (0, max(0, F.CLIP_SAMPLES - len(pcm))))
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(mel(fmin), mel(fmax), N + 2))
    hz = np.arange(frame // 2 + 1) * F.SAMPLE_RATE / frame
    fb = np.zeros((N, len(hz)))
    for b in range(N):
        lo, mid, hi = edges[b:b + 3]
        up, dn = (hz > lo) & (hz <= mid), (hz > mid) & (hz < hi)
        fb[b, up] = (hz[up] - lo) / (mid - lo)
        fb[b, dn] = (hi - hz[dn]) / (hi - mid)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame)
    x = pcm.astype(np.float64) / 32768
    spec = [np.abs(np.fft.rfft(x[s:s + frame] * win)) ** 2
            for s in range(0, len(x) - frame + 1, hop)]
    return np.array(spec) @ fb.T


def bins(n_bins, n_frames):
    return [(i * n_frames // n_bins, (i + 1) * n_frames // n_bins) for i in range(n_bins)]


def ref_loud(E, n):
    return [np.log(E[s:e].sum(axis=1) + 1e-6).mean() for s, e in bins(n, len(E))]


def ref_pitch(E, n):
    num = [(E[s:e] * np.arange(N)).sum() for s, e in bins(n, len(E))]
    den = np.array([E[s:e].sum() for s, e in bins(n, len(E))])
    return [x / (d * (N - 1)) if d > 0.01 * den.max() else 0.5 for x, d in zip(num, den)]


def dct(g, kt, kf):
    n = np.arange(N)
    a = lambda k: np.sqrt((1 if k == 0 else 2) / N)
    ct = np.cos(np.pi * (2 * n + 1) * kt / (2 * N))
    cf = np.cos(np.pi * (2 * n + 1) * kf / (2 * N))
    return a(kt) * a(kf) * ct @ g @ cf


def reference(pcm, mode):
    E = mel_energies(pcm)
    norm = mode.endswith("_norm")
    mode = mode.removesuffix("_norm")
    level = np.log(E.sum(axis=1) + 1e-6).max() if norm else 0.0
    if mode == "mean":
        return np.log(E + 1e-6).mean(axis=0) - level
    if mode == "contours":
        return np.array(np.array(ref_loud(E, 5)) - level).tolist() + ref_pitch(E, 5)
    grid = np.array([np.log(E[s:e] + 1e-6).mean(axis=0) for s, e in bins(N, len(E))])
    return np.array((np.array(ref_loud(E, 4)) - level).tolist() + ref_pitch(E, 3) +
                    [dct(grid, 0, 1), dct(grid, 0, 2), dct(grid, 1, 1)])


def speech_like(seed=2):
    """Noise bursts with different pitch/timing so every feature is exercised."""
    rng = np.random.default_rng(seed)
    t = np.arange(F.CLIP_SAMPLES) / F.SAMPLE_RATE
    return (4000 * np.sin(2 * np.pi * (300 + 3000 * t) * t) * (t > 0.2) * (t < 0.7)
            + rng.normal(0, 200, len(t))).astype(np.int16)


def test_all_modes_match_numpy_reference():
    for pcm in (np.random.default_rng(1).normal(0, 3000, F.CLIP_SAMPLES).astype(np.int16),
                speech_like()):
        for mode in F.MODES:
            np.testing.assert_allclose(F.extract(pcm, mode), reference(pcm, mode),
                                       atol=3e-3, err_msg=mode)


def test_short_clip_is_zero_padded():
    pcm = speech_like()[:9000]
    padded = np.pad(pcm, (0, F.CLIP_SAMPLES - len(pcm)))
    for mode in F.MODES:
        np.testing.assert_allclose(F.extract(pcm, mode), F.extract(padded, mode), atol=1e-6)


def test_tone_lands_in_expected_band():
    t = np.arange(F.CLIP_SAMPLES) / F.SAMPLE_RATE
    lo = F.extract((8000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16))
    hi = F.extract((8000 * np.sin(2 * np.pi * 4000 * t)).astype(np.int16))
    assert lo.argmax() < hi.argmax()


def test_loudness_bins_follow_timing():
    pcm = np.zeros(F.CLIP_SAMPLES, dtype=np.int16)
    pcm[:6000] = np.random.default_rng(0).normal(0, 5000, 6000)
    loud = F.extract(pcm, "contours")[:5]
    assert loud[0] > loud[1] > loud[4] + 5  # speech early, silence late


def test_pitch_bins_follow_a_rising_sweep_and_dct_sweep_term_flips():
    t = np.arange(F.CLIP_SAMPLES) / F.SAMPLE_RATE
    up = (8000 * np.sin(2 * np.pi * (200 + 3000 * t) * t)).astype(np.int16)
    down = up[::-1].copy()
    pu, pd = F.extract(up, "contours")[5:], F.extract(down, "contours")[5:]
    assert pu[0] < pu[2] < pu[4] and pd[0] > pd[2] > pd[4]
    su, sd = F.extract(up, "hybrid")[9], F.extract(down, "hybrid")[9]
    assert su * sd < 0  # opposite sweeps give opposite sign


def test_silence_is_finite_and_wav_roundtrip(tmp_path):
    sil = np.zeros(F.CLIP_SAMPLES, dtype=np.int16)
    for mode in F.MODES:
        assert np.isfinite(F.extract(sil, mode)).all()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from record_samples import save_wav
    save_wav(tmp_path / "x.wav", sil)
    assert (F.read_wav(tmp_path / "x.wav") == sil).all()


def test_normalised_modes_ignore_volume_but_raw_modes_do_not():
    quiet = speech_like()
    loud = (quiet.astype(np.int32) * 4).clip(-32768, 32767).astype(np.int16)
    for mode in ("mean", "contours", "hybrid"):
        raw_gap = np.abs(F.extract(loud, mode) - F.extract(quiet, mode)).max()
        norm_gap = np.abs(F.extract(loud, mode + "_norm") - F.extract(quiet, mode + "_norm")).max()
        assert norm_gap < 0.05, mode
        assert mode == "hybrid" or raw_gap > 1.0, mode  # loudness-based features move with volume
    assert abs(F.extract(quiet, "contours_norm")[:5].max()) < 0.05  # loudest bin ~ 0 dB
