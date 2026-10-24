import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_clips import audit  # noqa: E402

RATE = 16000


def word(onset=0.25, dur=0.3, amp=8000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 80, RATE)
    a, b = int(onset * RATE), int((onset + dur) * RATE)
    t = np.arange(b - a) / RATE
    x[a:b] += amp * np.sin(2 * np.pi * 400 * t)
    return x.astype(np.int16)


def test_clean_clip_passes_even_when_quiet():
    assert audit(word()) == []
    assert audit(word(amp=1500)) == []  # quiet but clear: not flagged


def test_heavy_clipping_flagged_light_clipping_not():
    loud = np.clip(word().astype(np.int32) * 6, -32768, 32767).astype(np.int16)  # saturates like a real mic
    assert any(r.startswith("clipped") for r in audit(loud))
    x = word()
    x[1000:1005] = 32767
    assert audit(x) == []


def test_weak_signal_flagged():
    x = np.random.default_rng(0).normal(0, 300, RATE).astype(np.int16)  # noise only
    assert any(r.startswith("weak") for r in audit(x))


def test_late_loud_burst_flagged():
    x = word(amp=2000)
    x[int(0.9 * RATE):int(0.95 * RATE)] += 20000
    assert any(r.startswith("late") for r in audit(x))
