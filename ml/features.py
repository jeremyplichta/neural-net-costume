"""Python access to the *same* C feature extractor the ESP32 firmware uses.

Compiles firmware/lib/nnfeatures/src/nnfeatures.c on first use and calls it via ctypes.
    python ml/features.py [--mode mean|contours|hybrid] data/training/ghost-1.wav

Modes (see nnfeatures.h for the exact definitions); each gives 10 features:
    mean      10 band energies averaged over the clip (no timing)
    contours  5 loudness time-bins + 5 pitch time-bins
    hybrid    4 loudness bins + 3 pitch bins + 3 2D-DCT coefficients
"""
import ctypes
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbuild import load_c_library  # noqa: E402

NUM_FEATURES = 10
SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000

MODES = {"mean": 0, "contours": 1, "hybrid": 2}
FEATURE_NAMES = {
    "mean": [f"band{i}" for i in range(10)],
    "contours": [f"loud{i}" for i in range(5)] + [f"pitch{i}" for i in range(5)],
    "hybrid": [f"loud{i}" for i in range(4)] + [f"pitch{i}" for i in range(3)]
    + ["dct_tilt", "dct_bump", "dct_sweep"],
}

_lib = None


def _load():
    global _lib
    if _lib is None:
        lib = load_c_library("nnfeatures", "nnfeatures")
        lib.nnf_extract.argtypes = [ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_float)]
        lib.nnf_extract.restype = None
        _lib = lib
    return _lib


def extract(pcm, mode="mean"):
    """int16 mono array (normally 16000 samples) -> float32[NUM_FEATURES]."""
    pcm = np.ascontiguousarray(pcm, dtype=np.int16)
    out = np.zeros(NUM_FEATURES, dtype=np.float32)
    _load().nnf_extract(pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(pcm), MODES[mode],
                        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return out


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, \
            f"{path}: expected 16 kHz mono 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = "mean"
    if args[:1] == ["--mode"]:
        mode, args = args[1], args[2:]
    for p in args:
        vals = extract(read_wav(p), mode)
        print(p, " ".join(f"{n}={v:.3f}" for n, v in zip(FEATURE_NAMES[mode], vals)))
