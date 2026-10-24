"""Python access to the *same* C feature extractor the ESP32 firmware uses.

Compiles firmware/lib/nnfeatures/src/nnfeatures.c on first use and calls it via ctypes.
    python ml/features.py data/training/ghost-1.wav
"""
import ctypes
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "firmware" / "lib" / "nnfeatures" / "src"
BUILD_DIR = ROOT / "ml" / ".build"
NUM_FEATURES = 10
SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000

_lib = None


def _load():
    global _lib
    if _lib is None:
        BUILD_DIR.mkdir(exist_ok=True)
        ext = "dylib" if sys.platform == "darwin" else "so"
        so = BUILD_DIR / f"libnnfeatures.{ext}"
        srcs = [SRC_DIR / "nnfeatures.c", SRC_DIR / "nnfeatures.h"]
        if not so.exists() or so.stat().st_mtime < max(s.stat().st_mtime for s in srcs):
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-I", str(SRC_DIR),
                            str(srcs[0]), "-o", str(so), "-lm"], check=True)
        lib = ctypes.CDLL(str(so))
        lib.nnf_extract.argtypes = [ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t,
                                    ctypes.POINTER(ctypes.c_float)]
        lib.nnf_extract.restype = None
        _lib = lib
    return _lib


def extract(pcm):
    """int16 mono array (>=512 samples, normally 16000) -> float32[NUM_FEATURES]."""
    pcm = np.ascontiguousarray(pcm, dtype=np.int16)
    out = np.zeros(NUM_FEATURES, dtype=np.float32)
    _load().nnf_extract(pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(pcm),
                        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return out


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, \
            f"{path}: expected 16 kHz mono 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, np.array2string(extract(read_wav(p)), precision=3))
