"""The 10-10-10-10 network: PyTorch definition, saved-model format, and the C forward pass.

Training uses PyTorch (make_model). Everything else, including evaluation, goes through
the same C code the ESP32 runs (CForward), loaded from a saved .npz model.
"""
import ctypes
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbuild import load_c_library  # noqa: E402

N = 10  # neurons per layer
LAYERS = 4


def make_model():
    """input(10) -> hidden1(10) -> hidden2(10) -> output(10). Softmax is in the loss / C code."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(N, N), nn.Sigmoid(),   # hidden 1
        nn.Linear(N, N), nn.Sigmoid(),   # hidden 2
        nn.Linear(N, N),                 # output logits
    )


def save_model(path, model, mean, std, mode):
    """Write a plain-numpy model file: W (3,10,10) as [layer][out][in], b (3,10), mean, std."""
    linears = [m for m in model if hasattr(m, "weight")]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             W=np.stack([l.weight.detach().numpy() for l in linears]).astype(np.float32),
             b=np.stack([l.bias.detach().numpy() for l in linears]).astype(np.float32),
             mean=np.asarray(mean, dtype=np.float32), std=np.asarray(std, dtype=np.float32),
             mode=mode)


class _Weights(ctypes.Structure):  # mirrors nn_weights_t in nnforward.h
    _fields_ = [("mean", ctypes.c_float * N), ("std", ctypes.c_float * N),
                ("W", ctypes.c_float * N * N * (LAYERS - 1)),
                ("b", ctypes.c_float * N * (LAYERS - 1))]


class CForward:
    """Run a saved model through the firmware's nn_forward()."""

    def __init__(self, path):
        m = np.load(path)
        self.mode = str(m["mode"])
        self.w = _Weights()
        for name in ("mean", "std", "W", "b"):
            arr = np.ascontiguousarray(m[name], dtype=np.float32)
            ctypes.memmove(getattr(self.w, name), arr.ctypes.data, arr.nbytes)
        self.lib = load_c_library("nnforward", "nnforward")
        self.lib.nn_forward.argtypes = [ctypes.POINTER(_Weights), ctypes.POINTER(ctypes.c_float),
                                        ctypes.POINTER(ctypes.c_float)]
        self.lib.nn_forward.restype = ctypes.c_int

    def forward(self, features):
        """Raw features (10,) -> (winner index, activations (4,10), all in 0..1)."""
        f = np.ascontiguousarray(features, dtype=np.float32)
        act = np.zeros((LAYERS, N), dtype=np.float32)
        win = self.lib.nn_forward(ctypes.byref(self.w),
                                  f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return win, act

    def predict(self, X):
        return np.array([self.forward(x)[0] for x in X])
