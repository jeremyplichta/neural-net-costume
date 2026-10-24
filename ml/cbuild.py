"""Compile a firmware C source into a shared library and load it with ctypes."""
import ctypes
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "ml" / ".build"


def load_c_library(lib_dir, name):
    """Build firmware/lib/<lib_dir>/src/<name>.c (if stale) and return the CDLL."""
    src_dir = ROOT / "firmware" / "lib" / lib_dir / "src"
    srcs = [src_dir / f"{name}.c", src_dir / f"{name}.h"]
    BUILD_DIR.mkdir(exist_ok=True)
    so = BUILD_DIR / f"lib{name}.{'dylib' if sys.platform == 'darwin' else 'so'}"
    if not so.exists() or so.stat().st_mtime < max(s.stat().st_mtime for s in srcs):
        subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-I", str(src_dir),
                        str(srcs[0]), "-o", str(so), "-lm"], check=True)
    return ctypes.CDLL(str(so))
