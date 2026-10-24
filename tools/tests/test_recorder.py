import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recorder import SAMPLE_RATE, TriggeredRecorder  # noqa: E402
from record_samples import clean_speaker, next_index, save_wav  # noqa: E402

BLOCK = 512


def run(signal, threshold=3000):
    rec = TriggeredRecorder(threshold)
    for i in range(0, len(signal) - BLOCK + 1, BLOCK):
        clip = rec.feed(signal[i:i + BLOCK])
        if clip is not None:
            return clip
    return None


def test_clip_is_one_second_and_keeps_preroll():
    sig = np.zeros(SAMPLE_RATE * 3, dtype=np.int16)
    onset = SAMPLE_RATE  # speech starts at 1.0 s
    sig[onset:onset + 4000] = 10000
    clip = run(sig)
    assert len(clip) == SAMPLE_RATE
    first_loud = int(np.argmax(clip > 0))
    assert 0.2 * SAMPLE_RATE <= first_loud <= 0.3 * SAMPLE_RATE  # ~0.25 s pre-roll


def test_quiet_input_never_triggers():
    sig = (np.random.default_rng(0).normal(0, 100, SAMPLE_RATE * 2)).astype(np.int16)
    assert run(sig) is None


def test_trigger_near_start_still_gives_full_clip():
    sig = np.zeros(SAMPLE_RATE * 2, dtype=np.int16)
    sig[BLOCK:BLOCK * 2] = 20000
    assert len(run(sig)) == SAMPLE_RATE


def test_next_index_and_wav(tmp_path):
    assert next_index(tmp_path, "ghost") == 1
    save_wav(tmp_path / "ghost-1.wav", np.zeros(10, dtype=np.int16))
    save_wav(tmp_path / "ghost-7.wav", np.zeros(10, dtype=np.int16))
    save_wav(tmp_path / "ghostly-9.wav", np.zeros(10, dtype=np.int16))
    assert next_index(tmp_path, "ghost") == 8


def test_speaker_numbering_is_independent(tmp_path):
    for n in ("ghost-1.wav", "ghost-2.wav", "ghost-wife-1.wav", "ghost-wife-4.wav"):
        save_wav(tmp_path / n, np.zeros(10, dtype=np.int16))
    assert next_index(tmp_path, "ghost") == 3
    assert next_index(tmp_path, "ghost", "wife") == 5
    assert next_index(tmp_path, "ghost", "kid") == 1
    assert clean_speaker("Mary-Ann!") == "maryann"
