"""Triggered, pre-rolled 1 second recording. Pure numpy: no audio hardware needed."""
from collections import deque

import numpy as np

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
PREROLL_SECONDS = 0.25  # audio kept from *before* the trigger so onsets aren't clipped


class TriggeredRecorder:
    """Feed mono int16 blocks; returns a full clip once a block peaks above threshold.

    The clip is the buffered pre-roll followed by live audio, exactly
    `clip_samples` long, so the trigger block sits ~PREROLL_SECONDS in.
    """

    def __init__(self, threshold, sample_rate=SAMPLE_RATE,
                 clip_seconds=CLIP_SECONDS, preroll_seconds=PREROLL_SECONDS):
        self.threshold = threshold  # int16 amplitude, 0..32767
        self.clip_samples = int(sample_rate * clip_seconds)
        self.preroll_samples = int(sample_rate * preroll_seconds)
        self.reset()

    def reset(self):
        self._pre = deque()
        self._pre_len = 0
        self._clip = None  # list of blocks once triggered
        self._clip_len = 0

    @property
    def triggered(self):
        return self._clip is not None

    def feed(self, block):
        """Returns the finished clip (int16 array) or None."""
        block = np.asarray(block, dtype=np.int16).reshape(-1)
        if self._clip is None:
            if len(block) and int(np.abs(block.astype(np.int32)).max()) >= self.threshold:
                self._clip = list(self._pre)
                self._clip_len = self._pre_len
            else:
                self._pre.append(block)
                self._pre_len += len(block)
                while self._pre_len - len(self._pre[0]) >= self.preroll_samples:
                    self._pre_len -= len(self._pre.popleft())
                return None
        self._clip.append(block)
        self._clip_len += len(block)
        if self._clip_len < self.clip_samples:
            return None
        # Trim the surplus from the end so the pre-roll (and the onset) is kept.
        clip = np.concatenate(self._clip)[:self.clip_samples]
        self.reset()
        return clip
