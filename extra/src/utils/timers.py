"""
timers.py
---------
Simple timing utilities for FPS and profiling.
"""

import time


class FPSTimer:
    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.last_time = time.time()
        self.fps = 0.0

    def update(self):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            return self.fps
        inst_fps = 1.0 / dt
        self.fps = self.smoothing * self.fps + (1 - self.smoothing) * inst_fps
        self.last_time = now
        return self.fps


class Timer:
    """General-purpose timer."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_t = time.time()

    def elapsed(self):
        return time.time() - self.start_t
