"""Utility helpers for accessing Intel RealSense cameras.

This module keeps the pyrealsense2 dependency optional. Importing the
module succeeds even if the library is not installed; a helpful error is
raised only when attempting to instantiate :class:`RealSenseStream`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional hardware dependency
    import pyrealsense2 as rs
except Exception:  # pragma: no cover
    rs = None  # type: ignore


@dataclass
class RealSenseConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    serial: Optional[str] = None
    use_depth: bool = False
    align_depth_to_color: bool = True
    timeout_ms: int = 10000  # Increased default timeout for more reliability
    color_only: bool = False


class RealSenseStream:
    """Thin wrapper around ``pyrealsense2.pipeline`` for color frames."""

    def __init__(self, **kwargs):
        if rs is None:  # pragma: no cover
            raise ImportError(
                "pyrealsense2 is not installed. Install via `pip install pyrealsense2` "
                "to enable Intel RealSense camera streaming."
            )

        cfg_data = RealSenseConfig()
        for key, value in kwargs.items():
            if hasattr(cfg_data, key) and value is not None:
                setattr(cfg_data, key, value)
        self.config = cfg_data

        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        if cfg_data.serial:
            self.rs_config.enable_device(cfg_data.serial)

        self.rs_config.enable_stream(
            rs.stream.color,
            cfg_data.width,
            cfg_data.height,
            rs.format.bgr8,
            cfg_data.fps,
        )

        self.align = None
        if cfg_data.use_depth and not cfg_data.color_only:
            self.rs_config.enable_stream(
                rs.stream.depth,
                cfg_data.width,
                cfg_data.height,
                rs.format.z16,
                cfg_data.fps,
            )
            if cfg_data.align_depth_to_color:
                self.align = rs.align(rs.stream.color)

        self.timeout_ms = max(1000, int(cfg_data.timeout_ms))
        self.profile = self.pipeline.start(self.rs_config)
        self.fps = cfg_data.fps
        
        # Warm up the camera - discard first few frames for stability
        print("🔄 Warming up RealSense camera...")
        for i in range(30):
            try:
                self.pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError:
                pass  # Ignore warmup timeouts
        print("✅ RealSense camera ready!")

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
        except RuntimeError as exc:  # pragma: no cover - hardware specific
            msg = str(exc)
            if "Frame didn't arrive" in msg:
                raise RuntimeError(
                    f"RealSense frame timeout after {self.timeout_ms} ms. "
                    "Ensure the camera is on USB3 and consider lowering --rs-width/height/fps or increasing --rs-timeout."
                ) from exc
            raise
        if self.align is not None:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        color_image = np.asanyarray(color_frame.get_data())
        return True, color_image

    def release(self):
        self.pipeline.stop()

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.release()
        except Exception:
            pass
