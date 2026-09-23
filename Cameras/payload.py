"""
payload.py

Builds and ships per-frame tracking payloads to the fusion engine.

The payload keeps the shape of
"Data Processing/legacy/sample_data/frame_payload.json", so
Quaternion_Scheme_12_2.py can consume it unchanged. Extra keys are additive.

Timestamps (all integer nanoseconds):
  ts_ns_image : camera capture time from the ZED SDK (sl.TIME_REFERENCE.IMAGE),
                on the Jetson's system clock. Use this one for camera/IMU fusion.
  ts_ns_mono  : same value as ts_ns_image. Kept because the legacy
                angular_velo_computation.py reads this field name.
  ts_ns_wall  : host time.time_ns() when the payload was built. The difference
                from ts_ns_image is the capture-to-publish latency.

Transport: a ZeroMQ PUSH socket that binds (default tcp://*:5557). The legacy
pose solver connects a PULL socket to that port. Sends are non-blocking; if no
consumer is attached, frames are dropped and counted instead of stalling capture.
"""

from __future__ import annotations

import json
import time
from typing import Dict, Optional

SCHEMA_VERSION = "1.1"
COORD_SYS = "zedx_left_rect_opencv_meters_v1"   # x right, y down, z forward


def build_frame_payload(camera_id: str, frame_id: int, ts_ns_image: int,
                        points: Dict[str, dict], extra: Optional[dict] = None) -> dict:
    """
    points: {marker_id: {"xyz": (x, y, z), ...optional per-point fields}}
    """
    pts = []
    for marker_id, p in points.items():
        x, y, z = p["xyz"]
        entry = {"id": marker_id, "pos": {"x": float(x), "y": float(y), "z": float(z)}}
        for k, v in p.items():
            if k != "xyz":
                entry[k] = v
        pts.append(entry)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "coord_sys": COORD_SYS,
        "camera_id": camera_id,
        "frame_id": int(frame_id),
        "ts_ns_image": int(ts_ns_image),
        "ts_ns_mono": int(ts_ns_image),
        "ts_ns_wall": time.time_ns(),
        "points": pts,
    }
    if extra:
        payload.update(extra)
    return payload


class FramePublisher:
    def __init__(self, bind_addr: str, send_hwm: int = 240):
        import zmq  # imported here so the rest of the module works without pyzmq
        self._zmq = zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, send_hwm)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(bind_addr)
        self.sent = 0
        self.dropped = 0

    def publish(self, payload: dict) -> bool:
        try:
            self.sock.send_string(json.dumps(payload), flags=self._zmq.NOBLOCK)
            self.sent += 1
            return True
        except self._zmq.Again:
            self.dropped += 1
            return False

    def close(self) -> None:
        self.sock.close()


class JsonlWriter:
    def __init__(self, path: str):
        self.f = open(path, "a", buffering=1)

    def write(self, obj: dict) -> None:
        self.f.write(json.dumps(obj) + "\n")

    def close(self) -> None:
        self.f.close()
