"""
config.py

Configuration for the ZED X optical capture subsystem.

All tunables live here so capture scripts don't carry hard-coded constants.
Defaults are chosen for the ZED X 4mm + ZED Link Duo setup:

  * SVGA (960x600) is the only ZED X mode that runs at 120 fps.
  * depth_mode "NONE" skips the SDK's neural depth entirely; bat markers are
    triangulated directly from the rectified left/right images (see stereo.py),
    which is cheap enough for 120 fps on an Orin Nano.
  * Exposure is manual and short, because a global shutter removes skew but not
    motion blur. A bat tip at ~35 m/s moves ~1.75 cm during a 0.5 ms exposure.

Configs can be saved/loaded as JSON so a tuned setup (exposure, marker hues)
can be committed and reused:

    python track_bat.py --save-config my_setup.json
    python track_bat.py --config my_setup.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class CameraConfig:
    resolution: str = "SVGA"            # sl.RESOLUTION name: SVGA | HD1080 | HD1200
    fps: int = 120                      # 120 only valid at SVGA on ZED X
    depth_mode: str = "NONE"            # NONE | NEURAL_LIGHT | NEURAL | NEURAL_PLUS
    exposure_us: Optional[int] = 500    # manual exposure in microseconds; None = auto
    analog_gain_mdb: Optional[int] = None   # sensor gain in mdB (default DTS range 1000-16000)
    digital_gain: Optional[int] = None      # ISP gain factor (default DTS range 1-256)
    depth_min_m: float = 1.0            # 4mm lens minimum usable depth is ~1-1.5 m
    depth_max_m: float = 10.0
    serial_number: Optional[int] = None     # pick a specific camera if several are attached
    sdk_verbose: int = 0


@dataclass
class MarkerConfig:
    # OpenCV HSV hue centers (0-179) for each marker. Keys match the MODEL dict in
    # "Data Processing/legacy/Quaternion_Scheme_12_2.py" so its pose solver can
    # consume our payloads unchanged. Values come from the previous team's tuning
    # (pink wraps around 0/180, hence several centers). Re-tune under cage lighting.
    hues: Dict[str, List[int]] = field(default_factory=lambda: {
        "neon_pink":   [165, 175, 5, 15],
        "neon_green":  [55, 65, 75],
        "neon_yellow": [28, 25, 30],
        "neon_blue":   [105, 95, 115],
    })
    hue_tol: int = 10
    s_min: int = 150
    s_max: int = 255
    v_min: int = 120
    v_max: int = 255
    min_area_px: float = 12.0           # full-resolution pixels
    detection_scale: float = 0.5        # coarse search runs on a downscaled frame
    roi_radius_px: int = 120            # full-resolution search radius around the prediction
    refine_pad_px: int = 6              # padding for the full-resolution centroid refinement
    max_row_mismatch_px: float = 3.0    # rectified images: a marker should sit on the same row L/R


@dataclass
class OutputConfig:
    camera_id: str = "cam_0"
    # PUSH socket. Quaternion_Scheme_12_2.py connects a PULL socket to port 5557.
    zmq_bind: Optional[str] = "tcp://*:5557"
    jsonl_path: Optional[str] = None    # also append every payload to this file
    min_points_to_publish: int = 1


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    markers: MarkerConfig = field(default_factory=MarkerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Optional[str] = None) -> Config:
    """Load a Config from JSON. Missing keys keep their defaults."""
    cfg = Config()
    if not path:
        return cfg
    with open(path, "r") as f:
        data = json.load(f)
    for section_name, section in (("camera", cfg.camera),
                                  ("markers", cfg.markers),
                                  ("output", cfg.output)):
        for key, value in data.get(section_name, {}).items():
            if not hasattr(section, key):
                raise KeyError(f"Unknown config key: {section_name}.{key}")
            setattr(section, key, value)
    return cfg


def save_config(cfg: Config, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
