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

import colorsys
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Tape color the camera looks for. Neon pink tracked best under cage lighting (Nov 2026), so it
# is now the ONLY default color: every band/stripe on the bat is the same pink, and bands are told
# apart by their order along the bat (see band_labeler.py), not by color.
MARKER_HEX: Dict[str, str] = {
    "neon_pink":   "#FF256B",   # Neon Pink
}

# The earlier multi-color scheme, kept so it can be switched back on through the config:
# set markers.layout = "colors" and markers.hues = {id: [hue]} for each color.
ALT_MARKER_HEX: Dict[str, str] = {
    "neon_yellow": "#D6E03D",   # Yellow-green
    "magenta":     "#C9218F",   # Magenta
    "neon_pink":   "#FF256B",   # Neon Pink
}

# Marker layouts (MarkerConfig.layout):
#   "bands"  - n_bands separate pink bands along the bat. Published as pink_1 (nearest the knob),
#              pink_2, ... Use UNEQUAL spacing (e.g. 20 / 40 / 70 cm) so the order can't flip.
#   "stripe" - one continuous pink stripe; its two ends are published as stripe_start (knob side)
#              and stripe_end.
#   "colors" - legacy: one blob per color, published under the color name.
LAYOUTS = ("bands", "stripe", "colors")


def hex_to_hue(hex_color: str) -> int:
    """'#RRGGBB' -> OpenCV hue (0-179)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return int(round(colorsys.rgb_to_hsv(r, g, b)[0] * 180.0)) % 180


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
    # OpenCV HSV hue centers (0-179) per color id, derived from MARKER_HEX above (pink ~170).
    # With pink alone there is no neighbouring color to collide with, so hue_tol is wider than
    # the old multi-color value of 4 (magenta sat only ~10 hue units away). Re-tune under cage lighting.
    hues: Dict[str, List[int]] = field(
        default_factory=lambda: {m: [hex_to_hue(hx)] for m, hx in MARKER_HEX.items()})
    layout: str = "bands"               # bands | stripe | colors  (see LAYOUTS above)
    color_id: str = "neon_pink"         # which hues entry the bands / stripe use
    n_bands: int = 3                    # "bands": how many pink bands are on the bat
    # Optional: band centres from the knob (m), same values as dckft's bat.marker_s_m. When set,
    # the spacing pattern helps tell the bands apart; it is required for nothing.
    band_s_m: Optional[List[float]] = None
    label_gate_m: float = 0.25          # max jump (m) per frame for a band to keep its label
    hue_tol: int = 7
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
    if cfg.markers.layout not in LAYOUTS:
        raise ValueError(f"markers.layout must be one of {LAYOUTS}, got {cfg.markers.layout!r}")
    return cfg


def save_config(cfg: Config, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
