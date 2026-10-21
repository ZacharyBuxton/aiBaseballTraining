"""
config.py -- all fusion / analysis tunables in one place, JSON save/load.

    python -m dckft.run process ... --config my_session.json
    python -m dckft.run dump-config > dckft_default.json

!! MEASURE THESE BEFORE THE FINAL DEMO !!
  * BatConfig.marker_s_m  -- distance (m) of each tape marker's centre from the end of the knob.
  * BatConfig.length_m / sweet_spot_s_m.
The defaults are placeholders for a 33" bat.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

G = 9.80665


@dataclass
class BatConfig:
    # Markers are collinear along the bat (current mounting). Distance from the knob end, metres.
    marker_s_m: Dict[str, float] = field(default_factory=lambda: {
        "neon_pink": 0.20,      # TODO measure
        "magenta": 0.45,        # TODO measure
        "neon_yellow": 0.70,    # TODO measure
    })
    length_m: float = 0.838            # 33 in
    sweet_spot_s_m: float = 0.686      # ~6 in from the end cap
    hands_s_m: float = 0.08            # where the hands sit (top hand), from the knob
    imu_s_m: float = 0.05              # wrist IMU position along the handle, from the knob (approx.)


@dataclass
class ImuConfig:
    port: Optional[str] = None          # None = auto-detect /dev/ttyUSB* / ttyACM* / COMx
    baud: int = 921600
    frame_size: int = 30
    sync_word: int = 4095               # status word in the last 2 bytes of every valid frame
    zmq_bind: Optional[str] = "tcp://*:5558"
    nominal_rate_hz: float = 400.0
    # Unit conversions inherited from the previous team's script (ICM45686 / ADXL375).
    accel_lsb: float = 0.009576806641   # m/s^2 per LSB (ICM45686, +-32 g)
    gyro_lsb: float = 0.002130528872    # rad/s per LSB (ICM45686, +-4000 dps)
    hg_lsb: float = 0.4788403320        # m/s^2 per LSB (ADXL375, +-200 g)
    gyro_saturation_rads: float = 69.0
    accel_saturation_ms2: float = 300.0


@dataclass
class SyncConfig:
    # Both streams are stamped with time.time_ns() on the Jetson, so the residual offset is
    # just radio + USB latency. Cross-correlation refines it inside +-max_offset_s.
    use_xcorr: bool = True
    max_offset_s: float = 0.15
    fixed_offset_s: float = 0.0         # added to IMU times if use_xcorr is False
    refine_max_s: float = 0.015         # joint calibration refines the offset by up to this (0 = off)


@dataclass
class EkfConfig:
    gyro_noise: float = 0.02            # rad/s / sqrt(Hz) (ICM45686 ~0.004 dps/rtHz + margin)
    gyro_bias_walk: float = 0.002       # rad/s^2 / sqrt(Hz)
    grip_flex_rad: float = 0.12         # camera-axis measurement std (wrist-to-bat isn't rigid)
    camera_axis_rad: float = 0.03       # extra noise from marker triangulation
    gravity_std_ms2: float = 0.6
    gravity_gate_ms2: float = 1.0       # use accel as gravity only if | |f| - g | < gate
    gravity_gate_rads: float = 1.5      # ... and |omega| below this
    calib_window_s: float = 0.6         # data used for the self-calibration before swing onset
    init_bias_from_stance: bool = True


@dataclass
class SwingConfig:
    onset_rads: float = 4.0             # |omega| crossing that marks swing start (searching back)
    trigger_rads: float = 10.0          # |omega| that confirms a swing (~570 dps; real swings 1500+ dps)
    end_rads: float = 4.0
    min_gap_s: float = 1.5
    pre_roll_s: float = 1.0             # data kept before onset (stance -> calibration)
    post_roll_s: float = 0.4
    contact_hg_ms2: float = 300.0       # ADXL375 spike threshold for ball contact (~30 g)


@dataclass
class RenderConfig:
    show: bool = True
    save_png: bool = True
    save_mp4: bool = False
    playback_speed: float = 0.1         # 0.1 = 10x slow motion


@dataclass
class FusionConfig:
    bat: BatConfig = field(default_factory=BatConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    ekf: EkfConfig = field(default_factory=EkfConfig)
    swing: SwingConfig = field(default_factory=SwingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    # Gravity in the camera (world) frame, m/s^2. OpenCV axes: y points down, so a level camera
    # sees gravity as +y. Overridden by the ZED IMU estimate in <session>.jsonl.meta.json if present.
    gravity_cam: List[float] = field(default_factory=lambda: [0.0, G, 0.0])
    # Knob trajectory: "kalman" = camera + wrist-accel RTS smoother (bridges occlusion),
    # "spline" = camera-only smoothing spline with smoothing weight knob_spline_lam.
    knob_track: str = "kalman"
    knob_spline_lam: Optional[float] = 3e-7
    # Wrist-flex complementary correction smoothing (0/None disables; larger = smoother).
    flex_lam: Optional[float] = 2e-6
    camera_zmq: str = "tcp://127.0.0.1:5557"
    imu_zmq: str = "tcp://127.0.0.1:5558"

    def to_dict(self) -> dict:
        return asdict(self)


_SECTIONS = ("bat", "imu", "sync", "ekf", "swing", "render")


def load_config(path: Optional[str] = None) -> FusionConfig:
    cfg = FusionConfig()
    if not path:
        return cfg
    with open(path) as f:
        data = json.load(f)
    for key, value in data.items():
        if key in _SECTIONS:
            section = getattr(cfg, key)
            for k, v in value.items():
                if not hasattr(section, k):
                    raise KeyError(f"Unknown config key: {key}.{k}")
                setattr(section, k, v)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise KeyError(f"Unknown config key: {key}")
    return cfg


def save_config(cfg: FusionConfig, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
