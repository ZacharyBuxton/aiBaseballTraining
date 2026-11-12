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
    # Neon pink tape only (Nov 2026). Each entry = one tracked point along the bat and its
    # distance from the END OF THE KNOB, metres. Names must match what Cameras/track_bat.py
    # publishes for the chosen layout:
    #   bands  (default): pink_1 (nearest the knob), pink_2, pink_3 -- use UNEQUAL spacing
    #   stripe:           {"stripe_start": <knob-side end>, "stripe_end": <barrel-side end>}
    # At least two points are needed: the camera must see the bat's direction, not just a spot.
    marker_s_m: Dict[str, float] = field(default_factory=lambda: {
        "pink_1": 0.20,         # TODO measure
        "pink_2": 0.40,         # TODO measure
        "pink_3": 0.70,         # TODO measure
    })
    length_m: float = 0.838            # 33 in
    sweet_spot_s_m: float = 0.686      # ~6 in from the end cap
    hands_s_m: float = 0.152           # hand-speed point: 6 in from the knob, as Blast measures it
    imu_s_m: float = 0.05              # wrist IMU position along the handle, from the knob (approx.)


@dataclass
class ImuConfig:
    port: Optional[str] = None          # None = auto-detect /dev/ttyUSB* / ttyACM* / COMx
    baud: int = 921600
    # "auto" picks per receiver: the RX node's 42-byte A5 5A protocol-v2 frames (needs the advisor's
    # serial_protocol.py in IMU/) or the original 30-byte v1 frames. Or force "rx_v2_42B" / "v1".
    frame_format: str = "auto"
    frame_size: int = 30                # informational; the format defines the real size
    sync_word: int = 4095               # status word in the last 2 bytes of every valid frame
    zmq_bind: Optional[str] = "tcp://*:5558"
    nominal_rate_hz: float = 400.0
    # Unit conversions inherited from the previous team's script (ICM45686 / ADXL375).
    accel_lsb: float = 0.009576806641   # m/s^2 per LSB (ICM45686, +-32 g)
    gyro_lsb: float = 0.002130528872    # rad/s per LSB (ICM45686, +-4000 dps)
    hg_lsb: float = 0.4788403320        # m/s^2 per LSB (ADXL375, +-200 g)
    gyro_saturation_rads: float = 69.0
    accel_saturation_ms2: float = 300.0
    hg_saturation_ms2: float = 1900.0   # ADXL375 +-200 g
    # ICM45686 ONLY (advisor, Nov 5): the ADXL375 channels are still decoded and logged when the
    # frame carries them, but nothing uses them unless use_adxl375 is turned back on. Impact
    # detection then runs on the ICM45686 accelerometer (see swing.contact_icm_ms2).
    use_adxl375: bool = False
    # Only with use_adxl375: swap in the ADXL375 where the ICM45686 accelerometer clips. Assumes
    # both sensors share a frame -- check on the bench first (tilt, compare accel_* vs accel_*2).
    use_highg_when_saturated: bool = True


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
    smooth: bool = True                 # RTS backward pass (recorded data is processed after the fact)


@dataclass
class SwingConfig:
    # Swing trigger. Default matches the advisor's ring-buffer example (Nov 12): |omega| of the
    # ICM45686 > trigger_rads, rising edge. Alternatives (Nov 5 notes): accel_mag | accel_xy |
    # accel_-y (wire side up, bottom hand) | accel_+x ... Tune with: python -m dckft.trigger_tune
    trigger_source: str = "gyro"
    trigger_accel_ms2: float = 30.0     # ~3 g beyond gravity; real swings reach 10-30 g at the hand
    onset_rads: float = 4.0             # |omega| crossing that marks swing start (searching back)
    trigger_rads: float = 10.0          # |omega| trigger (rad/s, ~570 dps), as in the example
    # Ring buffer (advisor example): frames kept before the trigger frame / from it onward,
    # and the largest device-time step allowed inside one window (bigger = missing frames).
    ring_pre_frames: int = 1200         # 3 s at 400 Hz
    ring_post_frames: int = 1200        # 3 s at 400 Hz
    ring_max_dt_us: float = 3750.0
    end_rads: float = 4.0
    min_gap_s: float = 1.5
    pre_roll_s: float = 1.0             # data kept before onset (stance -> calibration)
    post_roll_s: float = 0.4
    contact_hg_ms2: float = 300.0       # ADXL375 spike threshold for ball contact (~30 g)
    contact_icm_ms2: float = 120.0      # ICM45686 spike threshold (it clips at +-32 g = 313 m/s^2)
    impact_fallback_adxl: bool = True   # contact timing from the ADXL375 if the ICM spike is clipped
    end_hold_s: float = 0.10           # finish after sustained low wrist rate (hold the finish)
    end_requires_still_accel: bool = True   # ...and |a| within still_accel_ms2 of 1 g
    still_accel_ms2: float = 3.0
    max_duration_s: float = 1.2        # bound one swing; reset movement starts a new event
    max_camera_gap_s: float = 0.06     # longer optical gaps are NOT reliable speed samples
    reacquire_guard_s: float = 0.04    # exclude both edges of a long gap from speed metrics


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

