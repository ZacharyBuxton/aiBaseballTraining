"""
zed_camera.py

Thin wrapper around the ZED SDK (pyzed) for the ZED X. It handles:

  * opening a live camera or replaying an SVO recording with the same code path
  * manual exposure/gain (live only; SVO replays keep whatever was recorded)
  * grabbing rectified left/right BGR frames plus the hardware image timestamp
  * optional SDK depth (XYZ point cloud) when depth_mode != NONE
  * SVO recording
  * reading calibration and a static gravity estimate from the camera's IMU

pyzed is not on PyPI. It is installed by the ZED SDK installer, or afterwards
with the get_python_api.py script in the SDK install folder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np

try:
    import pyzed.sl as sl
except ImportError as exc:  # pragma: no cover - depends on the Jetson install
    raise ImportError(
        "pyzed is not installed. Install the ZED SDK for your JetPack version, then run "
        "get_python_api.py from the SDK folder inside this project's venv."
    ) from exc

from config import CameraConfig
from stereo import StereoCalib


class CameraError(RuntimeError):
    pass


@dataclass
class Frame:
    frame_id: int
    ts_ns: int                    # sl.TIME_REFERENCE.IMAGE, nanoseconds
    left: np.ndarray              # BGR, rectified
    right: np.ndarray             # BGR, rectified
    xyz: Optional[np.ndarray]     # H x W x 4 float32 (meters) or None when depth is off


def _ok(code) -> bool:
    # SDK 5.x returns warnings as codes that compare below SUCCESS; treat those as OK.
    return code <= sl.ERROR_CODE.SUCCESS


class ZedCamera:
    def __init__(self, cfg: CameraConfig, svo_path: Optional[str] = None):
        self.cfg = cfg
        self.svo_path = svo_path
        self.cam = sl.Camera()
        self.runtime = sl.RuntimeParameters()
        self._left = sl.Mat()
        self._right = sl.Mat()
        self._xyz = sl.Mat()
        self._frame_id = 0
        self.depth_enabled = cfg.depth_mode.upper() != "NONE"
        self.recording = False

    # ---------------------------------------------------------------- lifecycle
    def open(self) -> None:
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self.cfg.resolution.upper())
        init.camera_fps = int(self.cfg.fps)
        init.depth_mode = getattr(sl.DEPTH_MODE, self.cfg.depth_mode.upper())
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        init.depth_minimum_distance = float(self.cfg.depth_min_m)
        init.depth_maximum_distance = float(self.cfg.depth_max_m)
        init.sdk_verbose = int(self.cfg.sdk_verbose)
        if self.svo_path:
            init.set_from_svo_file(self.svo_path)
            init.svo_real_time_mode = False     # process every recorded frame
        elif self.cfg.serial_number:
            init.set_from_serial_number(int(self.cfg.serial_number))

        err = self.cam.open(init)
        if not _ok(err):
            raise CameraError(f"Camera open failed: {err}. "
                              "Check the ZED Link driver, cabling, and that the camera was "
                              "plugged in before boot (or restart the ZED X daemon).")

        self.runtime.enable_depth = self.depth_enabled
        if not self.svo_path:
            self.apply_exposure()

        # The SDK may silently fall back to another mode; make that loud.
        info = self.cam.get_camera_information()
        real_fps = info.camera_configuration.fps
        res = info.camera_configuration.resolution
        if not self.svo_path and int(round(real_fps)) != int(self.cfg.fps):
            print(f"[WARN] Requested {self.cfg.fps} fps but camera reports {real_fps:.1f} fps "
                  f"at {res.width}x{res.height}. 120 fps requires SVGA on the ZED X.")

    def close(self) -> None:
        if self.recording:
            self.stop_recording()
        self.cam.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # ----------------------------------------------------------------- controls
    def apply_exposure(self) -> None:
        c = self.cfg
        if c.exposure_us is None and c.analog_gain_mdb is None and c.digital_gain is None:
            self.cam.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
            return
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
        for setting, value in ((sl.VIDEO_SETTINGS.EXPOSURE_TIME, c.exposure_us),
                               (sl.VIDEO_SETTINGS.ANALOG_GAIN, c.analog_gain_mdb),
                               (sl.VIDEO_SETTINGS.DIGITAL_GAIN, c.digital_gain)):
            if value is None:
                continue
            err = self.cam.set_camera_settings(setting, int(value))
            if not _ok(err):
                print(f"[WARN] Could not set {setting} = {value}: {err}")

    def read_settings(self) -> dict:
        out = {}
        for name in ("AEC_AGC", "EXPOSURE_TIME", "ANALOG_GAIN", "DIGITAL_GAIN"):
            err, val = self.cam.get_camera_settings(getattr(sl.VIDEO_SETTINGS, name))
            out[name] = val if _ok(err) else None
        return out

    # --------------------------------------------------------------------- info
    def info(self) -> dict:
        info = self.cam.get_camera_information()
        cc = info.camera_configuration
        return {
            "sdk_version": sl.Camera.get_sdk_version(),
            "model": str(info.camera_model),
            "serial_number": info.serial_number,
            "firmware": cc.firmware_version,
            "resolution": [cc.resolution.width, cc.resolution.height],
            "fps": cc.fps,
            "depth_mode": self.cfg.depth_mode.upper(),
            "source": self.svo_path or "live",
        }

    def calibration(self) -> StereoCalib:
        cc = self.cam.get_camera_information().camera_configuration
        cal = cc.calibration_parameters      # rectified parameters by default
        left = cal.left_cam
        return StereoCalib(fx=left.fx, fy=left.fy, cx=left.cx, cy=left.cy,
                           baseline_m=cal.get_camera_baseline(),   # meters (UNIT.METER)
                           width=cc.resolution.width, height=cc.resolution.height)

    def estimate_gravity(self, seconds: float = 1.0) -> Optional[dict]:
        """
        Average the camera's accelerometer while it sits still. Gives the gravity
        direction for leveling the world frame. Live only.

        Returns the mean acceleration in the SDK's IMU output frame plus the
        camera<->IMU transform from the SDK. Confirm the frame convention with a
        quick tilt test before relying on it in the fusion engine.
        """
        if self.svo_path:
            return None
        data = sl.SensorsData()
        samples, last_ts = [], -1
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            if _ok(self.cam.get_sensors_data(data, sl.TIME_REFERENCE.CURRENT)):
                imu = data.get_imu_data()
                ts = imu.timestamp.get_nanoseconds()
                if ts != last_ts:
                    last_ts = ts
                    # Pass a fresh list: pyzed's default argument is a shared mutable list.
                    samples.append(list(imu.get_linear_acceleration([0.0, 0.0, 0.0])))
            time.sleep(0.002)
        if not samples:
            return None
        acc = np.asarray(samples)
        cfg = self.cam.get_camera_information().sensors_configuration
        return {
            "accel_mean_mps2": acc.mean(axis=0).tolist(),
            "accel_std_mps2": acc.std(axis=0).tolist(),
            "n_samples": int(acc.shape[0]),
            "rate_hz": acc.shape[0] / seconds,
            "camera_imu_transform": np.array(cfg.camera_imu_transform.m).tolist(),
        }

    def current_fps(self) -> float:
        return self.cam.get_current_fps()

    def dropped_frames(self) -> int:
        return self.cam.get_frame_dropped_count()

    # ---------------------------------------------------------------- recording
    RECORD_CODECS = {
        "h265": "H265",                  # GPU/hardware encoder, ~1% of raw size
        "h265-lossless": "H265_LOSSLESS",  # GPU/hardware encoder, ~25% of raw size
        "lossless-cpu": "LOSSLESS",      # CPU PNG/ZSTD, ~42% of raw size, no encoder needed
    }

    def start_recording(self, path: str, codec: str = "h265") -> None:
        """
        Record to an .svo2 file. The Orin Nano has no hardware video encoder, so
        use codec="lossless-cpu" there and watch the dropped-frame count.
        """
        mode = getattr(sl.SVO_COMPRESSION_MODE, self.RECORD_CODECS[codec])
        err = self.cam.enable_recording(sl.RecordingParameters(path, mode))
        if not _ok(err):
            hint = ("" if codec == "lossless-cpu" else
                    " If this board has no hardware encoder (Orin Nano), use codec 'lossless-cpu'.")
            raise CameraError(f"Could not start recording to {path}: {err}.{hint}")
        self.recording = True

    def stop_recording(self) -> None:
        self.cam.disable_recording()
        self.recording = False

    # --------------------------------------------------------------------- grab
    def grab(self) -> Optional[Frame]:
        """
        Blocks until the next frame. Returns None at the end of an SVO file.
        Raises CameraError on a real failure.
        """
        err = self.cam.grab(self.runtime)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            return None
        if not _ok(err):
            raise CameraError(f"grab() failed: {err}")

        ts_ns = self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        self.cam.retrieve_image(self._left, sl.VIEW.LEFT)
        self.cam.retrieve_image(self._right, sl.VIEW.RIGHT)
        # Retrieved images are BGRA views into SDK memory; cvtColor makes owned BGR copies.
        left = cv.cvtColor(self._left.get_data(), cv.COLOR_BGRA2BGR)
        right = cv.cvtColor(self._right.get_data(), cv.COLOR_BGRA2BGR)

        xyz = None
        if self.depth_enabled:
            self.cam.retrieve_measure(self._xyz, sl.MEASURE.XYZ)
            xyz = self._xyz.get_data()

        self._frame_id += 1
        return Frame(self._frame_id, ts_ns, left, right, xyz)
