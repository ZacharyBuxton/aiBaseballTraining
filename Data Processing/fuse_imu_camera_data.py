"""
fuse_imu_camera_data.py

Fuses the optical (camera/marker) tracking data with the IMU data stream into
a single, time-aligned dataset.

Pipeline this script assumes (matches the rest of the repo):
    1. Quaternion_Scheme_12_2.py      -> logs camera pose per frame to
                                          changing_outputs.jsonl
                                          (fields: ts_ns_mono, quaternion_wxyz, ...)
    2. angular_velo_computation.py    -> reads changing_outputs.jsonl, differentiates
                                          the quaternion stream, and writes
                                          cam_ang_velo.csv (time, omega_x, omega_y, omega_z)
    3. imu_data_collection.py         -> logs the IMU stream to a CSV in ../reports/
                                          (fields include imu_meas_time, omega_x, omega_y, omega_z)
    4. THIS SCRIPT                    -> loads both CSVs, time-aligns them, and produces
                                          one fused CSV that both the IMU and camera streams
                                          can be evaluated against.

Why cross-correlation:
    The camera and IMU are not started at the same instant and don't share a clock,
    so there's an unknown constant time offset between "camera time" and "IMU time."
    Last year's team (Dmitrii Kapranov, cross_correlation.m) found this offset in
    MATLAB using alignsignals() on the two angular-velocity-magnitude signals. This
    script does the same thing in Python (scipy.signal.correlate), then actually
    builds the fused dataset instead of just reporting the offset.

Usage:
    python fuse_imu_camera_data.py --imu path/to/imu_data.csv --cam path/to/cam_ang_velo.csv --out fused_output.csv

Notes / TODO once real hardware data exists:
    - Column names below match the CSV headers currently written by
      imu_data_collection.py and angular_velo_computation.py. If those change,
      update IMU_COLUMNS / CAM_COLUMNS.
    - Sign conventions / axis mapping between the camera frame and IMU frame are
      NOT handled here yet -- that requires a physical calibration once both
      pieces of hardware are mounted on the same rigid body. Flagged below with
      a TODO so it isn't silently assumed to be correct.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags
from scipy.interpolate import PchipInterpolator


IMU_TIME_COL = "imu_meas_time"
IMU_OMEGA_COLS = ["omega_x", "omega_y", "omega_z"]
IMU_ACCEL_COLS = ["accel_x", "accel_y", "accel_z"]

CAM_TIME_COL = "time"
CAM_OMEGA_COLS = ["omega_x", "omega_y", "omega_z"]


def load_imu_csv(path):
    df = pd.read_csv(path)
    missing = [c for c in [IMU_TIME_COL, *IMU_OMEGA_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"IMU CSV is missing expected columns: {missing}")
    return df


def load_cam_csv(path):
    df = pd.read_csv(path)
    missing = [c for c in [CAM_TIME_COL, *CAM_OMEGA_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"Camera CSV is missing expected columns: {missing}")
    return df


def omega_magnitude(df, cols):
    return np.sqrt(np.sum(df[cols].to_numpy() ** 2, axis=1))


def find_time_offset(t_imu, mag_imu, t_cam, mag_cam):
    """
    Resample both magnitude signals onto a common uniform time grid, then
    cross-correlate to find the lag (in seconds) that best aligns the camera
    stream to the IMU stream. Positive offset means the camera stream needs to
    be shifted forward in time to line up with the IMU stream.
    """
    dt = min(np.median(np.diff(t_imu)), np.median(np.diff(t_cam)))
    t_start = max(t_imu.min(), t_cam.min())
    t_end = min(t_imu.max(), t_cam.max())
    grid = np.arange(t_start, t_end, dt)

    imu_grid = np.interp(grid, t_imu, mag_imu)
    cam_grid = np.interp(grid, t_cam, mag_cam)

    corr = correlate(imu_grid - imu_grid.mean(), cam_grid - cam_grid.mean(), mode="full")
    lags = correlation_lags(len(imu_grid), len(cam_grid), mode="full")
    best_lag_samples = lags[np.argmax(corr)]
    offset_seconds = best_lag_samples * dt
    return offset_seconds


def interpolate_camera_onto_imu(t_imu, cam_df, time_offset):
    """
    Shift the camera timeline by time_offset, then PCHIP-interpolate every
    camera column onto the IMU timestamps (same interpolation style already
    used in angular_velo_computation.py).
    """
    t_cam_shifted = cam_df[CAM_TIME_COL].to_numpy() + time_offset

    # Only interpolate within the overlapping time range; outside it, leave NaN
    # rather than extrapolate a physically meaningless value.
    valid = (t_imu >= t_cam_shifted.min()) & (t_imu <= t_cam_shifted.max())

    out = {}
    cam_value_cols = [c for c in cam_df.columns if c != CAM_TIME_COL]
    for col in cam_value_cols:
        itp = PchipInterpolator(t_cam_shifted, cam_df[col].to_numpy())
        vals = np.full_like(t_imu, np.nan, dtype=float)
        vals[valid] = itp(t_imu[valid])
        out[f"cam_{col}"] = vals
    return pd.DataFrame(out, index=range(len(t_imu)))


def fuse(imu_df, cam_df):
    t_imu = imu_df[IMU_TIME_COL].to_numpy()
    t_cam = cam_df[CAM_TIME_COL].to_numpy()

    mag_imu = omega_magnitude(imu_df, IMU_OMEGA_COLS)
    mag_cam = omega_magnitude(cam_df, CAM_OMEGA_COLS)

    offset = find_time_offset(t_imu, mag_imu, t_cam, mag_cam)
    print(f"Estimated camera->IMU time offset: {offset:.6f} s")

    cam_interp = interpolate_camera_onto_imu(t_imu, cam_df, offset)

    fused = imu_df.copy().reset_index(drop=True)
    fused = pd.concat([fused, cam_interp], axis=1)

    # Agreement check, same spirit as cross_correlation.m's percent-difference
    # calculation, but on the fused/interpolated rows so it's a real per-sample
    # comparison rather than a single aggregate number.
    if all(f"cam_{c}" in fused.columns for c in CAM_OMEGA_COLS):
        cam_mag_on_imu_grid = np.sqrt(
            sum(fused[f"cam_{c}"] ** 2 for c in CAM_OMEGA_COLS)
        )
        fused["omega_mag_imu"] = mag_imu
        fused["omega_mag_cam"] = cam_mag_on_imu_grid
        with np.errstate(divide="ignore", invalid="ignore"):
            fused["omega_pct_diff"] = (
                np.abs(fused["omega_mag_imu"] - fused["omega_mag_cam"])
                / fused["omega_mag_imu"]
            )
        valid_pct = fused["omega_pct_diff"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_pct):
            print(f"Mean angular-velocity %% diff over overlap: {valid_pct.mean()*100:.2f}%")

    return fused, offset


def main():
    parser = argparse.ArgumentParser(description="Fuse IMU and camera tracking data.")
    parser.add_argument("--imu", required=True, help="Path to IMU CSV (imu_data_collection.py output)")
    parser.add_argument("--cam", required=True, help="Path to camera angular velocity CSV (angular_velo_computation.py output)")
    parser.add_argument("--out", default="fused_output.csv", help="Path to write the fused CSV")
    args = parser.parse_args()

    imu_df = load_imu_csv(args.imu)
    cam_df = load_cam_csv(args.cam)

    fused, offset = fuse(imu_df, cam_df)
    fused.to_csv(args.out, index=False)
    print(f"Fused data written to: {args.out}")


if __name__ == "__main__":
    main()
