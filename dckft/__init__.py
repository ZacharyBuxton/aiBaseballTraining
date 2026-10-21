"""
dckft -- DC-KFT fusion, analysis and rendering package.

Pipeline (see README.md for the full picture):

    Cameras/track_bat.py  --ZMQ :5557 / JSONL-->  dckft.camera_io   (bat knob + axis, 120 Hz)
    dckft.imu.reader      --ZMQ :5558 / CSV  -->  dckft.imu_io      (wrist gyro/accel, 400 Hz)
                                                       |
                         dckft.sync      clock mapping + cross-correlation offset
                         dckft.calib     IMU-to-world alignment + grip axis (self-calibrating)
                         dckft.ekf       quaternion MEKF (gyro predict, gravity + camera-axis update)
                         dckft.trajectory knob spline + fused axis -> bat points at 400 Hz
                         dckft.swing     swing detection / contact estimate
                         dckft.metrics   bat speed, attack angle, time to contact, ...
                         dckft.coaching  rule-based notes (LLM hook for later)
                         dckft.render    matplotlib 3D replay window

Quaternions are always scalar-first [w, x, y, z] and Hamilton convention.
"""

__version__ = "1.0.0"
