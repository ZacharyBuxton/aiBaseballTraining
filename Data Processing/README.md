# Data Processing (Fusion Engine)

## Status: math is reusable, integration will change

This is the core fusion logic — quaternion pose solving, angular velocity computation, and
time-alignment between the camera and IMU streams. Unlike the camera hardware, this math doesn't
depend on *which* camera or radio module is used, so it's the most likely legacy code to carry
forward into the new pivot largely intact. What will need rework is the plumbing around it: the
input source (a single stereo camera stream instead of two Kinects; the 900 MHz radio instead of
a serial IMU link) and the still-missing Extended Kalman Filter stage (see below).

## Pipeline (previous group's design — order matters, each stage consumes the last)

| Stage | Script | Input |
|---|---|---|
| Pose (quaternion) from tracked points | `legacy/Quaternion_Scheme_12_2.py` | Camera tracker output (`../Cameras/legacy/Device_Tracking_2_16.py`), via ZMQ |
| Angular velocity from quaternion stream | `legacy/angular_velo_computation.py` | `changing_outputs.jsonl` written by the pose step above |
| Time-align camera vs. IMU angular velocity | `legacy/cross_correlation.m` (MATLAB) | CSV outputs from the angular-velocity step and `../IMU/legacy/imu_data_collection.py` |

`legacy/Subscription_Ball_Test_11_4_2.py` is an earlier/alternate ZMQ subscriber + quaternion
calculator (a simpler position-vector-to-quaternion approach, not the Horn/Davenport method used
by `Quaternion_Scheme_12_2.py`) — kept here for comparison, not part of the main pipeline above.

`sample_data/` (in this folder) holds example JSON payloads: `frame_payload.json` and
`frame_dummy_payload.json` show the shape the optical tracker publishes; `quaternion_output.json`
shows the shape of the pose output.

## What's still missing (open work for the new pivot)

- **Extended Kalman Filter** over the quaternion state — never implemented by the previous team.
  This is the main new development target for this subsystem (gyro drift correction, magnetic
  interference rejection, per `../docs/background.md`).
- Everything here currently assumes hard-coded local file paths and a hard-coded Jetson IP
  address (`Quaternion_Scheme_12_2.py`) — will need to be made configurable.
- Validating the fused output hits the project's <5% tracking-error target.
