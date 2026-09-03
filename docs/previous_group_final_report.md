# Previous Team's Final Report — Summary

_Summarized from "Enhanced Baseball Training: Virtual Reality Simulator to Improve Traditional
Hitting Practice," Dmitrii Kapranov, Amiri Prescod, Michael Kokolis, Christopher Powers, and
Julian Frank, ECE 4972/4973 capstone report, Villanova University (project ran August–December
2025)._
_Full interview transcripts and personal details from the original report have been omitted here
for interviewee privacy — this summary retains only the technical content that the current
project inherits or builds on. The working code this report describes lives in the `legacy/`
subfolders of `../Cameras`, `../IMU`, and `../Data Processing`._

## What this project was

A predecessor project with a similar goal (fuse camera + IMU data into a 3D swing render) but a
different end target: displaying the result in a **VR/XR headset** rather than a 2D screen. The
current DC-KFT project inherited this team's hardware and code, but is not pursuing the VR
delivery format.

## Hardware inherited / referenced

- **GPU:** NVIDIA Jetson Orin Nano (used by the previous team; the current project has since
  moved to an NVIDIA THOR GPU per `docs/background.md`).
- **Cameras:** 2x Microsoft Azure Kinect (depth-sensing, 30 FPS each, interlaced for 60 FPS
  effective) — being replaced by a single stereo 120 fps camera per the September 2026 pivot.
- **IMU:** a custom sensor (ICM45686 + ADXL375 chipset) built by Dr. Jupina, connected via a
  wireless transmitter/receiver pair presenting as a serial (COM) port at 921600 baud, 30-byte
  frames. This is the same IMU electronics inherited by the current team; the pivot to a 900 MHz
  radio receiver (see `docs/background.md`) is a new telemetry link layered on top of/replacing
  this connection — exact integration TBD.
- **Trackable object:** a 3D-printed sphere with four high-contrast colors (green, blue, pink,
  yellow), tracked by the Kinect depth cameras — this hand/sphere-based tracking is being replaced
  by direct bat tracking in the current pivot.

## Software pipeline (five completed stages + one incomplete)

1. **Camera data collection** (`../Cameras/legacy/Device_Tracking_2_16.py`) — OpenCV + Azure Kinect,
   HSV color-blob detection for the four marker colors, 3D position via the Kinect depth
   calibration, published over a ZeroMQ PUB/SUB socket as JSON.
2. **IMU data collection** (`../IMU/legacy/imu_data_collection.py`) — asyncio + pyserial-asyncio
   reading fixed-size frames from the IMU receiver, validated by a status word, converted to
   accel/gyro physical units via hard-coded scale factors, written to CSV.
3. **Quaternion pose calculation** (`../Data Processing/legacy/Quaternion_Scheme_12_2.py`) — Horn's method
   (via the Davenport q-method: largest eigenvector of a 4×4 symmetric cross-covariance matrix)
   solving for the optimal quaternion, rotation matrix, and translation vector from ≥3 visible
   markers per frame, with per-point and RMS reprojection error reported. Output logged as JSONL.
4. **Angular velocity computation** (`../Data Processing/legacy/angular_velo_computation.py`) — Piecewise
   Cubic Hermite Interpolating Polynomial (PCHIP) interpolation of the quaternion stream, taking
   its time derivative, and converting to angular velocity via a standard quaternion-kinematics
   W-matrix.
5. **Cross-correlation alignment** (`../Data Processing/legacy/cross_correlation.m`, MATLAB) — aligns the
   camera-derived and IMU-derived angular velocity magnitude signals in time via cross-correlation,
   then reports the average percent difference between aligned values (their target: <5%).
6. **Swing rendering via Kalman filter** — **not completed** by the previous team; identified as
   the natural next step once the above five stages produce synchronized, aligned data. This
   remains open work for the current project's `Data Processing` and `Rendering` subsystems.

## Test plan & pass criteria (as designed by the previous team)

- **Camera test:** ≥55 FPS sustained, 3D position within 5 cm of a known physical location.
- **Data test:** wireless IMU collection working; <5% average percent difference between aligned
  camera/IMU angular velocity values.
- **Physical setup test:** reliable tracking at ≥10 ft from the camera.
- **PTP standard test:** IEEE 1588 Precision Time Protocol offset ≤1 ms for 95% of samples over a
  60-second window (camera/GPU time sync).
- **Weight test:** tracker + IMU casing under 5 oz combined, so as not to affect swing feel.

## Achievements & lessons learned (relevant to the current project)

- Camera tracking worked but reliable range topped out around 7 ft (vs. a 10 ft target) —
  attributed largely to lighting sensitivity of color-based detection and reflections off the
  painted tracking sphere. **Relevant to the pivot:** since the new design tracks the bat directly
  rather than a painted sphere, lighting/reflection sensitivity should be re-evaluated for
  whatever marking or detection approach is chosen.
- GPU bandwidth/processing power (Jetson Orin Nano) was a major bottleneck and schedule risk —
  running the Azure Kinect depth-sensing library exceeded its capacity, delaying the whole
  pipeline. The move to a stronger GPU (NVIDIA THOR, per the current proposal) is a direct
  response to this.
- The cross-correlation program only reached ~25% completion and the Kalman-filter rendering
  stage was never started — both remain open work.
- Recommended next steps (from the previous team): consider tracking algorithms less sensitive to
  lighting than color-blob detection; use a higher-framerate depth camera if available; make IMU
  data collection event-triggered (detect the start of a swing automatically) rather than
  terminating after a fixed sample count.

## Key references

The full report cites 23 sources; most relevant to ongoing work: Sabatini (2006) on
quaternion-based EKF for orientation from inertial + magnetic sensing (drift correction, magnetic
interference rejection); Horn (1987) closed-form absolute orientation via unit quaternions (basis
for the Horn/Davenport pose solver above); Diebel (2006) on representing attitude via quaternions
(basis for the angular-velocity-from-quaternion-derivative computation). Consult the original
report PDF (not included in this repository) for the complete reference list.
