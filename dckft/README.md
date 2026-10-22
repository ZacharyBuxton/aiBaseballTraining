# dckft: fusion engine, IMU reader, metrics, render

| Module | What it does |
|---|---|
| `run.py` | CLI: `capture`, `postprocess`, `process`, `live`, `simulate`, `validate`, `dump-config` |
| `config.py` | All tunables (bat geometry, IMU scaling, EKF noise, swing thresholds), JSON load/save |
| `imu/frames.py` | 30-byte frame decode/encode, sync-word resync parser, tx-clock unwrap |
| `imu/reader.py` | Serial/radio reader (Linux/Windows), ZMQ PUSH :5558, CSV, `--calibrate`, `--replay` |
| `data.py` | Load camera JSONL / IMU CSV; device-clock → host-clock mapping; bat line from collinear markers |
| `sync.py` | Cross-correlation time offset (bat-axis rate vs wrist rate) |
| `calib.py` | Per-swing self-calibration: IMU orientation, grip vector, gyro bias, residual time offset |
| `ekf.py` | Multiplicative quaternion EKF (gyro predict; camera-axis and gravity updates) + RTS backward smoother |
| `trajectory.py` | Flex correction, per-marker knob fixes, knob track (accel-aided Kalman/RTS), bat point velocities |
| `markers.py` | Per-marker 400 Hz tracks; camera frames' missing markers filled from the fusion |
| `swing.py` | Swing + impact detection (offline and live) |
| `metrics.py` | Bat/hand speed, attack angle, vertical bat angle, time to contact, swing plane, ... |
| `coaching.py` | Rule-based coaching notes; the LLM coach plugs in here later |
| `render.py` | matplotlib 3D replay window + PNG/MP4 |
| `pipeline.py` | `process_swing()`: everything above in order, with timings and diagnostics |
| `simulate.py` | Synthetic swing with ground truth in the exact on-disk formats |
| `replay.py` | Publish a recorded session over ZMQ as if live |
| `validate.py` | Monte-Carlo accuracy/latency report vs simulated truth |
| `quat.py` | Quaternion math, scalar-first `[w, x, y, z]` (no Euler angles anywhere) |

## Frames and conventions

- **World frame.** The ZED left rectified camera frame, OpenCV axes (x right, y down, z forward), in metres.
- **Up.** `up = -gravity`. Gravity comes from the ZED IMU (`.meta.json`); if that is missing, it comes
  from `gravity_cam` in the config.
- **Bat axis.** Unit vector pointing from the knob to the tip. Marker positions along the bat are
  given as distance from the knob (`bat.marker_s_m`).
- **IMU orientation.** `q` maps IMU-frame vectors into the world frame (`v_w = R(q) v_imu`).
- **Time.** Seconds on the Jetson host clock (`time.time_ns()/1e9`), shared by both streams.

## Outputs per swing

- `swing.json`: metrics, coaching notes, and diagnostics (sync offset, camera coverage, calibration
  residuals, EKF update counts, timings).
- `fused.csv`: per-sample output at 400 Hz: time from onset, quaternion, knob, axis, tip, sweet
  spot, sweet-spot speed, and attitude uncertainty.
- `markers.csv`: every marker at 400 Hz.
- `markers_camera.csv` / `camera_filled.jsonl`: every marker at every camera frame, each tagged
  `camera` (measured) or `imu_fill` (the camera missed it; position from the fusion).
- `swing.png` (and `swing.mp4` with `--mp4`).
