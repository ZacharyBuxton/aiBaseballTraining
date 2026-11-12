# Demo-Day Runbook

Everything runs on the Jetson. Run the commands from the repo root with the venv active.

## Swing recorder app (Nov 12): the easy way

```bash
python3 run_app.py                     # on the Orin; add --port /dev/ttyUSB0 --cam-config Cameras/<file>.json if needed
```

Open `http://<orin-ip>:8000` in any browser on the same network (`hostname -I` prints the IP), or
`http://localhost:8000` on the Orin. The page has four buttons:

- **Start recording**: starts the IMU reader and the camera tracker. Every swing is caught by the ring buffer,
  processed, rendered and added to the list on its own. Press **Stop** when done.
- **Simulate swing**: runs one simulated swing through the same ring buffer and pipeline. Use it to test with no hardware.
- **Update from GitHub**: runs `git pull` and restarts the app. The app also pulls once on start (`--no-pull` to skip).
- Click any swing in the list to see its render, attack angle, vertical bat angle, bat speed and the raw files.

Swings are stored in `app_data/swings/<id>/`, which git ignores. The app uses `run_schema_config.json`, the measured
bat geometry written by `run_schema.py`, when that file exists.

### Ring buffer trigger (from the advisor's Inertial_Sensors_Capture.py)

`dckft/ring.py` keeps the last 1200 IMU frames (3 s at 400 Hz). When |ω| rises above 10 rad/s, it saves those 1200
frames plus the next 1200, so each swing is a 6 s window. These rules match the example:

- the pre-roll must be full before a trigger
- the gyro must drop below the threshold once before the next trigger (no re-trigger while held high)
- a gap longer than 3.75 ms resets the buffer, so frames are never stitched across a dropout
- duplicate and out-of-order frames are ignored

Settings are `swing.trigger_source`, `swing.trigger_rads`, `swing.ring_pre_frames`, `swing.ring_post_frames` and
`swing.ring_max_dt_us`. `run_stream.py` and `python -m dckft.run live` use the same ring.

### Protocol-v2 receiver frames (42 bytes, A5 5A)

The receiver in the example sends 42-byte frames: header, 30-byte payload, 8 diagnostic bytes, then CRC16. The reader
detects them on its own (`imu.frame_format: "auto"`) and checks the CRC. To unpack the payload, it needs the advisor's
**`serial_protocol.py`**. Copy that file into the repo's `IMU/` folder, or set `DCKFT_SERIAL_PROTOCOL=/path/to/its/folder`.
Without it, the reader stops and says so. The old frame format still works with no changes.

## 0. One-time setup (per Jetson)

```bash
pip install -r requirements.txt            # plus pyzed from the ZED SDK (Cameras/README.md)
sudo usermod -aG dialout $USER             # serial access for the radio receiver; log out/in
python -m dckft.run dump-config > dckft_config.json
```

Tape the bat with **neon pink only**: 3 bands at unequal distances from the knob (e.g. 20 / 40 / 70 cm).

Edit `dckft_config.json`:
- `bat.marker_s_m`: measured band centres from the knob in metres. Use `pink_1` for the band nearest the
  knob, then `pink_2` and `pink_3`.
- `bat.length_m` and `bat.sweet_spot_s_m`.

Pass the same file to the camera tracker with `--bat-config dckft_config.json`, so it can tell the bands apart by spacing.

## 1. Bring-up checks (5 min)

```bash
python Cameras/bringup_check.py                  # must report ~120 fps, no timestamp gaps
python -m dckft.imu.reader --list-ports          # find the radio receiver (e.g. /dev/ttyUSB0)
python -m dckft.imu.reader --port /dev/ttyUSB0 --seconds 5 --no-zmq --csv /tmp/imu_check.csv
```

The IMU check passes when:
- the rate is about 400 Hz
- `resyncs` stays near 0
- the CSV shows |accel| ≈ 9.8 while the IMU sits still

## 2a. Recommended: record now, process after (most accurate)

```bash
python -m dckft.run --config dckft_config.json capture --port /dev/ttyUSB0 --cam-config Cameras/cage.json   # Orin Nano: add --codec lossless-cpu
```

- While it runs, the camera records every frame with no marker detection, and the IMU logs every sample.
- Press Ctrl+C when done (or pass `--seconds 60`).
- It then detects markers in every recorded frame and fuses every swing into
  `sessions/capture_*/out/swing_NN/`.
- To re-run the processing later (e.g. after re-tuning hues):

```bash
python -m dckft.run postprocess --session sessions/capture_YYYYmmdd_HHMMSS --redetect
```

## 2b. Live feedback: three processes (three terminals)

```bash
# T1: camera tracker (publishes on :5557)
python Cameras/track_bat.py --config Cameras/cage.json --bat-config dckft_config.json --jsonl sessions/cam_$(date +%H%M).jsonl

# T2: IMU reader (publishes on :5558)
python -m dckft.imu.reader --port /dev/ttyUSB0 --csv sessions/imu_$(date +%H%M).csv

# T3: fusion + render: start it before the first swing
python -m dckft.run live --config dckft_config.json --session-dir sessions
```

## 3. Each swing

1. The hitter holds still in the stance for about 1 second. That gives the gravity reference for calibration.
2. The hitter swings.
3. About 1–2 s later, the 3D replay window opens and the terminal prints the metrics.

Each swing is saved to `sessions/session_*/swing_NN/`:
- `swing.json`: metrics, coaching, diagnostics
- `fused.csv`: fused bat pose at 400 Hz
- `swing.png`

Raw camera and IMU data for the whole session are saved alongside, so any swing can be reprocessed:

```bash
python -m dckft.run process --cam sessions/session_X/camera.jsonl --imu sessions/session_X/imu.csv --all
```

## 4. If something goes wrong

| Symptom | Fix |
|---|---|
| `only N camera frames in the window` | `track_bat.py` isn't running/publishing, or markers aren't detected (check `--preview`) |
| No swing detected | Swing faster, or lower `swing.trigger_rads` in the config (default 10 rad/s) |
| `Not enough camera bat-axis samples` | Fewer than 2 pink bands visible; re-tune the pink hue or improve lighting |
| `Need >= 2 configured pink points` | The band names in `bat.marker_s_m` don't match the camera's layout (`pink_1..N` or `stripe_start/stripe_end`) |
| Bands swapped (bat looks reversed) | Space the bands unequally, pass `--bat-config`, and hold still in the stance at the start |
| `calib_rms_axis_deg` > 8 in `swing.json` | Markers mis-detected or grip slipped; re-check marker distances in config |
| Live is slow / dropped frames warning | Record first (`track_bat.py --record x.svo2`), then process offline |
| No hardware at all | `python -m dckft.run simulate` or replay a saved session: `python -m dckft.replay --cam ... --imu ...` with `dckft.run live` running |

## 5. Rehearse without hardware

```bash
python -m dckft.run live --session-dir /tmp/live &
python -m dckft.replay --cam sessions/sim/camera.jsonl --imu sessions/sim/imu.csv --loops 3
```



## Completed swings and tracking quality (November 2026)

The raw SVO and IMU CSV remain the complete recording. Analysis retains the stance for
calibration, then stops at the IMU-detected swing end. A sustained 100 ms low-rate period
ends an event; return-to-stance movement during the 1.5 s cooldown is ignored instead of
being merged into the finished swing. `swing.max_duration_s` bounds an active event (default
1.2 s); adjust it for deliberately slow drills. This is not a ball-contact trigger.

Translation smoothing restarts across optical gaps longer than `swing.max_camera_gap_s`
(default 60 ms), so reacquisition cannot pull an earlier segment through the backward pass.
Speed samples inside those gaps and within `swing.reacquire_guard_s` (40 ms) of the edges
are excluded. The replay draws unsupported poses as orange IMU/model estimates and breaks
the trusted speed curve at those times. It stops at swing end and no longer displays the post-swing hold, coaching
panel, extra metrics, or fitted plane.

`fused.csv` retains raw model estimates with `speed_valid` and `in_swing` columns;
`markers.csv` has `pose_supported`. Marker exports use `source: unsupported` for missing
camera observations with no reliable optical support, rather than calling them validated
IMU fills. `swing.json` includes `speed_reliable`, `speed_note`, and
`speed_valid_fraction`. If optical support is missing near the wrist peak or selected
contact, speed/related angles are unavailable (`null` in JSON), not a capped or guessed
number. Supported results are still model estimates, not a speed accuracy guarantee.

A wrist sensor cannot directly measure bat orientation during an arbitrary wrist hinge.
Check calibration and synchronization diagnostics and compare against recorded video or
an independent speed reference. Do not tune a speed cap to obtain a desired mph value.

Reprocess an existing launcher session with `python3 run_schema.py --session sessions/schema_...`.
Each attempt gets a new analysis directory; old results and raw recordings are kept.
For a new swing use the same capture command and measured config as before. The camera's
new default saturation threshold is 120 (the right camera's pink tape was being fragmented
at 150); existing saved camera configs override that default.

### Continuous IMU-assisted replay

After successful camera/IMU calibration, replay keeps a pose at every available IMU sample
through swing end. Pink shows camera-constrained poses. Orange dashed bat/path sections show
estimated motion while the camera is hidden. The cleaned camera marker positions and
measured band distances define each available bat line. These camera poses anchor the
display at their exact timestamps. Short optical intervals use linear position interpolation
and shortest-arc axis interpolation. Across actual gaps, the fused gyro supplies bat-axis
prediction and corrected wrist acceleration supplies translation prediction. For gaps
with camera support on both sides, a smooth cubic endpoint correction constrains predicted
position and velocity to the next supported segment. This uses future observations offline.
Unbounded edges dead-reckon from the nearest supported pose; unusable acceleration coasts.
The interpolation/rigid-grip assumptions and wrist-to-bat flex remain limitations, particularly
for long occlusions. No camera frames or missing IMU measurements are recovered by this step.

`replay.csv` saves this continuous display trajectory, all configured marker positions,
`pose_source` (`camera_supported_fusion` or `imu_estimate`), `nearest_camera_support_s`,
`imu_accel_used`, `speed_valid`, and `in_swing`. Its timestamps are the received IMU times,
not invented camera frame times. `swing.json` includes a replay estimate count and method.
The conservative `fused.csv`, marker exports, and headline metrics retain their support flags;
filling a display gap does not validate a speed. Long gaps may still make speed unavailable.
If there is no supported camera anchor, the continuous replay cannot be established.

Pose support and speed validity are separate. The 40 ms reacquisition guard can withhold
speed without replacing a camera-constrained display pose. An isolated camera bat line
anchors the gap predictor even when the speed mask is invalid. The render's distance to
camera support now means distance to an actual usable camera bat-line timestamp.
`replay.csv` uses `pose_source` for pose provenance and `speed_valid` for metric eligibility.
`swing.json` reports both active-swing fractions, original fused-marker residuals, replay
marker residuals against the optical measurements, and camera/analysis time ranges. Residuals
measure agreement with camera observations, not independent ground-truth accuracy.

For an end-on view, an end-cap patch can provide a visible point but cannot determine the bat
axis alone. The current three-band tracker selects three pink blobs; do not add a fourth
pink spot and assume it has an end-cap identity. A cap needs a separately configured/detected
marker identity (preferably a distinct color) before it can be used without band ambiguity.
Keep the measured band centers and black separators unchanged.

## Always-on capture with an IMU trigger (advisor meeting, Nov 5)

The goal is to always be recording instead of taking fixed clips. The camera and the IMU stream continuously, and the
IMU marks each swing.

### Setup

- **IMU:** on the **bottom hand**, **wire side up**. Only the ICM45686 is used; the ADXL375 is ignored unless
  `imu.use_adxl375` is set to true.
- **Tape:** pink bands on the **top of the bat**.
- **Camera:** level and **straight on** (not angled), about **8–10 ft (2.4–3 m)** from the hitter. Try a few
  distances and keep the one with the best `seen_pct` in `swing.json`.
- **Each swing:** hold the stance still for about 1 s, swing, then **hold the finish still**. The held finish is how
  the software knows the swing ended.

### 1. Pick the trigger (once per mount)

```bash
python3 -m dckft.trigger_tune --record 30 --port /dev/ttyUSB0
```

During the 30 s recording:
- stand still
- take 3–5 swings, holding each finish
- waggle and walk around in between

The tool compares |a|−g, |(a_x, a_y)|, each signed axis (−y is the advisor's guess) and the wrist rate. It then
prints the best `swing.trigger_source` and `trigger_accel_ms2`. Put those in `run_schema_config.json`, in the
`"swing"` section.

### 2. Run

```bash
python3 run_stream.py --port /dev/ttyUSB0 --cam-config Cameras/cage.json
```

This starts three things:
- the camera tracker, detecting pink live
- the IMU reader
- the live consumer

Every frame and IMU sample is kept in `sessions/stream_*/` (`camera.jsonl`, `imu.csv`, plus `camera.log` and `imu.log`).

When the trigger fires and the finish is held, the window around the swing is fused and shown in the same 3D render
as `run_schema.py`. Outputs go to `swing_NN/`. Press Ctrl+C to stop.

- **Orin Nano:** if the camera log warns that processing is slower than the frame period, set the camera to
  60 fps in the camera config.
- **Reprocess a session later:**

```bash
python3 -m dckft.run --config run_schema_config.json process \
    --cam sessions/stream_X/camera.jsonl --imu sessions/stream_X/imu.csv --all
```

### What to read for each swing (the end-of-project goal)

The terminal and `swing.json` lead with these, using Blast Motion's definitions:
- **Attack angle:** the angle of the barrel's path at impact relative to horizontal; positive means swinging up.
  Blast's target is +6 to +14°.
- **Vertical bat angle:** the angle of the bat, knob to barrel, relative to horizontal at impact; negative means the
  barrel is below the hands. Blast's college range is −23 to −33°.

Both come from the fusion and, as a cross-check, from the camera alone (`*_cam_deg`).
- **Camera-only orientation:** the camera-only value comes from a quaternion built from the pink centroids in each
  frame. Its rate of change gives the bat's angular velocity (ω = 2 q̇ q*). That is saved in
  `camera_orientation.csv` and compared with the IMU in `diagnostics.camera_quaternion_vs_imu`.
- **Availability:** the angles are reported whenever the camera saw the bat pose at contact, even if the speed is
  unavailable. Hand speed is measured 6 in from the knob, as Blast does.

### New IMU firmware

When the IMU's new firmware (with better timestamps) is ready, add its frame layout to `FORMATS` in
`dckft/imu/frames.py`. There is a commented example: an ICM-only frame with a 64-bit µs timestamp. Then set
`imu.frame_format` in the config. Nothing else changes. A 64-bit device timestamp never wraps, and it is mapped onto
the Jetson clock the same way.
