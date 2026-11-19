# Camera-led web capture

Start the app with the measured bat geometry and the tuned camera detector settings:

```bash
cd ~/aiBaseballTraining
export LD_PRELOAD="/lib/aarch64-linux-gnu/libGLdispatch.so.0${LD_PRELOAD:+:$LD_PRELOAD}"
python3 run_app.py --no-pull --port /dev/ttyACM0 \
  --config run_schema_black30.json \
  --cam-config Cameras/web_120fps.json
```

The preload is the workaround used on this project's JetPack 5/Python 3.8 Jetson for
the OpenGL static-TLS import error. Other machines generally do not need it.
Keep the existing working dependencies on that Jetson; do not reinstall the current
requirements wholesale on Python 3.8. The new Python source uses Python 3.8 syntax.

## Continuous IMU-triggered capture (default)

1. Open `http://localhost:8000`. Select 960×600/120 FPS or 1920×1080/60 FPS, and leave
   capture set to **Continuous: IMU trigger, 3 s before + 3 s after**.
2. Click **Start continuous capture**. The camera records SVO parts and the receiver logs
   actual IMU samples. Wait for **ARMED**, after the camera is open and three seconds of
   elapsed IMU pre-roll are available. A connected receiver alone does not prove data is arriving.
3. Swing once and hold the finish. A rising edge above the configured IMU threshold
   (default gyro magnitude > 10 rad/s) defines time zero for the six-second saved window.
4. After three seconds, the window is queued for processing. The live camera and receiver
   keep running. The signal must return below threshold before a subsequent rising edge.
   Swing again after the post-roll completes and the status returns to **armed**.
5. Each completed window appears in the swing list after offline detection and fusion.
   A failed window is preserved and does not stop processing later triggers.
6. **Stop** ends continuous capture and drains the analysis queue. Stopping during a
   post-roll preserves a partial window labelled `interrupted_post_roll`; it is not a full six seconds.

The web ring uses `swing.capture_pre_s` / `capture_post_s` (both default 3), not an assumed
1200 samples at 400 Hz. Ordinary positive timestamp gaps are counted without resetting;
missing IMU samples remain missing. Device-clock restarts clear the history and require
new pre-roll. A silent radio cannot trigger; if it stops after a trigger, a wall-clock
watchdog closes the requested interval with only the actual received samples.

The live recorder does no marker detection. It closes an SVO part after a completed
trigger request, or every eight seconds, while keeping the camera device open. Detection
and fusion for closed parts run in a queue alongside recording. Encoder file handoffs
and shared CPU/GPU work can still drop camera frames; timestamps reveal those gaps.
On Linux the detector runs at lower CPU priority. Continuous recording keeps raw data
on disk until the session is stopped; low disk space stops the recorder with an error.

The strict sample-count advisor ring remains unchanged for the separate streaming CLI
and synthetic demo. **Manual test clip** retains the previous 5/10-second record-first
workflow. Saved SVO timestamps, not an encoder warning or requested FPS, determine actual FPS.

## What the replay means

- Pink sections are fitted from camera observations and short interpolation between
  camera poses. Individual detector candidates are not automatically valid 3D poses.
- Orange dashed sections are camera-anchored IMU/model predictions across optical gaps.
  Wrist rotation is not identical to bat rotation; flex and integration drift limit
  these estimates. Endpoint constraints smooth reacquisition but do not recover truth.
- Speed is computed on the conservative metric trajectory, not the display replay.
  Optical gaps/reacquisition and IMU gaps over 10 ms remain excluded from validation.
  Speed may be unavailable while the bat remains visible in the replay.
- With no usable IMU data or calibration, fusion fails explicitly and preserves the raw
  video/CSV. It does not substitute a synthetic swing or pretend to fill with an absent IMU.
- A unique consistent marker pair can survive a third bad band. Ambiguous geometry is
  still rejected. Raw camera JSONL is retained for comparison with cleaned observations.
- The browser and static render label X (camera-right, levelled), Y (horizontal forward),
  and Z (up), relative to the knob at replay start. Scrubbing shows knob and barrel XYZ
  in metres. Choose 3D, XY, XZ or YZ views; the speed chart leaves unsupported intervals blank.

## Saved files

For continuous capture, `app_data/sessions/<id>/` keeps the full `imu_session.csv`, logs,
both config snapshots and `camera_parts/` (closed SVO parts, metadata and acquired-frame CSVs).
Each `swing_001/`, `swing_002/`, etc. contains:

- `window.json`: exact requested start/trigger/end timestamps, threshold, actual IMU span,
  packet-gap counts and interrupted-window flag.
- `imu_session.csv`: observed samples inside that time window, with original timestamps.
- `camera_manifest.json` and immutable SVO parts: offline decoding filters to the requested
  bounds, sorts across file boundaries and does not synthesize missing images. The raw
  SVO parts may contain extra context outside the window; the manifest defines the saved
  six-second selection. Hard links avoid duplicating shared video on the Jetson filesystem.
- `camera_session.jsonl`, `camera_frames.csv` and `detection.log` after processing.

Manual test clips retain `camera.svo2` directly in their session folder.

`app_data/swings/<id>/` contains the raw camera/IMU copies, config snapshots,
`camera_quality.json`, `swing.json`, metric `fused.csv`, `replay.csv`, `replay.json`,
marker exports and `swing.png`. `replay.json`, `replay.csv`, browser playback and the
PNG use the same camera-led poses. Triggered swings also keep `window.json`,
`camera_manifest.json` and their SVO parts in the swing folder. The replay ends at detected
swing end, even though the raw saved interval includes three seconds of post-trigger data.

The page separates whole-clip detection counts from active-swing coverage and the
fraction of active replay samples estimated. These are availability measures, not
proof of centimetre-level position or accurate bat speed. Compare marker residuals,
time alignment and calibration error before using metrics for coaching. The audit includes
the largest decoded-frame gap, estimated missing frame slots and missing window edges.

Replay the exact selected raw camera window from a triggered swing folder:

```bash
python3 Cameras/track_bat.py --svo-manifest app_data/swings/SWING_ID/camera_manifest.json \
  --config app_data/swings/SWING_ID/camera_config.json \
  --bat-config app_data/swings/SWING_ID/fusion_config.json --no-zmq --preview
```

## Reprocess an existing paired recording

After updating the code, existing camera JSONL and IMU CSV can use the connected
replay without a new hardware recording:

```bash
python3 -m dckft.run process --config run_schema_black30.json \
  --cam app_data/swings/SESSION_ID/camera.jsonl \
  --imu app_data/swings/SESSION_ID/imu.csv \
  --out reports/reprocessed_SESSION_ID --no-show
```

This preserves original data. It cannot recover camera frames never captured or
IMU samples never received. The separate camera-only renderer remains useful for
checking the optical observations before evaluating fusion.
