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

## One swing per clip

1. Open `http://localhost:8000`. Select 960×600/120 FPS or 1920×1080/60 FPS, and a
   five or ten second clip. Changing mode preserves the supplied hue/exposure settings.
2. Click **Record one swing**. Wait for **RECORDING NOW** in the page log.
3. Hold the stance still for about one second, swing once, and hold the finish.
4. Capture stops at the configured frame count. **Stop** finalizes an earlier clip.
5. The app closes both recording processes before detecting every saved SVO frame,
   fitting camera geometry, aligning the recorded IMU and running fusion.
6. Open the saved swing. Play or scrub the camera-led 3D replay; drag to rotate.

There is no live preview/detection/fusion workload during video capture. The strict
advisor ring buffer is still used by the synthetic demo and separate streaming CLI,
but it no longer blocks web recording when radio packets are missing. Saved SVO
timestamps, not an encoder warning or the requested FPS, determine actual saved FPS.

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

## Saved files

`app_data/sessions/<id>/` keeps `camera.svo2`, `imu_session.csv`, capture and detection
logs, both config snapshots, `capture_frames.csv` (acquired timestamps), and
`camera_frames.csv` (every decoded SVO frame, including failed detections).

`app_data/swings/<id>/` contains the raw camera/IMU copies, config snapshots,
`camera_quality.json`, `swing.json`, metric `fused.csv`, `replay.csv`, `replay.json`,
marker exports and `swing.png`. `replay.json`, `replay.csv`, browser playback and the
PNG use the same camera-led poses. The replay ends at detected swing end.

The page separates whole-clip detection counts from active-swing coverage and the
fraction of active replay samples estimated. These are availability measures, not
proof of centimetre-level position or accurate bat speed. Compare marker residuals,
time alignment and calibration error before using metrics for coaching.

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
