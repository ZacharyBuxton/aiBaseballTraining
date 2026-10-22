# Demo-Day Runbook

Everything runs on the Jetson. Run the commands from the repo root with the venv active.

## 0. One-time setup (per Jetson)

```bash
pip install -r requirements.txt            # plus pyzed from the ZED SDK (Cameras/README.md)
sudo usermod -aG dialout $USER             # serial access for the radio receiver; log out/in
python -m dckft.run dump-config > dckft_config.json
```

Edit `dckft_config.json`:
- `bat.marker_s_m`: measured marker distances from the knob, in metres.
- `bat.length_m` and `bat.sweet_spot_s_m`.

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
python -m dckft.run capture --port /dev/ttyUSB0 --cam-config Cameras/cage.json   # Orin Nano: add --codec lossless-cpu
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
python Cameras/track_bat.py --config Cameras/cage.json --jsonl sessions/cam_$(date +%H%M).jsonl

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
| `Not enough camera bat-axis samples` | Fewer than 2 markers visible; re-tune hues or improve lighting |
| `calib_rms_axis_deg` > 8 in `swing.json` | Markers mis-detected or grip slipped; re-check marker distances in config |
| Live is slow / dropped frames warning | Record first (`track_bat.py --record x.svo2`), then process offline |
| No hardware at all | `python -m dckft.run simulate` or replay a saved session: `python -m dckft.replay --cam ... --imu ...` with `dckft.run live` running |

## 5. Rehearse without hardware

```bash
python -m dckft.run live --session-dir /tmp/live &
python -m dckft.replay --cam sessions/sim/camera.jsonl --imu sessions/sim/imu.csv --loops 3
```
