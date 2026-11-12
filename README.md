# DC-KFT — Kinematic Fusion Tracker

AI Baseball Training capstone (ECE, Villanova University, 2026). A portable motion-capture system that fuses a
120 fps stereo camera tracking the bat with a wrist-worn IMU (900 MHz radio link). It shows a 3D replay of the
swing, a few headline metrics, and coaching notes within seconds of the swing.

**Team:**
- Zachary Buxton: Team Captain & Data Fusion Lead
- Charles Power: Hardware & IMU/Radio Lead
- Xavier Kolmer: Optical Capture & UI Rendering Lead

**Advisors:** Dr. Klein, Dr. Jupina, Dr. Wang.

## How it works

```
ZED X (SVGA@120) ──> Cameras/track_bat.py ──ZMQ :5557──┐
  neon pink tape bands along the bat                   │      dckft.run live / process
                                                       ├──>   sync → self-calibration → quaternion MEKF
Wrist IMU (ICM45686 + ADXL375) ─900 MHz─> USB ──>      │      → trajectory → swing metrics → coaching
  dckft.imu.reader ──────────────────────ZMQ :5558─────┘      → 3D replay window + saved results
```

1. **Camera.** All tape is **neon pink**, which tracked best under cage lighting.
   - `Cameras/track_bat.py` finds the pink bands (default 3) in both images and triangulates them.
   - Since the bands are all one color, it names them `pink_1` (nearest the knob) … `pink_3` by their order along the bat.
   - From these it gets the bat line (knob position and bat axis) at 120 Hz.
   - A single long pink stripe is also supported.
2. **IMU.** `dckft.imu.reader` decodes the 30-byte radio frames at 400 Hz. Each sample is stamped
   with the Jetson clock, the same clock the camera uses.
3. **Sync.** Cross-correlation gives a coarse offset. It is then refined jointly with the calibration,
   to within about 1 ms of the radio latency.
4. **Self-calibration.** Each swing solves for the IMU's orientation in the world and the "grip
   vector" (the bat axis as seen from the wrist). It uses the stance gravity and the camera bat axis.
   No calibration procedure is needed.
5. **Quaternion MEKF + RTS smoother.**
   - The gyro drives the prediction.
   - The camera bat axis corrects heading and pitch, replacing a magnetometer. The ICM45686 has
     none, so fence magnetic interference is a non-issue.
   - Gravity corrects tilt while the hitter is still.
   - Gyro bias is estimated.
   - Recorded data is processed after the fact, so a backward (Rauch–Tung–Striebel) pass lets each
     sample also use the camera fixes that came after it.
   - Quaternions only, no Euler angles.
6. **Trajectory and per-marker gap filling.**
   - Every marker the camera sees gives its own knob fix along the fused bat axis, so one visible
     marker is enough. The knob track is an accelerometer-driven Kalman/RTS smoother over those fixes; where the ICM45686
     clips at ±32 g, the ADXL375 (±200 g) is used instead.
   - Each marker's full track is rebuilt at 400 Hz; frames the camera missed are filled from the
     IMU-driven fusion (`markers_camera.csv`, `camera_filled.jsonl`, flagged `camera` / `imu_fill`).
   - A complementary correction removes wrist-to-bat flex.
   - The IMU bridges blur and occlusion gaps near contact.
7. **Metrics and coaching.** Bat speed, hand speed, attack angle, vertical bat angle, time to
   contact, swing plane, and wrist rate. Coaching is rule-based for now; the LLM coach (OpenRouter)
   plugs in later.

## Results on simulated swings (`python -m dckft.run validate --n 20`)

| Metric | Mean abs. error | 95th pct |
|---|---|---|
| Bat speed at contact | **1.9 %** | 4.5 % |
| Peak hand speed | 2.9 % | 6.8 % |
| Attack angle | 1.2° | 2.5° |
| Vertical bat angle | 0.8° | 1.6° |
| IMU-filled marker position (frames the camera missed) | 10 mm median | 25 mm |
| Processing time (per swing) | 0.48 s mean, 0.61 s max | |

**Simulation conditions.** Every swing in the run includes:
- camera noise
- blur dropouts and an occlusion right before contact
- radio latency and batching
- gyro bias
- quantisation and saturation
- 6° of wrist flex

Both specs pass on simulated swings: <5 % tracking error and <5 s render latency. Live replay from swing end to
render took ~1.8 s. Real-rig numbers still need to be measured (see `docs/FINAL_PLAN.md`).

## Always-on capture (Nov 2026)

`python3 run_stream.py` streams the camera and the IMU continuously. The IMU's acceleration triggers each swing
(tune it with `python3 -m dckft.trigger_tune`), and holding the finish still marks the end of the swing.

Attack angle and vertical bat angle are the headline outputs, using Blast Motion's definitions. Each is reported from
the fusion and from the camera alone, with a camera quaternion built from the pink centroids. Setup and details are in
`docs/RUNBOOK.md`, under "Always-on capture".

## Swing recorder app (Nov 2026)

`python3 run_app.py` on the Orin, then open `http://<orin-ip>:8000` from any device. Press a button to record, and click
any swing to see its render with attack angle and vertical bat angle. **Simulate swing** works without hardware, and
**Update from GitHub** pulls the latest code. Swings are caught by an advisor-style ring buffer: 1200 frames before
and 1200 after |ω| > 10 rad/s. See `docs/RUNBOOK.md`.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# No hardware: simulate a swing and open the 3D replay
python -m dckft.run simulate --out sessions/sim

# Run the tests (no hardware needed)
python -m pytest -q tests
```

On the rig, follow **[docs/RUNBOOK.md](docs/RUNBOOK.md)**. For accurate data, record first and process after:

```bash
python -m dckft.run capture --port /dev/ttyUSB0      # records camera (no detection) + IMU; Ctrl+C to stop
                                                     # then detects markers in every frame and fuses every swing
```

`python -m dckft.run live` gives instant feedback instead, at the risk of dropped frames.

## Repository layout

```
run_app.py         Swing recorder web app (start here on the Orin)
app/               Web app backend + pages (stdlib only)
dckft/             Fusion engine, IMU reader, metrics, render, simulator (new; see dckft/README.md)
Cameras/           ZED X capture + bat-marker tracker (publishes to the fusion engine)
IMU/               IMU subsystem notes, sample capture, legacy Windows reader
Data Processing/   Earlier fusion prototypes + previous team's code (superseded by dckft/)
Rendering/         Rendering subsystem notes (code lives in dckft/render.py)
docs/              Runbook, final plan, background, proposal, previous team's report
tests/             pytest suite (cameras + fusion), runs in CI on every push
```

## Configuration you must set before the final demo

Run `python -m dckft.run dump-config > dckft_config.json`, edit the values below, then pass
`--config dckft_config.json`.

- `bat.marker_s_m`: measured distance (m) of each pink band's centre from the knob end: `pink_1` (nearest the
  knob), `pink_2`, `pink_3`. Space them **unequally** (e.g. 20 / 40 / 70 cm). For a single stripe, use
  `{"stripe_start": …, "stripe_end": …}` and set `markers.layout = "stripe"` in the camera config.
- `bat.length_m`, `bat.sweet_spot_s_m`, `bat.imu_s_m`: bat length, sweet-spot position, and the
  wrist IMU's position along the handle.
- The pink hue in the `Cameras/` config, tuned under cage lighting (see `Cameras/README.md`). Pass the same
  `dckft_config.json` to the camera with `--bat-config`, so it can use the band spacing to name the bands.

## Docs

- `docs/RUNBOOK.md`: demo-day operating steps
- `docs/FINAL_PLAN.md`: the plan to finish, status, and the open hardware items
- `docs/background.md`: system overview
- `docs/proposal.md`: April 2026 proposal
- `docs/previous_group_final_report.md`: the previous team's final report
