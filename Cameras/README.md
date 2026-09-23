# Cameras (Optical Capture Subsystem)

Tracks colored markers on the bat with a **StereoLabs ZED X (4 mm)** through a **ZED Link Duo** on a
Jetson Orin / Orin Nano. It publishes each marker's 3D position to the fusion engine
(`../Data Processing`) at up to 120 fps.

```
ZED X ──GMSL2──> ZED Link Duo ──CSI──> Jetson
                                         │
   grab rectified L/R (SVGA @ 120 fps, manual exposure)
   → detect markers in both images           (marker_detector.py)
   → triangulate each marker                  (stereo.py)
   → JSON payload → ZeroMQ PUSH :5557         (payload.py)
                  → optional JSONL log
```

## Files

| File | Purpose |
|---|---|
| `bringup_check.py` | **Run first.** Opens the camera and verifies real fps, dropped frames, exposure, calibration, and IMU. |
| `track_bat.py` | Main tracker. Works live or on a recorded `.svo2`, and can record while tracking. |
| `zed_camera.py` | Thin ZED SDK wrapper: open live/SVO, manual exposure, grab, record, calibration, gravity estimate. |
| `marker_detector.py` | HSV color-marker detector: coarse search at low resolution, then a full-resolution sub-pixel centroid. |
| `stereo.py` | Rectified-stereo triangulation, depth-error model, and SDK point-cloud lookup. |
| `payload.py` | Payload schema, non-blocking ZMQ PUSH publisher, and JSONL writer. |
| `config.py` | All tunables as dataclasses, with JSON save/load. |
| `legacy/` | The previous team's dual-Kinect sphere tracker, kept for reference. |

Tests live in `../tests/test_cameras.py` and run without a camera or the ZED SDK:
`pytest tests/test_cameras.py -v`.

## Setup (Jetson)

1. Flash JetPack with a version the ZED Link driver page lists. Check yours with `cat /etc/nv_tegra_release`.
2. With the board powered off, mount the ZED Link Duo:
   - **AGX Orin:** Samtec camera connector, no extra power.
   - **Orin Nano:** two 22-pin to 15-pin FPC cables (≤12 cm) plus a 9–19 V barrel supply for the card.
3. Connect the cable chain. The Duo only takes male-to-female cables, so a female-to-female cable will not fit anywhere in this chain.
   - **Chain:** Duo → included 1-to-4 adapter → M-F extension → camera.
   - **Before booting:** plug in the camera.
4. Install the matching ZED Link driver `.deb`, then reboot.
5. Install the ZED SDK for your JetPack.
6. Install `pyzed` into the project venv. With the venv active, run the SDK's Python API script (default install: `python /usr/local/zed/get_python_api.py`).
7. Install the Python dependencies: `pip install -r requirements.txt`.

GMSL2 cameras are not hot-pluggable. After any cable change, reboot or restart the ZED X daemon.

## First power-on

```bash
python Cameras/bringup_check.py                             # 10 s at SVGA/120, 500 µs exposure
python Cameras/bringup_check.py --preview --exposure-us 800 # look at the image
python Cameras/bringup_check.py --record                    # AGX Orin (hardware encoder)
python Cameras/bringup_check.py --record --codec lossless-cpu   # Orin Nano (no hardware encoder)
```

A run passes when measured fps is ≥95% of the target with no timestamp gaps.

Each run writes to `reports/bringup_<time>/`:
- `report.json`: all measurements
- `left.png` / `right.png`: first frame from each side
- `rectification_check.png`: side-by-side image with horizontal lines. A feature should sit on the same row in both halves.

Things to check:
- **Frame rate:** 120 fps is only available at SVGA (960×600). If the report shows 60 fps, the resolution is wrong.
- **Image brightness:** if the image is too dark at short exposure, add light first, then analog gain (`--analog-gain`, mdB). Longer exposure brings back motion blur.
- **Depth budget:** the depth table is computed from the camera's real `fx` and baseline. It shows how depth error scales with distance and centroid precision.

## Tracking

```bash
# Live, preview window, publish to the fusion engine
python Cameras/track_bat.py --preview

# Save tuned settings, then reuse them
python Cameras/track_bat.py --preview --exposure-us 700 --save-config Cameras/cage.json
python Cameras/track_bat.py --config Cameras/cage.json

# Record-then-process (recommended until live speed is proven on the Jetson)
python Cameras/track_bat.py --record reports/swing01.svo2 --no-zmq --quiet
python Cameras/track_bat.py --svo reports/swing01.svo2 --jsonl reports/swing01.jsonl

# Cross-check the stereo path against the SDK's neural depth
python Cameras/track_bat.py --svo reports/swing01.svo2 --depth-source sdk --jsonl reports/swing01_sdk.jsonl
```

**Why record first:** at 120 fps the whole per-frame budget is 8.3 ms. Detection on both images took about 6–7 ms on an x86 dev machine, and the Orin Nano's CPU is slower. If processing falls behind live, the SDK drops frames; the tracker prints a warning when that happens. SVO replay processes every frame no matter how long it takes, which comfortably fits the <5 s render target for a single swing.

**Start the consumer first.** The publisher never blocks: payloads sent before a consumer connects are dropped and counted (`no-consumer drops` in the status line).

## Payload

This keeps the legacy shape (`../Data Processing/legacy/sample_data/frame_payload.json`), so
`Quaternion_Scheme_12_2.py` can consume it. Extra fields are additive.

```json
{
  "schema_version": "1.1",
  "coord_sys": "zedx_left_rect_opencv_meters_v1",
  "camera_id": "cam_0",
  "frame_id": 812,
  "ts_ns_image": 1789657031153087628,
  "ts_ns_mono":  1789657031153087628,
  "ts_ns_wall":  1789657031158201113,
  "depth_source": "stereo",
  "points": [
    {"id": "neon_pink", "pos": {"x": 0.299, "y": -0.085, "z": 2.994},
     "disparity_px": 28.05, "row_err_px": 0.003,
     "uv_left": [549.86, 280.2], "uv_right": [521.8, 280.2]}
  ]
}
```

**Coordinates:**
- **Frame:** the left camera's rectified frame, in meters.
- **Axes:** OpenCV convention (x right, y down, z forward).

**Timestamps:**
- `ts_ns_image`: the SDK's image capture time on the Jetson system clock. **Use this for fusion.** The IMU/radio receiver should stamp packets with `time.time_ns()` on the same Jetson so both streams share a clock.
- `ts_ns_mono`: a copy of `ts_ns_image`, kept because `angular_velo_computation.py` reads that field name.
- `ts_ns_wall`: publish time. `ts_ns_wall - ts_ns_image` is the pipeline latency.

**Transport:** a PUSH socket binding `tcp://*:5557`. The legacy pose solver connects a PULL socket to that port. Change its hard-coded `JETSON_IP` to `127.0.0.1` when both run on the Jetson.

`track_bat.py --jsonl X` also writes `X.meta.json`, containing:
- camera info
- calibration
- config
- the camera IMU's gravity estimate, for leveling the world frame. Confirm the IMU axis convention with a tilt test before relying on it.

## Markers: important for the bat

- **Tuning:** marker ids and starting hues come from the previous team (`neon_pink`, `neon_green`, `neon_yellow`, `neon_blue`). Re-tune `markers.hues` under the cage lighting and save it with `--save-config`.
- **Roll is unobservable along the bat.** Markers placed only along the bat's length are nearly collinear. Horn's method (`Quaternion_Scheme_12_2.py`) needs **≥3 non-collinear points** for a full orientation. With collinear markers the camera gives position plus the bat's long axis, but not roll about that axis.
  - **Option 1:** mount markers off-axis, for example on a small fin or cross at the knob.
  - **Option 2:** let the IMU supply roll in the EKF.
- **Update the model coordinates.** `MODEL` in `Quaternion_Scheme_12_2.py` still holds the old sphere's coordinates. Replace them with the measured positions of the bat markers.
- **Marker size:** keep markers big enough to stay above `min_area_px` at the working distance.
- **Glare:** matte tape reflects less than the painted sphere did, which was the previous team's range limiter.

## Known limitations / next steps

- **Swing detection:** there is no automatic swing trigger yet. Sessions record continuously; trimming to the swing window belongs in `Data Processing` or a later trigger (IMU threshold).
- **Blur and depth:** color detection is sensitive to lighting, and depth error grows as Z². Keep the camera around 2–3 m from the swing, and at least 1–1.5 m (the 4 mm lens's minimum depth).
- **Live speed:** live 120 fps processing has not been verified on the actual Jetson. Measure it with `track_bat.py` and fall back to record-then-process if needed.
