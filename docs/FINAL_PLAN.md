# Plan to Finish (post-midterm, Oct 21 2026 → mid/late November)

## Decisions locked in

- **Compute.** The Jetson runs everything: camera, fusion, and render. The IMU receiver is being moved onto the Jetson.
- **IMU.** The ICM45686 + ADXL375 is worn on the **wrist/hand**. The radio link works and carries the
  same 30-byte frames at 921600 baud.
- **Markers.** Three colored tape markers sit **collinear** along the bat. The camera gives the bat axis
  and knob; roll comes from the IMU.
- **Render.** A bat-only 3D replay in a Python desktop window (matplotlib).
- **LLM coaching.** Deferred. Rule-based coaching is in place, with a hook for OpenRouter.

## Status

| # | Work item | Status |
|---|---|---|
| 1 | Repo foundation: `dckft/` package, config, requirements, CI | ✅ done |
| 2 | Cross-platform IMU/radio reader (resync, ZMQ, CSV, calibrate, replay) | ✅ done, needs a run on the Jetson with the radio |
| 3 | Time sync: shared Jetson clock + device-clock regression + xcorr + joint refinement | ✅ done |
| 4 | Per-swing self-calibration (IMU-to-world orientation, grip vector, gyro bias) | ✅ done |
| 5 | Quaternion MEKF (gyro predict; camera-axis + gravity updates; gating) | ✅ done |
| 6 | Trajectory: accel-aided knob smoother, lever-arm and flex correction | ✅ done |
| 7 | Swing detection, impact detection, metrics | ✅ done |
| 8 | 3D replay window + PNG/MP4 export | ✅ done |
| 9 | Live runner (`dckft.run live`): FIFO buffers, IMU-triggered, raw recording | ✅ done (tested via replay) |
| 10 | Simulator + validation (<5 % speed error, <5 s latency) | ✅ passing on simulated swings |
| 11 | Coaching: rule-based (LLM later) | ✅ rule-based done |
| 12 | Docs: README, runbook, this plan | ✅ done |
| 13 | Per-marker gap filling from the IMU (`markers.py`, outputs tagged camera / imu_fill) | ✅ done (Oct 22) |
| 14 | Record-then-process: `dckft.run capture` / `postprocess`, `track_bat.py --record-only` | ✅ done, needs a Jetson run |
| 15 | Kalman RTS backward smoother; per-marker knob fixes; ADXL375 when the ICM clips | ✅ done (Oct 22) |

## Remaining work, all on the real rig (in order)

1. **Measure the bat** and put the values in the config: marker distances from the knob, length, and sweet spot.
2. **Get the IMU reader running on the Jetson.**
   - Run `python -m dckft.imu.reader --list-ports`, then do a 5-second capture.
   - Confirm 400 Hz, no resyncs, and |accel| ≈ g at rest.
   - Run `--calibrate 3` to record the gyro bias.
3. **Tilt test the ZED gravity.**
   - Check that the `gravity` in `track_bat`'s `.meta.json` points down (+y for a level camera).
   - If not, set `gravity_cam` in the config.
4. **First real swings.** Record 10 swings with `python -m dckft.run capture`, then check the diagnostics in each
   `swing.json`:
   - `calib_rms_axis_deg` should be under about 5°
   - `ekf_camera_rejected` should be about 0
   - `sync_correlation` should be above 0.5
   - `markers`: `seen_pct`, `longest_gap_ms`, `residual_rms_mm` per marker (residual under ~15 mm)
   - first, check the ADXL375 axes match the ICM45686 (tilt the board, compare `accel_*` with `accel_*2`);
     if not, set `imu.use_highg_when_saturated` to false until fixed
   - check that SVO replay keeps the original capture timestamps (the IMU sync relies on it)
5. **Tune the thresholds** on real data:
   - `swing.trigger_rads` and `swing.contact_hg_ms2` (impact spike on the ADXL375)
   - `ekf.grip_flex_rad`
6. **Accuracy check against a reference.** Pick at least one:
   - compare bat speed with a radar gun or a commercial bat sensor (if MakerSpace or the team has one)
   - check that knob height and bat length are consistent in the render
   - check that SVO-replay and live results agree

   Record the results for the final report (the <5 % spec).
7. **Latency on the Jetson.** Note the "swing end -> render" line that `dckft.run live` prints for each swing (spec < 5 s).
8. **Demo rehearsal** using `docs/RUNBOOK.md`.
9. *(Later)* **OpenRouter coaching.** Add a class with `notes(metrics)` in `dckft/coaching.py`; the key goes in the
   `OPENROUTER_API_KEY` environment variable, never in the repo.

## Known limitations

- **Rigid-grip assumption.** The model assumes the bat and wrist turn together. The flex correction
  needs camera coverage; during long camera gaps the gyro is used as-is.
- **Roll about the bat axis.** Roll comes only from the IMU, so it is only as good as the grip vector estimate.
- **Hands point.** Hand speed is measured at a fixed point on the handle (`bat.hands_s_m`), not tracked hands.
- **Live processing speed.** Camera detection at 120 fps has not been proven on the Orin Nano. The live
  runner only *consumes* payloads; if `track_bat` lags, use the record-then-process path.
