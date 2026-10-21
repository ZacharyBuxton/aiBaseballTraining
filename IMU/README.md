# IMU / Telemetry Subsystem

**Current code: `dckft/imu/`.** The cross-platform reader for the 900 MHz radio receiver.

- **Sensor:** ICM45686 + ADXL375 board (built by Dr. Jupina), worn on the **wrist**.
- **Link:** the radio carries the same 30-byte frames as the old serial link: 921600 baud, 400 Hz.

```bash
python -m dckft.imu.reader --list-ports
python -m dckft.imu.reader --port /dev/ttyUSB0 --csv sessions/imu.csv      # also publishes ZMQ :5558
python -m dckft.imu.reader --calibrate 3                                    # gyro bias at rest
python -m dckft.imu.reader --replay reports/ICM45686_and_ADXL375_test_data_2026-09-30_16-10-23.csv
```

## What changed from `legacy/imu_data_collection.py`

- **Cross-platform.** No `ctypes.windll` and no hard-coded `COM10`. The port is auto-detected, or set
  with `--port`.
- **Resync.** A bad byte costs one frame. The old reader threw away its whole buffer.
- **Timestamps.** Every sample carries a host `ts_ns_host` (Jetson clock) for camera fusion.
  `dckft.data` regresses it against the device µs counter to remove USB batching jitter.
- **Run length.** It runs until Ctrl+C or `--samples/--seconds`, not a fixed 2400 samples.
- **Same CSV columns** as before, plus `ts_ns_host`. Old CSVs still load; fusion then relies on
  cross-correlation alone.

## Contents

- **`reports/`:** a sample capture from 2026-09-30 (bench, then hand motion).
- **`legacy/`:** the previous team's Windows reader and setup notes, kept for reference.
