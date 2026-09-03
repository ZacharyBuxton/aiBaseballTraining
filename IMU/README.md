# IMU / Telemetry Subsystem

## Status: electronics likely carry over; link layer is changing

The custom IMU sensor (ICM45686 + ADXL375 chipset, built by Dr. Jupina) was inherited from the
previous capstone team and is still in the current team's possession — so unlike the camera
schema, this hardware itself probably isn't being replaced. What **is** changing, per the
September 2026 pivot (see `../docs/background.md`), is the telemetry link: the project is moving
to a **900 MHz radio receiver**, which may replace or sit alongside the serial/COM-port link the
legacy script below uses. Treat the data-format and frame-validation logic here as reusable;
treat the transport layer (`pyserial_asyncio` over a Windows COM port) as likely needing
replacement to match the new radio hardware.

## Contents

- `legacy/imu_data_collection.py` — reads IMU frames over a serial connection at 921600 baud
  (30-byte frames, validated by a status word), converts to physical accel/gyro units via
  hard-coded scale factors, and writes to CSV. Windows-only as written (uses `ctypes.windll` and
  a hard-coded `COM10` port).
- `legacy/INSTRUCTIONS.md` — the previous group's setup notes, including a note that this script
  only runs from `cmd`, not from inside VS Code.

## New work (once the 900 MHz radio link is finalized)

New telemetry ingestion code will live directly in this folder. It should reuse the frame
validation and unit-conversion logic from the legacy script where it still applies, but replace
the transport with whatever the 900 MHz radio receiver exposes (raw serial, a vendor SDK, etc.),
and should be made cross-platform rather than Windows-only.
