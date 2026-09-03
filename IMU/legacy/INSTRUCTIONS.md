# IMU Folder Instructions

_Written by Dmitrii Kapranov (previous group)_

To run `imu_data_collection.py`, you will need the IMU tracker and its corresponding receiver.
Some must-haves to run the file:

1. Tracker is either plugged into a laptop or connected to a charged 3.7 V battery.
2. Receiver is plugged into a laptop (shows up as a serial/COM port, 921600 baud, 30-byte frames,
   `status_var == 4095` marks a valid frame).
3. The folder where you saved the python file contains a folder called `reports/`.
4. The file has admin permissions.

Note from the previous group: in their experience the file only runs from the command prompt
(`cmd`), not from VS Code — suspected to be a Python interpreter mismatch within VS Code, but
never confirmed.

Sensor conversion constants (accel/gyro scale factors) are hard-coded in `process_frame()` — see
`imu_data_collection.py` for the raw values, which are specific to the ICM45686 / ADXL375 sensor
pair used by the custom IMU board.
