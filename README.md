DC-KFT — Kinematic Fusion Tracker
AI Baseball Training capstone project (ECE, Villanova University, Spring/Fall 2026). A portable
motion-capture system that fuses a single stereo camera's optical bat-tracking with radio-telemetry
IMU data into a 3D swing render, plus LLM-generated coaching feedback.
Team: Zachary Buxton (Team Captain & Data Fusion Lead), Charles Power (Hardware & IMU/Radio
Lead), Xavier Kolmer (Optical Capture & UI Rendering Lead). Advisors: Dr. Klein, Dr. Jupina, Dr. Wang.
Hardware
Camera: one stereo 120 fps camera, tracking the bat directly (no colored markers, no
frame-interleaving across two cameras).
Telemetry: a 900 MHz radio link carrying IMU data. Sensor electronics (ICM45686 gyro/accel +
ADXL375 high-g accel, built by Dr. Jupina) are inherited from the previous team; the radio link is
new and replaces their serial/COM connection. Exact radio module specs are still being finalized.
Full inherited specs and lessons learned: `docs/previous_group_final_report.md`.
Software
Python. Optical and radio/IMU streams are fused with quaternion math (Horn/Davenport pose solving,
no Euler angles — avoids gimbal lock) and, as the main open development target, an Extended Kalman
Filter for gyro-drift correction and magnetic-interference rejection. Final output renders as a 3D
swing digital twin with OpenRouter-generated coaching notes.
Targets: ≥60 fps effective capture, <5% tracking error, <5s render latency, ISB joint-coordinate
standards.
Repository layout
```
docs/              Background, proposal, previous team's report
Cameras/           Optical capture (stereo camera + bat tracking)
IMU/               Radio/IMU telemetry
Data Processing/   Fusion engine — quaternion pose, angular velocity, EKF
Rendering/         3D render + coaching output (new, no legacy code)
tests/             Test suite
```
Each subsystem folder has its own `legacy/` subfolder with pre-pivot code from the previous team,
plus a README noting what carries over to the new hardware.
Getting started
```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```
Docs
`docs/background.md` (system overview & roadmap) · `docs/proposal.md` (April 2026 proposal,
pre-pivot) · `docs/previous_group_final_report.md` (inherited hardware & lessons learned)
