# DC-KFT — Dual-Camera Kinematic Fusion Tracker

AI Baseball Training capstone project (ECE, Villanova University, Spring/Fall 2026). DC-KFT is a
portable, multimodal motion capture system that fuses optical bat-tracking with radio telemetry to
produce an intuitive 3D digital twin of a batting swing, plus LLM-generated coaching feedback —
without handing the player a spreadsheet of raw numbers.

**Team:** Zachary Buxton (Team Captain & Data Fusion Lead), Charles Power (Hardware & Telemetry
Lead), Xavier Kolmer (Optical Capture & UI Rendering Lead). Advisors: Dr. Klein, Dr. Jupina, Dr.
Wang.

## ⚠️ Hardware pivot in progress (September 2026)

The project inherited a dual-camera (2x Azure Kinect) + Bluetooth IMU design from a previous
capstone team, and the current team's own April 2026 proposal continued that same two-camera
approach. **As of September 2026 the design is changing** to:

- A **single stereo 120 fps camera** (replacing the two interleaved 30 FPS cameras — no more
  frame-weaving needed).
- **Tracking the bat directly**, instead of colored balls/nodes on the batter's hand.
- A **900 MHz radio receiver** for telemetry, instead of the original Bluetooth IMU link.

Exact hardware specs for the new camera and radio module are still being finalized — see
`docs/background.md`. The repo is organized by subsystem (matching how the team already keeps
its files), with each subsystem folder holding a `legacy/` subfolder for inherited pre-pivot code
(reference only) and a README explaining what, if anything, carries over to the new hardware.

## Repository layout

```
docs/                  Project background, proposal, and the previous team's final report
Cameras/                Optical capture subsystem
  legacy/               Dual-Kinect colored-sphere tracking (old schema — likely not reusable, see README)
IMU/                    Telemetry subsystem
  legacy/               Serial IMU data collection — electronics likely reusable, link layer changing
Data Processing/        Fusion engine
  legacy/               Quaternion pose solving, angular velocity, cross-correlation alignment — mostly reusable
Rendering/              UI / coaching-output subsystem (no legacy code — new work)
tests/                  Test suite (empty scaffold)
requirements.txt        Python dependencies
```

Each subsystem folder's `README.md` explains what's in its `legacy/` subfolder, whether it's
expected to carry over to the new pivot, and where new code for that subsystem should go.

## System architecture (four subsystems)

1. **Optical Capture** — stereo camera tracks the bat in 3D.
2. **Telemetry** — 900 MHz radio receiver captures rotational/acceleration data.
3. **Data Fusion Engine (the core)** — quaternion-based Extended Kalman Filter fuses the two
   streams into a single accurate spatial-mechanics estimate. Quaternions are mandatory
   (not Euler angles / rotation matrices) to avoid gimbal lock during fast multi-axis rotation.
4. **UI / Rendering** — turns fused data into a 3D render plus simplified metrics (max bat speed,
   angle of attack, trunk twist, etc.) and OpenRouter-generated plain-text coaching notes.

**Target specs:** ≥60 FPS effective capture, <5% tracking error margin, <5s render latency,
ISB-standard joint coordinate systems.

## Getting started

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The `legacy/` scripts in each subsystem folder additionally require hardware-specific packages
(Azure Kinect SDK for `Cameras/legacy`, etc.) — see each subsystem's `README.md` and the
`INSTRUCTIONS.md` files alongside the legacy scripts for details, and note the hard-coded
paths/ports called out there that need updating before those scripts will run on a new machine.

## Documentation

- `docs/background.md` — current system overview, hardware pivot notes, roadmap.
- `docs/proposal.md` — the current team's April 2026 capstone proposal (pre-pivot).
- `docs/previous_group_final_report.md` — the previous team's final report, including the
  inherited hardware specs and lessons learned that motivated several of this project's design
  choices.

## Roadmap (Fall 2026)

- **September** — Hardware assembly & sensor calibration for the new stereo camera + radio setup.
- **October/November** — Software integration & data fusion (the `Data Processing` EKF + quaternion
  pipeline).
- **December** — Final system validation, UI polish, OpenRouter coaching-feedback integration.
