# Project Background

_Source: project knowledge base "Background info" doc, current as of September 2026._

## 1. Overview & Problem Statement

**The problem:** Traditional baseball batting practice coaching relies on the naked eye or
standard 30 FPS video, both of which miss "micro-inefficiencies" in elite collegiate and
professional swings. Modern technological solutions often hand athletes raw data spreadsheets
(angular velocity, rotational acceleration), causing "paralysis by analysis" and disrupting
natural athleticism.

**The solution (DC-KFT):** A highly portable, multimodal motion capture system that fuses optical
video data with inertial/radio sensor telemetry. By combining these data streams, the system
eliminates the visual occlusion errors of cameras and the data drift of sensors. The system
outputs a highly accurate, intuitive 3D digital twin of the swing alongside LLM-generated
coaching recommendations.

## 2. Team & Roles

- **Zachary Buxton** — Team Captain & Data Fusion Lead (Computer Engineering). Project management,
  Python scripts, asynchronous data synchronization (queues/stacks), API integration.
- **Charles Power** — Hardware & IMU/Radio Subsystem Lead. WitMotion IMU / radio hardware,
  mitigating interference (Extended Kalman Filters), raw telemetry capture.
- **Xavier Kolmer** — Optical Capture & UI Rendering Lead. Camera rig calibration, 3D rendering,
  UI dashboard.
- **Advisors:** Dr. Klein (course instructor), Dr. Jupina, Dr. Wang.

## 3. System Architecture (four subsystems)

- **A. Optical Capture Subsystem** — see the hardware pivot note below for the current design.
- **B. Telemetry Subsystem** — see the hardware pivot note below.
- **C. Data Fusion Engine (the core)** — a GPU processing pipeline that interleaves the optical
  feed(s) and overlays quaternion telemetry data to calculate the exact spatial mechanics of the
  swing.
- **D. UI / Rendering Subsystem** — converts the fused 4D vector data into a 3D stick-figure
  render with simplified, actionable metrics (e.g. max bat speed, angle of attack, trunk twist).

## 4. Hardware & Bill of Materials (as of the original proposal)

- **Processing:** NVIDIA THOR dual-GPU computing rig (provided by MakerSpace).
- **Optical (original design):** 2x high-speed cameras (Logitech C920x or MakerSpace equivalents).
- **Sensors (original design):** 1x WitMotion BWT901BLE 9-axis Bluetooth IMU sensor.
- **Physical props:** standard practice baseball bat with custom 3D-printed knob mounts and
  high-contrast neon pink/green gaffer tape for optical nodes.

**⚠️ Hardware pivot (September 2026):** the camera scheme is changing to a **single stereo 120 fps
camera**, so frame-interleaving across two cameras is no longer needed. The system will track the
**bat itself** instead of colored balls/nodes, and will use a **900 MHz radio receiver** for
telemetry instead of the original Bluetooth IMU link. See `docs/previous_group_final_report.md`
for hardware/specs inherited from the prior team (radio receiver and custom IMU electronics),
keeping in mind that document predates the camera pivot. See also `docs/proposal.md`
(the current team's April 2026 proposal), which also predates this pivot.

## 5. Software Stack & Engineering Approaches

Entirely Python-based.

- **Primary libraries:** OpenCV (optical tracking), NumPy/SciPy (matrix math), OpenRouter API
  (LLM coaching feedback generation).
- **Quaternion mathematics:** the system strictly uses quaternion-based 4D vectors rather than
  Euler angles or rotation matrices, to prevent mathematical singularities (gimbal lock) during
  rapid multi-axis rotation.
- **Extended Kalman Filters (EKF):** correct inherent gyroscope data drift; use gravity to correct
  pitch/roll drift and filter magnetic interference from batting-cage chain-link fencing.
- **Asynchronous data synchronization (queues & stacks):** FIFO queues buffer camera and radio
  packets so scripts can interleave/align streams without dropping frames or bottlenecking the
  CPU.

## 6. Strict Constraints & Specifications

- **Frame rate:** minimum 60 FPS combined video capture (previously via interleaving two 30 FPS
  feeds; the new single stereo 120 fps camera exceeds this natively).
- **Multimodal error correction:** tracking error margin < 5% (cross-referencing optical
  coordinates with radio/IMU telemetry).
- **Processing latency:** 3D visual output rendered in under 5 seconds (split GPU load: one GPU
  for optical, one for inertial/telemetry).
- **Engineering standards:** joint coordinate systems per International Society of Biomechanics
  (ISB) standards.

## 7. Fall 2026 Roadmap

- **September:** Hardware assembly & sensor calibration (mounting camera(s), securing radio/IMU
  hardware, establishing raw data pipelines).
- **October/November:** Software integration & data fusion (queues to interleave/align feeds,
  quaternion EKF logic).
- **December:** Final system validation & UI polish (OpenRouter LLM coaching notes, finalized 3D
  render).
