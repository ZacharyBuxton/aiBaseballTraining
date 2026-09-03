# Project Proposal (April 2026) — Summary

_Summarized from "AI Baseball Training — Proposal Document," Charles Power, Zachary Buxton, and
Xavier Kolmer, ECE capstone proposal, Villanova University, April 14, 2026._
_Full interview transcripts and personal details from the original report have been omitted here
for interviewee privacy — this summary retains only the technical content._

## Abstract

The project aims to improve baseball batting-practice feedback using a high-frame-rate camera
combined with a wrist-worn gyroscope, fusing both data streams into a 3D animation of the swing
that is easy for players and coaches to interpret, rather than a spreadsheet of raw numbers.

## Problem & approach

Traditional coaching misses "micro-inefficiencies" in elite swings that are invisible to the naked
eye and to standard 30 FPS video. Purely optical systems suffer visual occlusion; purely inertial
sensors drift over time. Ethnographic research (interviews with coaches, an athletic trainer, a
sports data analyst, and field observations at batting cages) informed three design constraints:
uneven batting-cage lighting requires high-contrast optical tracking targets; chain-link cage
fencing introduces magnetic interference affecting raw IMU data; and portability opens a
secondary market for recruiting evaluation at high school fields.

The literature review (summarized findings, full citations in the original report) supports:
multimodal sensor fusion of optical + inertial data to eliminate each modality's individual
bottlenecks (up to ~23% accuracy improvement over single-sensor systems in comparable
research); quaternion-based orientation math (not Euler angles/rotation matrices) to avoid
gimbal lock during fast multi-axis rotation; and a quaternion-based Extended Kalman Filter (EKF)
to correct gyroscope drift and reject magnetic interference. The design also commits to
International Society of Biomechanics (ISB) joint-coordinate-system standards for the resulting
3D digital twin.

## Design constraints & criteria

1. **Processing latency** — target a 3D render within ~5 seconds of a swing.
2. **Tracking accuracy** — target a 2% error margin (vs. weaker designs at ~10%).
3. **Usability / visual feedback** — avoid "data overload"; an intuitive 3D render beats a raw
   4D-vector spreadsheet.
4. **Combined 60 FPS output** — interweaving the (then) two 30 FPS camera feeds was a strict
   pass/fail criterion.
5. **Quaternion math** — mandatory; a design that could hit gimbal lock was disqualified.
6. **ISB standard compliance** for joint coordinate systems and rotational math.

## Decision & solution (as proposed, pre-pivot)

The team selected "increase the sampling rate" as the winning design direction (14/possible points
in the decision matrix) over alternatives (multi-colored bat tape, a low-pass filter on swing
path), reasoning that higher sample rates reduce compute burden, improve tracking precision, and
make interleaving the dual 30 FPS feeds into 60 FPS more reliable.

The proposed solution (as of April 2026, before the September 2026 hardware pivot described in
`docs/background.md`): two synchronized 30 FPS cameras at offset angles tracking colored spheres
on the batter's hand, plus a gyroscopic hand sensor, fused via Python quaternion math on a single
GPU, producing a 3D render and a swing datasheet. Four subsystems were identified: (1) the
interlaced cameras, (2) the IMU, (3) the GPU fusing both streams, and (4) the Python code tying it
together.

## Materials & tools (as of the proposal)

- NVIDIA THOR dual-GPU rig (MakerSpace-provided).
- Custom IMU sensor built by Dr. Jupina (inherited from the previous capstone team).
- 2x high-frame-rate cameras (to be purchased for the new MakerSpace facility).
- OpenRouter AI API (development assistance, and eventually AI-assisted coaching
  recommendations).
- Python stack: OpenCV, NumPy, SciPy.
- Budget: $250 total (OpenRouter API credits) — GPU, IMU, and cameras were provided/inherited at
  no cost to the team.

## Personnel (as of the proposal)

- **Zachary Buxton** — Team Captain; Gantt chart & deadlines; software/data-fusion focus.
- **Charles Power** — Hardware/sensor focus; gyroscopic sensor implementation.
- **Xavier Kolmer** — Optical tracking & UI; camera setup and data display.

## References

The original proposal cites 16 sources (IEEE journals/conferences on sensor fusion, quaternion
control, motion-capture engineering standards, and existing solutions in golf/basketball/running
biomechanics) plus ethnographic interviews and a field observation study. See the previous
capstone team's final report (`docs/previous_group_final_report.md`) for closely related citations
and hardware specifics; consult the original proposal PDF (not included in this repository) for
the full reference list and interview notes.
