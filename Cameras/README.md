# Cameras (Optical Capture Subsystem)

## ⚠️ Status: legacy files likely not reusable

Everything in `legacy/` here targets the **old dual-camera schema**: two Microsoft Azure Kinect
depth cameras tracking a 3D-printed, multi-colored sphere on the batter's hand. As of the
September 2026 pivot (see `../docs/background.md`), the project is moving to a **single stereo
120 fps camera that tracks the bat directly**. Because the hardware, the number of camera feeds,
and the tracking target (bat vs. hand-held sphere) are all changing, the team's expectation is
that little to none of this legacy code carries over as-is — it's kept here for reference only
(HSV color-blob detection approach, ZMQ publishing pattern, threading/capture-queue structure).

## Contents

- `legacy/Device_Tracking_2_16.py` — the previous group's dual-Kinect, colored-sphere tracking
  script. Publishes 3D coordinates over a ZeroMQ `PUB` socket (port 5555) as JSON — see
  `../Data Processing/legacy/Subscription_Ball_Test_11_4_2.py` for a matching subscriber, and
  `../Data Processing/legacy/sample_data/frame_payload.json` for the payload shape.
- `legacy/INSTRUCTIONS.md` — the previous group's setup notes for running that script.

## New work (once stereo camera hardware is finalized)

New capture code for the single stereo camera + direct bat tracking will live directly in this
folder (not under `legacy/`). Open questions to resolve before writing it: exact camera model/SDK,
whether bat tracking uses fiducial markers/tape or model-based detection, and how pose data will
be handed off to `../Data Processing`.
