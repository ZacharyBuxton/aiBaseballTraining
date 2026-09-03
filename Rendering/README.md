# Rendering (UI / Coaching Output Subsystem)

No legacy code exists for this subsystem — the previous team never reached the rendering stage
(see `../docs/previous_group_final_report.md`), and it isn't part of the three folders of code
this team inherited/wrote either. This folder is a placeholder for new work.

## Scope

- Converts fused pose/motion data (from `../Data Processing`) into a 3D swing/bat render.
- Computes simplified, actionable swing metrics (max bat speed, angle of attack, trunk twist,
  time-to-contact, etc.) rather than exposing raw 4D vector data — see `../docs/proposal.md` and
  `../docs/previous_group_final_report.md` for metrics the team and prior research have
  identified as useful.
- Integrates the OpenRouter API to turn metrics into plain-text coaching feedback.

## Target

Render within 5 seconds of a completed swing (see `../docs/background.md` for the full spec
list).
