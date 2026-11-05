#!/usr/bin/env python3
"""Always-on capture: stream the camera and IMU continuously; the IMU triggers each swing.

Run from the Jetson desktop:   python3 run_stream.py
                               python3 run_stream.py --port /dev/ttyUSB0 --cam-config Cameras/cage.json

Unlike run_schema.py (one fixed 600-frame clip per run), this keeps the camera grabbing and
detecting pink bands and the IMU logging until Ctrl+C. Each swing is cut out by the IMU
acceleration trigger (dckft config: swing.trigger_source / trigger_accel_ms2), ends when the
bat is held still at the finish, and is fused and rendered with the same render as
run_schema.py. Every frame and IMU sample stays on disk in sessions/stream_*/.

Setup (advisor, Nov 5): IMU on the BOTTOM hand, wire side UP; pink tape on the top of the bat;
camera level and straight on, ~8-10 ft (2.4-3 m) from the hitter; hold the stance ~1 s, swing,
and hold the finish still.

Uses the measured bat geometry saved by run_schema.py (run_schema_config.json) unless --config
is given. Pick the trigger first with:  python3 -m dckft.trigger_tune --record 30
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        from dckft.run import main as run_main
        return run_main(["stream", "--help"])
    config = None
    if "--config" in argv:
        i = argv.index("--config")
        config = argv[i + 1]
        del argv[i:i + 2]
    elif (ROOT / "run_schema_config.json").exists():
        config = str(ROOT / "run_schema_config.json")
    else:
        print("No measured bat geometry found. Run `python3 run_schema.py` once (it asks for the "
              "pink band distances and saves run_schema_config.json), or pass --config FILE.")
        return 1
    print(f"Using bat geometry / trigger settings from {config}")
    from dckft.run import main as run_main
    return run_main(["--config", config, "stream", *argv])


if __name__ == "__main__":
    sys.exit(main())
