#!/usr/bin/env python3
"""DC-KFT swing recorder app -- one button to record, a list of swings, a render for each.

On the Orin:
    python3 run_app.py                                    # then open http://<orin-ip>:8000
    python3 run_app.py --port /dev/ttyUSB0 --cam-config Cameras/cage.json

From any laptop / phone on the same network, open http://<orin-ip>:8000
(`hostname -I` on the Orin prints its IP). On the Orin itself: http://localhost:8000

On start it pulls the latest code from GitHub (git pull --ff-only); the page's "Update from
GitHub" button pulls again and restarts the app. "Simulate swing" tests the whole path with no
camera or IMU attached. Swings are stored in app_data/swings/ (not tracked by git).

Uses the measured bat geometry saved by run_schema.py (run_schema_config.json) when present,
or --config FILE.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from other devices")
    ap.add_argument("--port", help="IMU receiver serial port (default: auto-detect)")
    ap.add_argument("--cam-config", help="Cameras/ JSON config (pink hue, exposure)")
    ap.add_argument("--config", help="dckft config (default: run_schema_config.json if present)")
    ap.add_argument("--no-pull", action="store_true", help="don't git pull on start")
    a = ap.parse_args(argv)

    from app.backend import git_pull, git_info
    if not a.no_pull and os.environ.get("DCKFT_APP_PULLED") != "1":
        print("Checking GitHub for updates ...")
        print(git_pull())
        os.environ["DCKFT_APP_PULLED"] = "1"
        # restart once so freshly pulled code is what actually runs
        os.execv(sys.executable, [sys.executable] + sys.argv)
    info = git_info()
    print(f"Code: {info['commit']} ({info['date']}) {info['subject']}")

    config = a.config
    if config is None and os.path.exists(os.path.join(ROOT, "run_schema_config.json")):
        config = os.path.join(ROOT, "run_schema_config.json")
    from app.backend import Backend
    from app.server import serve
    backend = Backend(config, a.port, a.cam_config)
    backend.say("App started." + ("" if config else
                " No measured bat config found: using example geometry (run run_schema.py once)."))

    def restart():
        import time
        time.sleep(1.0)
        os.environ["DCKFT_APP_PULLED"] = "1"         # the Update button already pulled
        os.execv(sys.executable, [sys.executable] + sys.argv)

    serve(backend, a.host, a.http_port, restart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
