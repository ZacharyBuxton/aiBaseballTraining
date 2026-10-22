"""Hardware-free checks for capture lifecycle and reporting."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("run_schema", Path(__file__).resolve().parents[1] / "run_schema.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

BAT = {"bat": {"length_m": .84, "sweet_spot_s_m": .69, "hands_s_m": .08,
               "imu_s_m": .05, "marker_s_m": {"pink": .2, "magenta": .45}}}


class SchemaTests(unittest.TestCase):
    def test_saved_timestamps_distinguish_missing_frames_from_detections(self):
        result = runner.timestamp_stats([0, 8333333, 25000000], 120)
        self.assertEqual(result["saved_frames"], 3)
        self.assertEqual(result["estimated_missing_slots_within_recording"], 1)
        self.assertAlmostEqual(result["effective_fps"], 80)
        self.assertEqual(runner.timestamp_stats([], 120)["saved_frames"], 0)

    def test_geometry_rejects_coincident_and_out_of_bat_markers(self):
        runner.validate_geometry(BAT)
        for value in (.2, .9, float("nan")):
            config = copy.deepcopy(BAT)
            config["bat"]["marker_s_m"]["magenta"] = value
            with self.assertRaises(ValueError):
                runner.validate_geometry(config)

    def test_quality_counts_paired_markers_and_host_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "camera_timestamps.json").write_text(json.dumps({
                "nominal_fps": 120, "timestamps_ns": [1000000000, 1008333333, 1016666667]}))
            (path / "camera.jsonl").write_text(json.dumps({"points": [
                {"id": "pink"}, {"id": "magenta"}]}) + "\n")
            (path / "imu.csv").write_text("ts_ns_host\n990000000\n1020000000\n")
            runner.summarize(path)
            stats = json.loads((path / "quality.json").read_text())
            self.assertEqual(stats["tracking"]["frames_with_two_or_more_markers"], 1)
            self.assertAlmostEqual(stats["tracking"]["fraction_of_saved_frames"], 1 / 3)
            self.assertGreater(stats["imu"]["overlap_with_camera_s"], 0)

    def test_worker_stop_flushes_data_and_closes_process(self):
        with tempfile.TemporaryDirectory() as temp:
            worker = runner.Worker("test", [sys.executable, "-u", "-c",
                "import signal,time; signal.signal(signal.SIGINT, lambda *a: exit(0)); "
                "print('ready',flush=True); time.sleep(60)"], Path(temp))
            self.assertEqual(worker.lines.get(timeout=5).strip(), "ready")
            self.assertEqual(worker.stop(), 0)
            self.assertFalse(worker.thread.is_alive())
            self.assertIn("ready", (Path(temp) / "test.log").read_text())

    def fake_rig(self, root, fail_camera=False, silent_imu=False):
        (root / "Cameras").mkdir()
        (root / "dckft/imu").mkdir(parents=True)
        (root / "dckft/__init__.py").touch()
        (root / "dckft/imu/__init__.py").touch()
        (root / "Cameras/track_bat.py").write_text(
            "import sys,time\n"
            + ("raise SystemExit(7)\n" if fail_camera else
               "from pathlib import Path\n"
               "Path(sys.argv[sys.argv.index('--record')+1]).write_bytes(b'fake svo')\n"
               "print('Recording test',flush=True)\n"
               "time.sleep(.2)\n"))
        (root / "dckft/imu/reader.py").write_text(
            "import sys,time,signal\nfrom pathlib import Path\n"
            "signal.signal(signal.SIGINT, lambda *a: exit(0))\n"
            + ("" if silent_imu else
               "Path(sys.argv[sys.argv.index('--csv')+1]).write_text('header\\n'+'1\\n'*100)\n")
            + "time.sleep(60)\n")

    def test_capture_lifecycle_and_early_failure_cleanup(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.fake_rig(root, fail_camera=fail)
                session = root / "session"
                session.mkdir()
                args = SimpleNamespace(port="fake", startup_timeout=3, mode="120fps", frames=600, codec="h265")
                with patch.object(runner, "ROOT", root):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "Camera failed"):
                            runner.capture(session, args, sys.executable, sys.executable, BAT)
                    else:
                        runner.capture(session, args, sys.executable, sys.executable, BAT)
                        self.assertTrue((session / "camera.svo2").exists())
                # The receiver must be stopped and its log closed in either case.
                self.assertTrue((session / "imu.log").exists())

    def test_missing_radio_data_times_out_before_camera_starts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fake_rig(root, silent_imu=True)
            session = root / "session"
            session.mkdir()
            args = SimpleNamespace(port="fake", startup_timeout=.3, mode="120fps", frames=600, codec="h265")
            with patch.object(runner, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "No usable IMU"):
                    runner.capture(session, args, sys.executable, sys.executable, BAT)
            self.assertFalse((session / "camera.log").exists())

    def test_reprocess_simulated_session_through_real_fusion_and_render(self):
        # Only acquisition/SDK replay is substituted; fusion and PNG rendering run unchanged.
        from dckft.config import FusionConfig
        from dckft.simulate import SimParams, simulate, write_session
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp)
            fixture = session / "fixture"
            fixture.mkdir()
            cfg = FusionConfig()
            paths = write_session(simulate(SimParams(seed=3), cfg), str(fixture))
            shutil.copy2(paths["imu"], session / "imu.csv")
            runner.write_json(session / "fusion_config.json", cfg.to_dict())
            runner.write_json(session / "camera_config.json", {})
            real_stage = runner.run_stage
            def stage(name, command, attempt):
                if name == "audit":
                    with open(paths["camera"]) as f:
                        timestamps = [json.loads(line)["ts_ns_image"] for line in f]
                    runner.write_json(attempt / "camera_timestamps.json",
                                      {"nominal_fps": 120, "timestamps_ns": timestamps})
                elif name == "detection":
                    shutil.copy2(paths["camera"], attempt / "camera.jsonl")
                else:
                    return real_stage(name, command, attempt)
            with patch.object(runner, "choose_python", return_value=sys.executable), \
                 patch.object(runner, "run_stage", side_effect=stage):
                result = runner.main(["--session", str(session), "--no-show"])
            self.assertEqual(result, 0)
            manifest = json.loads((session / "run_schema.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            out = Path(manifest["latest_analysis"]) / "out"
            self.assertTrue((out / "swing.json").exists())
            self.assertTrue((out / "fused.csv").exists())
            self.assertTrue((out / "swing.png").exists())


if __name__ == "__main__":
    unittest.main()
