"""
Tests for the Cameras/ subsystem that run without a ZED camera or the ZED SDK.
They cover marker detection, stereo triangulation, and the payload contract
with the legacy pose solver.

    pytest tests/test_cameras.py -v
"""
import json

import cv2 as cv
import numpy as np
import pytest

from config import Config, MarkerConfig, load_config, save_config
from marker_detector import ColorMarkerDetector
from payload import build_frame_payload
from stereo import StereoCalib, depth_error, match_and_triangulate, triangulate

H, W = 600, 960                       # ZED X SVGA
CALIB = StereoCalib(fx=700.0, fy=700.0, cx=480.0, cy=300.0, baseline_m=0.12, width=W, height=H)
HUES = {"neon_pink": 170, "neon_green": 60, "neon_yellow": 28, "neon_blue": 105}


def hue_to_bgr(h):
    return tuple(int(c) for c in cv.cvtColor(np.uint8([[[h, 255, 255]]]), cv.COLOR_HSV2BGR)[0, 0])


def draw_marker(img, u, v, hue, radius=6.0):
    """Anti-aliased filled circle at a sub-pixel center (4 bits of sub-pixel precision)."""
    s = 16
    cv.circle(img, (int(round(u * s)), int(round(v * s))), int(round(radius * s)),
              hue_to_bgr(hue), -1, cv.LINE_AA, shift=4)


def project(xyz, calib=CALIB):
    x, y, z = xyz
    u_l = calib.fx * x / z + calib.cx
    u_r = calib.fx * (x - calib.baseline_m) / z + calib.cx
    v = calib.fy * y / z + calib.cy
    return u_l, u_r, v


def background():
    rng = np.random.default_rng(0)
    return rng.integers(20, 90, size=(H, W, 3), dtype=np.uint8)   # dull, low-saturation clutter


# ------------------------------------------------------------------ detector
def test_detects_each_marker_with_subpixel_accuracy():
    img = background()
    truth = {"neon_pink": (100.3, 80.7), "neon_green": (500.6, 300.2),
             "neon_yellow": (800.25, 450.75), "neon_blue": (250.5, 520.4)}
    for m, (u, v) in truth.items():
        draw_marker(img, u, v, HUES[m])
    dets = ColorMarkerDetector(MarkerConfig()).detect(img)
    assert set(dets) == set(truth)
    for m, (u, v) in truth.items():
        assert abs(dets[m].u - u) < 0.25 and abs(dets[m].v - v) < 0.25, (m, dets[m])


def test_pink_hue_wraparound():
    for hue in (2, 176):
        img = background()
        draw_marker(img, 400, 300, hue)
        dets = ColorMarkerDetector(MarkerConfig()).detect(img)
        assert "neon_pink" in dets, f"hue {hue} not detected as pink"


def test_ignores_tiny_blobs():
    img = background()
    draw_marker(img, 400, 300, HUES["neon_green"], radius=1.0)
    assert "neon_green" not in ColorMarkerDetector(MarkerConfig()).detect(img)


def test_tracking_follows_fast_motion_and_recovers_after_loss():
    det = ColorMarkerDetector(MarkerConfig())
    # ~30 px per frame, roughly a bat tip at 120 fps
    for i in range(10):
        img = background()
        draw_marker(img, 100 + 30 * i, 300, HUES["neon_green"])
        d = det.detect(img)["neon_green"]
        assert abs(d.u - (100 + 30 * i)) < 0.5
    assert det.detect(background()) == {}                 # marker occluded
    img = background()
    draw_marker(img, 850, 100, HUES["neon_green"])        # reappears far away
    d = det.detect(img)["neon_green"]
    assert abs(d.u - 850) < 0.5 and abs(d.v - 100) < 0.5


def test_jump_outside_roi_found_on_same_frame():
    det = ColorMarkerDetector(MarkerConfig())
    img = background()
    draw_marker(img, 100, 100, HUES["neon_blue"])
    det.detect(img)
    img = background()
    draw_marker(img, 900, 550, HUES["neon_blue"])         # far outside the prediction window
    assert abs(det.detect(img)["neon_blue"].u - 900) < 0.5


# -------------------------------------------------------------------- stereo
def test_triangulate_roundtrip():
    xyz = (0.25, -0.40, 3.0)
    u_l, u_r, v = project(xyz)
    assert np.allclose(triangulate(u_l, v, u_r, CALIB), xyz, atol=1e-9)


def test_triangulate_rejects_nonpositive_disparity():
    assert triangulate(400.0, 300.0, 400.0, CALIB) is None
    assert triangulate(400.0, 300.0, 410.0, CALIB) is None


def test_depth_error_grows_with_square_of_distance():
    assert depth_error(4.0, CALIB) == pytest.approx(4 * depth_error(2.0, CALIB))


def test_end_to_end_synthetic_stereo():
    truth = {"neon_pink": (0.10, -0.20, 2.5), "neon_green": (-0.30, 0.10, 3.0),
             "neon_yellow": (0.00, 0.00, 3.5), "neon_blue": (0.40, 0.30, 2.0)}
    left, right = background(), background()
    for m, xyz in truth.items():
        u_l, u_r, v = project(xyz)
        draw_marker(left, u_l, v, HUES[m])
        draw_marker(right, u_r, v, HUES[m])
    cfg = MarkerConfig()
    pts = match_and_triangulate(ColorMarkerDetector(cfg).detect(left),
                                ColorMarkerDetector(cfg).detect(right),
                                CALIB, cfg.max_row_mismatch_px, 1.0, 10.0)
    assert set(pts) == set(truth)
    for m, xyz in truth.items():
        # 0.25 px disparity error at 3.5 m with these intrinsics is ~3.6 cm
        tol = depth_error(xyz[2], CALIB, 0.25)
        assert np.linalg.norm(np.subtract(pts[m].xyz, xyz)) < tol, (m, pts[m].xyz, xyz)


def test_rejects_row_mismatch_and_out_of_range():
    left, right = background(), background()
    draw_marker(left, 500, 300, HUES["neon_green"])
    draw_marker(right, 470, 320, HUES["neon_green"])       # 20 px off the epipolar row
    draw_marker(left, 300, 200, HUES["neon_blue"])
    draw_marker(right, 299.5, 200, HUES["neon_blue"])      # 0.5 px disparity -> ~168 m away
    cfg = MarkerConfig()
    pts = match_and_triangulate(ColorMarkerDetector(cfg).detect(left),
                                ColorMarkerDetector(cfg).detect(right),
                                CALIB, cfg.max_row_mismatch_px, 1.0, 10.0)
    assert pts == {}


# ------------------------------------------------------------------- payload
def test_payload_matches_legacy_pose_solver_contract():
    """Mirrors world_from_payload_any() in Quaternion_Scheme_12_2.py."""
    payload = build_frame_payload("cam_0", 42, 1_700_000_000_123_456_789,
                                  {"neon_pink": {"xyz": (0.1, -0.2, 3.0), "disparity_px": 28.0}})
    payload = json.loads(json.dumps(payload))                  # must be JSON-serializable
    assert payload["camera_id"] == "cam_0" and payload["frame_id"] == 42
    assert payload["ts_ns_mono"] == payload["ts_ns_image"] == 1_700_000_000_123_456_789
    p = payload["points"][0]
    assert p["id"] == "neon_pink"
    assert (p["pos"]["x"], p["pos"]["y"], p["pos"]["z"]) == (0.1, -0.2, 3.0)
    assert p["disparity_px"] == 28.0


def test_marker_ids_match_legacy_model_keys():
    legacy_model_keys = {"neon_yellow", "neon_pink", "neon_green", "neon_blue"}
    assert set(MarkerConfig().hues) == legacy_model_keys


def test_zmq_push_pull_roundtrip():
    zmq = pytest.importorskip("zmq")
    from payload import FramePublisher
    pub = FramePublisher("tcp://127.0.0.1:55570")
    ctx = zmq.Context.instance()
    pull = ctx.socket(zmq.PULL)
    pull.connect("tcp://127.0.0.1:55570")
    try:
        payload = build_frame_payload("cam_0", 1, 1, {"neon_blue": {"xyz": (0, 0, 2)}})
        for _ in range(50):                      # connection is established asynchronously
            if pub.publish(payload):
                break
            import time
            time.sleep(0.02)
        assert pull.poll(2000)
        assert pull.recv_json()["points"][0]["id"] == "neon_blue"
    finally:
        pull.close(0)
        pub.close()


# -------------------------------------------------------------------- config
def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.camera.exposure_us = 750
    cfg.markers.hues["neon_green"] = [58]
    path = tmp_path / "cfg.json"
    save_config(cfg, str(path))
    loaded = load_config(str(path))
    assert loaded.camera.exposure_us == 750
    assert loaded.markers.hues["neon_green"] == [58]


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"camera": {"exposure_ms": 1}}))
    with pytest.raises(KeyError):
        load_config(str(path))
