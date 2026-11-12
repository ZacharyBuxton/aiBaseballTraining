"""Nov 12: advisor-style ring buffer, protocol-v2 framing, and the swing-recorder web app."""

import json
import os
import threading
import types
import urllib.request

import numpy as np
import pytest

from dckft.ring import CameraRing, SwingRing
from dckft.imu import protocol_v2 as v2

DT = 0.0025          # 400 Hz


def _stream(sig, t0=0.0):
    return [{"t_dev_s": t0 + i * DT, "gyro": np.array([s, 0.0, 0.0])} for i, s in enumerate(sig)]


def _push_all(ring, samples):
    return [w for w in (ring.push(s) for s in samples) if w is not None]


# ---------------------------------------------------------------- ring buffer (matches example)
def test_ring_window_has_pre_and_post_around_rising_edge():
    ring = SwingRing(pre_frames=50, post_frames=40, threshold=10.0)
    sig = [0.0] * 80 + [20.0] * 30 + [0.0] * 60
    wins = _push_all(ring, _stream(sig))
    assert len(wins) == 1
    w = wins[0]
    assert len(w.samples) == 90 and w.trigger_index == 50
    assert np.linalg.norm(w.samples[50]["gyro"]) > 10 >= np.linalg.norm(w.samples[49]["gyro"])
    assert w.t_trigger == pytest.approx(80 * DT)


def test_ring_needs_full_preroll_and_quiet_sample_before_new_edge():
    ring = SwingRing(pre_frames=50, post_frames=20, threshold=10.0)
    # spike before pre-roll is full -> ignored; held-high after a window -> no retrigger
    sig = [0.0] * 10 + [20.0] * 5 + [0.0] * 60 + [20.0] * 200
    wins = _push_all(ring, _stream(sig))
    assert len(wins) == 1
    # going quiet again re-arms; next edge triggers
    wins = _push_all(ring, _stream([0.0] * 5 + [20.0] * 30, t0=275 * DT))
    assert len(wins) == 1


def test_ring_resets_on_gap_and_ignores_duplicates():
    ring = SwingRing(pre_frames=50, post_frames=20, threshold=10.0)
    s = _stream([0.0] * 60)
    _push_all(ring, s)
    assert ring.push(dict(s[-1])) is None                  # duplicate timestamp ignored
    assert len(ring.pre) == 50
    gap = _stream([0.0] * 5, t0=60 * DT + 0.01)            # 10 ms gap > 3.75 ms
    _push_all(ring, gap)
    assert ring.discontinuities == 1 and len(ring.pre) == 5


def test_camera_ring_cut():
    cr = CameraRing(seconds=1, fps=100)
    for i in range(150):
        cr.push({"ts_ns_image": int(i * 1e7)})
    got = cr.cut(0.995, 1.205)
    assert len(got) == 21 and cr.latest_t() == pytest.approx(1.49)


# ---------------------------------------------------------------- protocol v2 framing
def _fake_protocol():
    class Rec:
        def __init__(self, payload):
            v = np.frombuffer(payload[:24], "<f4").astype(float)
            self.icm45686_accel_mps2, self.icm45686_gyro_rad_s = v[:3], v[3:6]
            self.icm_samd_timestamp_unwrapped_us = int.from_bytes(payload[24:28], "little")
            self.impact_flag = payload[28]
            self.icm_valid = True
            self.status_word = 0
            self.tx_timestamp_event = "ok"

    class Dec:
        def decode(self, payload, diag):
            return Rec(payload)
    return types.SimpleNamespace(ProtocolV2Decoder=Dec)


def _frame(t_us, gyro=(0, 0, 0), impact=0, bad_crc=False):
    payload = (np.r_[[0, 0, 9.81], gyro].astype("<f4").tobytes()
               + int(t_us).to_bytes(4, "little") + bytes([impact, 0]))
    assert len(payload) == 30
    diag = bytes(8)
    crc = v2._crc16_ccitt_false(payload + diag) ^ (1 if bad_crc else 0)
    fr = v2.HEADER + payload + diag + crc.to_bytes(2, "big")
    assert len(fr) == v2.FRAME_SIZE
    return fr


def test_crc16_ccitt_false_known_value():
    assert v2._crc16_ccitt_false(b"123456789") == 0x29B1


def test_v2_parser_resyncs_and_drops_bad_crc():
    p = v2.V2Parser(protocol=_fake_protocol())
    data = b"\x00\x13" + _frame(1000) + _frame(3500, bad_crc=True) + _frame(6000, (0, 0, 12), impact=1)
    out = p.feed(data[:30]) + p.feed(data[30:])
    assert [s.tx_us for s in out] == [1000, 6000]
    assert out[1].impact_flag == 1 and out[1].gyro[2] == pytest.approx(12)
    assert p.crc_failures == 1
    assert v2.looks_like_v2(data)


def test_v2_parser_without_serial_protocol_explains(monkeypatch):
    monkeypatch.setattr(v2, "load_serial_protocol", lambda: None)
    with pytest.raises(RuntimeError, match="serial_protocol.py"):
        v2.V2Parser()


# ---------------------------------------------------------------- web app (demo mode)
@pytest.fixture
def app(tmp_path, monkeypatch):
    import app.backend as be
    monkeypatch.setattr(be, "DATA", str(tmp_path))
    monkeypatch.setattr(be, "SWINGS", str(tmp_path / "swings"))
    import app.server as sv
    monkeypatch.setattr(sv, "SWINGS", str(tmp_path / "swings"))
    from http.server import ThreadingHTTPServer
    backend = be.Backend()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), sv.make_handler(backend, restart=lambda: None))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield backend, "http://127.0.0.1:%d" % httpd.server_address[1]
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read()


def test_app_demo_swing_saved_and_viewable(app):
    backend, base = app
    sid = backend.demo_swing()
    assert sid
    swings = json.loads(_get(base + "/api/swings")[1])
    assert swings[0]["id"] == sid and swings[0]["status"] == "ok"
    assert abs(swings[0]["attack_angle_deg"]) < 40 and -60 < swings[0]["vertical_bat_angle_deg"] < 10
    code, page = _get(base + "/swing/" + sid)
    assert code == 200 and b"Attack angle" in page and b"Vertical bat angle" in page
    code, png = _get(base + f"/files/{sid}/swing.png")
    assert code == 200 and png[:4] == b"\x89PNG"
    meta = json.loads(_get(base + f"/files/{sid}/meta.json")[1])
    assert meta["imu_frames"] == 2400                         # 1200 before + 1200 after trigger
    st = json.loads(_get(base + "/api/status")[1])
    assert st["mode"] == "idle" and "10 rad/s" in st["trigger"]
    req = urllib.request.Request(base + "/api/delete/" + sid, method="POST")
    assert json.loads(urllib.request.urlopen(req).read())["ok"]
    assert json.loads(_get(base + "/api/swings")[1]) == []


def test_app_files_route_blocks_path_traversal(app):
    _, base = app
    with pytest.raises(urllib.error.HTTPError):
        _get(base + "/files/../../dckft/config.py")
