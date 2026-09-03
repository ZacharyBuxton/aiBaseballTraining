import cv2 as cv
import numpy as np
import math
import time
import threading
from collections import deque
import queue
import zmq
import json

# ---------- Azure Kinect ----------
import pykinect_azure as pykinect
from pykinect_azure import (
    K4A_CALIBRATION_TYPE_COLOR,
    K4A_CALIBRATION_TYPE_DEPTH,
    k4a_float2_t,
)

cv.setUseOptimized(True)
cv.setNumThreads(16)

# ==================== CONFIG ====================
COLOR_RES  = pykinect.K4A_COLOR_RESOLUTION_720P
DEPTH_MODE = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED

HAVE_SYNC_CABLE = False
SUBORD_DELAY_US = 16667

# Detection tuning
FAR_MODE = True
USE_ROI_TRACK = True
ROI_RADIUS = 120

DETECTION_SCALE = 0.33

H_TOL = 10
S_MIN, S_MAX = 150, 255
V_MIN, V_MAX = 120, 255

FAR_MIN_AREA_FRAC = 2e-5
MIN_AREA_ABS_FLOOR = 8.0

FONT = cv.FONT_HERSHEY_SIMPLEX

NEON_HUES = {
    "neon_pink":   [165, 175, 5, 15],
    "neon_green":  [55, 65, 75],
    "neon_yellow": [28, 25, 30],
    "sky_blue":    [105, 95, 115],
}
DRAW_COLORS = {
    "neon_pink":   (255, 0, 255),
    "neon_green":  (57, 255, 20),
    "neon_yellow": (0, 255, 255),
    "sky_blue":    (255, 191, 0),
}

KERNEL_SIZE = (3, 3) if FAR_MODE else (5, 5)
KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, KERNEL_SIZE)

# ==================== ZMQ PUBLISHER ====================
class CoordinatePublisher:
    def __init__(self, port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        print(f"Publishing coordinates on port {port}")
        time.sleep(0.5)  # Allow time for socket binding

    def publish(self, camera_id, color_name, x_m, y_m, z_m, timestamp):
        """Publish 3D coordinates with camera ID and timestamp"""
        message = {
            "camera_id": camera_id,
            "color": color_name,
            "position": {
                "x": x_m,
                "y": y_m,
                "z": z_m
            },
            "timestamp": timestamp
        }
        self.socket.send_string(json.dumps(message))

    def close(self):
        self.socket.close()
        self.context.term()

# ==================== HELPERS ====================
class FPSMeter:
    def __init__(self, window=30):
        self.t_last = time.perf_counter()
        self.samples = deque(maxlen=window)
    def tick(self):
        t = time.perf_counter()
        self.samples.append(t - self.t_last)
        self.t_last = t
    def fps(self):
        if len(self.samples) < 2:
            return 0.0
        avg = sum(self.samples) / len(self.samples)
        return (1.0 / avg) if avg > 0 else 0.0

def build_ranges_for_hue(h_center, h_tol, s_min, s_max, v_min, v_max):
    lo_h = h_center - h_tol
    hi_h = h_center + h_tol
    if lo_h < 0:
        return [
            (np.array([0, s_min, v_min]),           np.array([hi_h,        s_max, v_max])),
            (np.array([180 + lo_h, s_min, v_min]),  np.array([179,         s_max, v_max])),
        ]
    elif hi_h > 179:
        return [
            (np.array([lo_h, s_min, v_min]),        np.array([179,         s_max, v_max])),
            (np.array([0,    s_min, v_min]),        np.array([hi_h - 180,  s_max, v_max])),
        ]
    else:
        return [(np.array([lo_h, s_min, v_min]),    np.array([hi_h,        s_max, v_max]))]

def make_color_ranges():
    ranges = {}
    for name, centers in NEON_HUES.items():
        spans = []
        for c in centers:
            spans.extend(build_ranges_for_hue(c, H_TOL, S_MIN, S_MAX, V_MIN, V_MAX))
        ranges[name] = spans
    return ranges

COLOR_RANGES = make_color_ranges()

def clamp_roi(x, y, r, W, H):
    x0 = max(0, x - r); y0 = max(0, y - r)
    x1 = min(W, x + r); y1 = min(H, y + r)
    return x0, y0, x1, y1

def choose_best_blob(blobs, pred):
    if not blobs:
        return None
    if pred is None:
        return max(blobs, key=lambda b: b["area"])
    px, py = pred
    return min(blobs, key=lambda b: (b["center"][0]-px)**2 + (b["center"][1]-py)**2)

def safe_sample_depth(depth_img, x, y, win=3):
    H, W = depth_img.shape[:2]
    if x < 0 or y < 0 or x >= W or y >= H:
        return 0
    d = int(depth_img[y, x])
    if d > 0:
        return d
    x0, y0 = max(0, x - win), max(0, y - win)
    x1, y1 = min(W, x + win + 1), min(H, y + win + 1)
    patch = depth_img[y0:y1, x0:x1]
    vals = patch[patch > 0]
    if vals.size == 0:
        return 0
    return int(np.median(vals))

def pixel_to_3d(device, px, py, depth_mm):
    if depth_mm <= 0:
        return None
    try:
        pixels = k4a_float2_t((float(px), float(py)))
        pos3d = device.calibration.convert_2d_to_3d(
            pixels, depth_mm, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR
        )
        if pos3d is None:
            return None
        return (pos3d.xyz.x, pos3d.xyz.y, pos3d.xyz.z)
    except Exception:
        return None

def blobs_from_mask_fast(mask, min_area):
    num, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, num):
        area = float(stats[i, cv.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = stats[i, :4]
        cx, cy = centroids[i]
        blobs.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "center": (int(round(cx)), int(round(cy))),
            "area": area
        })
    return blobs

def detect_color_blobs_optimized(bgr_small, color_ranges, last_pos, last_vel, scale, use_roi=USE_ROI_TRACK):
    hsv = cv.cvtColor(bgr_small, cv.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]
    out = {c: [] for c in color_ranges}

    dyn_min_area = max(MIN_AREA_ABS_FLOOR * (scale**2), FAR_MIN_AREA_FRAC * (W * H))

    for name, ranges in color_ranges.items():
        mask = None
        for lo, hi in ranges:
            temp_mask = cv.inRange(hsv, lo, hi)
            if mask is None:
                mask = temp_mask
            else:
                cv.bitwise_or(mask, temp_mask, mask)

        if mask is None:
            continue

        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, KERNEL, iterations=1)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, KERNEL, iterations=1)

        submask = mask
        offset = (0, 0)
        pred_center = None
        if use_roi and last_pos.get(name) is not None:
            lx, ly = last_pos[name]
            lx_s, ly_s = int(lx * scale), int(ly * scale)
            vx, vy = last_vel.get(name, (0, 0))
            vx_s, vy_s = int(vx * scale), int(vy * scale)
            px = int(round(lx_s + vx_s)); py = int(round(ly_s + vy_s))
            pred_center = (px, py)
            roi_r = int(ROI_RADIUS * scale)
            x0, y0, x1, y1 = clamp_roi(px, py, roi_r, W, H)
            submask = mask[y0:y1, x0:x1]
            offset = (x0, y0)

        blobs = blobs_from_mask_fast(submask, dyn_min_area)
        if not blobs:
            continue

        inv_scale = 1.0 / scale
        for b in blobs:
            cx, cy = b["center"]
            x, y, w, h = b["bbox"]
            b["center"] = (int((cx + offset[0]) * inv_scale), int((cy + offset[1]) * inv_scale))
            b["bbox"]   = (int((x + offset[0]) * inv_scale), int((y + offset[1]) * inv_scale),
                           int(w * inv_scale), int(h * inv_scale))
            b["area"] = b["area"] * (inv_scale ** 2)

        best = choose_best_blob(blobs, pred_center)
        if best is not None:
            out[name].append(best)

    return out

# ==================== CAPTURE THREAD ====================
class OptimizedCaptureThread(threading.Thread):
    def __init__(self, dev_index, wired_mode, delay_us=0):
        super().__init__(daemon=True)
        self.dev_index = dev_index
        self.wired_mode = wired_mode
        self.delay_us = delay_us
        self.device = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.has_new_frame = threading.Event()

    def open(self):
        cfg = pykinect.default_configuration
        cfg.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        cfg.color_resolution = COLOR_RES
        cfg.depth_mode = DEPTH_MODE
        cfg.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        cfg.wired_sync_mode = self.wired_mode
        cfg.subordinate_delay_off_master_usec = self.delay_us
        self.device = pykinect.start_device(config=cfg, device_index=self.dev_index)
        time.sleep(0.5)

    def run(self):
        self.running = True
        while self.running:
            cap = None
            try:
                cap = self.device.update()
                if cap is None:
                    continue

                okc, cimg = cap.get_color_image()
                okd, dimg = cap.get_transformed_depth_image()
                if not okc or not okd or cimg is None or dimg is None:
                    continue

                cimg_np = cimg.copy()
                dimg_np = dimg.copy()

                frame_data = (cimg_np, dimg_np)

                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                try:
                    self.frame_queue.put_nowait(frame_data)
                    self.has_new_frame.set()
                except queue.Full:
                    pass

            except Exception:
                time.sleep(0.005)
            finally:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass

    def get_latest(self):
        if self.has_new_frame.is_set():
            self.has_new_frame.clear()
            try:
                return self.frame_queue.get_nowait()
            except queue.Empty:
                return None
        return None

    def stop(self):
        self.running = False

# ==================== PROCESS & DISPLAY ====================
frame_count = 0

def process_and_display_frame(bgra, depth, cam_id, device, last_pos, last_vel, fps, publisher):
    global frame_count
    frame_count += 1

    H_full, W_full = bgra.shape[:2]
    H_small = int(H_full * DETECTION_SCALE)
    W_small = int(W_full * DETECTION_SCALE)

    bgr_view  = cv.cvtColor(bgra, cv.COLOR_BGRA2BGR)
    bgr_small = cv.resize(bgr_view, (W_small, H_small), interpolation=cv.INTER_LINEAR)

    detections = detect_color_blobs_optimized(
        bgr_small, COLOR_RANGES, last_pos[cam_id],
        last_vel[cam_id], DETECTION_SCALE, use_roi=USE_ROI_TRACK
    )

    should_print = (frame_count % 30 == 0)
    timestamp = time.time()

    for name, blobs in detections.items():
        if not blobs:
            continue
        blob = blobs[0]
        (cx, cy) = blob["center"]
        color = DRAW_COLORS[name]

        cx = max(0, min(W_full - 1, cx))
        cy = max(0, min(H_full - 1, cy))

        depth_mm = safe_sample_depth(depth, cx, cy, win=3)
        pos3d_mm = pixel_to_3d(device, cx, cy, depth_mm) if depth_mm > 0 else None

        if pos3d_mm is not None:
            Xm, Ym, Zm = pos3d_mm[0]/1000.0, pos3d_mm[1]/1000.0, pos3d_mm[2]/1000.0

            # Publish coordinates with camera ID
            publisher.publish(cam_id, name, Xm, Ym, Zm, timestamp)

            if should_print:
                print(f"Cam{cam_id} {name}: dist={depth_mm}mm, 3D=[{Xm:.3f}, {Ym:.3f}, {Zm:.3f}]m")

        cv.circle(bgr_view, (cx, cy), 6, color, -1)

        if last_pos[cam_id][name] is not None:
            lx, ly = last_pos[cam_id][name]
            last_vel[cam_id][name] = (cx - lx, cy - ly)
        last_pos[cam_id][name] = (cx, cy)

    for name in NEON_HUES:
        if not detections.get(name):
            last_pos[cam_id][name] = None
            vx, vy = last_vel[cam_id][name]
            last_vel[cam_id][name] = (int(vx * 0.8), int(vy * 0.8))

    cv.putText(bgr_view, f"Camera {cam_id} | FPS: {fps:.1f}", (10, 30), FONT, 0.7, (0, 255, 0), 2, cv.LINE_AA)
    cv.imshow("Dual Kinect", bgr_view)

# ==================== MAIN ====================
if __name__ == "__main__":
    print("Starting optimized dual Kinect tracking with ZMQ publisher...")
    print(f"Detection scale: {DETECTION_SCALE}x (processing at {int(DETECTION_SCALE*100)}% resolution)")
    pykinect.initialize_libraries()

    # Initialize publisher
    publisher = CoordinatePublisher(port=5555)

    cam0 = cam1 = None
    try:
        mode0 = pykinect.K4A_WIRED_SYNC_MODE_MASTER if HAVE_SYNC_CABLE else pykinect.K4A_WIRED_SYNC_MODE_STANDALONE
        cam0 = OptimizedCaptureThread(0, mode0, 0)
        cam0.open()
    except Exception as e:
        print(f"Failed to open camera 0: {e}")
        raise SystemExit(1)

    try:
        mode1 = pykinect.K4A_WIRED_SYNC_MODE_SUBORDINATE if HAVE_SYNC_CABLE else pykinect.K4A_WIRED_SYNC_MODE_STANDALONE
        delay = SUBORD_DELAY_US if HAVE_SYNC_CABLE else 0
        cam1 = OptimizedCaptureThread(1, mode1, delay)
        cam1.open()
    except Exception as e:
        print(f"Camera 1 unavailable: {e}")
        cam1 = None

    cam0.start()
    if cam1: cam1.start()

    last_pos = {0: {name: None for name in NEON_HUES},
                1: {name: None for name in NEON_HUES}}
    last_vel = {0: {name: (0, 0) for name in NEON_HUES},
                1: {name: (0, 0) for name in NEON_HUES}}

    cv.namedWindow("Dual Kinect", cv.WINDOW_NORMAL)
    fps_meter = FPSMeter(60)
    display_camera = 0

    try:
        while True:
            key = cv.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            frames_processed = []

            pkt0 = cam0.get_latest()
            if pkt0 is not None:
                bgra0, depth0 = pkt0
                frames_processed.append((bgra0, depth0, 0, cam0.device))

            if cam1 is not None:
                pkt1 = cam1.get_latest()
                if pkt1 is not None:
                    bgra1, depth1 = pkt1
                    frames_processed.append((bgra1, depth1, 1, cam1.device))

            if frames_processed:
                fps_meter.tick()
                current_fps = fps_meter.fps()

                for bgra, depth, cam_id, device in frames_processed:
                    if cam_id == display_camera:
                        process_and_display_frame(bgra, depth, cam_id, device, last_pos, last_vel, current_fps, publisher)
                        break
                else:
                    bgra, depth, cam_id, device = frames_processed[0]
                    process_and_display_frame(bgra, depth, cam_id, device, last_pos, last_vel, current_fps, publisher)

                display_camera = 1 if display_camera == 0 else 0

    finally:
        if cam0: cam0.stop()
        if cam1: cam1.stop()

        if cam0: cam0.join(timeout=2.0)
        if cam1: cam1.join(timeout=2.0)

        publisher.close()

        try:
            cv.destroyAllWindows()
        except Exception:
            pass

        try:
            if cam0 and cam0.device: cam0.device.stop_cameras(); cam0.device.close()
        except Exception:
            pass
        try:
            if cam1 and cam1.device: cam1.device.stop_cameras(); cam1.device.close()
        except Exception:
            pass
