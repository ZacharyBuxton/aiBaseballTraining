import zmq, time, json, sys
import numpy as np
import os

#ALWAYS BEGIN TO RUN THIS BEFORE TRACKING SCHEME**********
JETSON_IP = "10.132.52.95"
ZMQ_PORT = 5557
ZMQ_ADDR = f"tcp://{JETSON_IP}:{ZMQ_PORT}"
ANCHOR_FROM_CAM0 = None
ANCHOR_MIN_POINTS = 3 #require three points
# -------- ZeroMQ setup --------
ctx = zmq.Context.instance()
sock = ctx.socket(zmq.PULL)
sock.connect("tcp://10.132.52.95:5557") #"tcp://<jetson_ip>:5557" //laptop: 10.132.22.62

poller = zmq.Poller()
poller.register(sock, zmq.POLLIN)

# -------- Output log --------
import os

# Directory where THIS script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# One persistent file for all future quaternion outputs
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "changing_outputs.jsonl")

# Open in append mode: keep existing data, add new runs to the end
out = open(OUTPUT_PATH, "a")

print("Logging quaternion output to:", OUTPUT_PATH)


np.set_printoptions(precision=6, suppress=True, floatmode="fixed")

# ================================================================
# 1) DEFINE YOUR FOUR LOCAL (MODEL) COORDINATES HERE (one per color)
#    Replace these placeholder coordinates with your CAD/eDrawings values.
#    Units can be arbitrary but must be consistent with your world units.
# ================================================================
MODEL = {
    "neon_yellow": np.array([0.0, 0.03699186, 0.0]),
    "neon_pink":   np.array([0.0, 0.01294052, -0.03463435]),
    "neon_green":  np.array([-0.02999423, 0.01294052, 0.01731717]),
    "neon_blue":   np.array([0.02999422, 0.01294051, 0.01731718]),
}
COLOR_ORDER = ["neon_yellow", "neon_pink", "neon_green", "neon_blue"]  # deterministic use-order

# ================================================================
# 2) Build world points dict from an incoming payload.
#    Returns (W_dict, used_colors):
#       - W_dict: {color: np.array([x,y,z])} for colors present & finite
#       - used_colors: ordered list of colors present from COLOR_ORDER
# ================================================================
def world_from_payload_any(data):
    pts = data.get("points", [])
    m = {}
    for p in pts:
        try:
            cid = p["id"]            # e.g., "neon_yellow"
            pos = p["pos"]
            x, y, z = float(pos["x"]), float(pos["y"]), float(pos["z"])
            if np.isfinite([x, y, z]).all():
                m[cid] = np.array([x, y, z], dtype=float)
        except Exception:
            continue

    present = [c for c in COLOR_ORDER if c in m]
    return m, present

# ================================================================
# 3) Horn (Davenport q-method) — works for N >= 3
# ================================================================
def horn_quaternion(B, W):
    cb = B.mean(axis=0); cw = W.mean(axis=0)
    X = B - cb;          Y = W - cw
    M = X.T @ Y
    Sxx,Sxy,Sxz = M[0]; Syx,Syy,Syz = M[1]; Szx,Szy,Szz = M[2]
    N = np.array([
        [ Sxx+Syy+Szz,      Syz-Szy,         Szx-Sxz,         Sxy-Syx],
        [ Syz-Szy,          Sxx-Syy-Szz,     Sxy+Syx,         Szx+Sxz],
        [ Szx-Sxz,          Sxy+Syx,        -Sxx+Syy-Szz,     Syz+Szy],
        [ Sxy-Syx,          Szx+Sxz,         Syz+Szy,        -Sxx-Syy+Szz]
    ], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(N)
    q = eigvecs[:, np.argmax(eigvals)]
    q = q / np.linalg.norm(q)
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)]
    ], dtype=float)
    t = cw - R @ cb
    return q, R, t

# ================================================================
# 4) Main loop: uses 4 or 3 visible markers; skips frame if < 3
# ================================================================
def main():
    print("QuatCalc: starting")
    print("  connecting to {ZMQ_ADDR}")
    print(f"  python: {sys.executable}")
    print(f"  cwd:    {os.getcwd()}")
    print("FULL OUTPUT PATH:", os.path.abspath("quat_output_quat.jsonl"))


    last_heartbeat = 0

    while True:
        events = dict(poller.poll(1000))  # wait up to 1s for a message
        if events.get(sock) == zmq.POLLIN:
            data = sock.recv_json()
            print("\nquatchat: received payload")
            camera_id = data.get("camera_id", "cam_0")

            # ---- collect any visible world points for known colors ----
            Wdict, present = world_from_payload_any(data)
            # we want 4 if available, otherwise any 3; skip if < 3
            avail = [c for c in COLOR_ORDER if c in present]
            if len(avail) < 3:
                print(f"  only {len(avail)} visible ({avail}) → skipping")
                continue

            #Translation shared origin anchored by cam0
            global ANCHOR_FROM_CAM0 #does not map camera1 on to camera0 like we need, will need to calibrate when we have our final camera setup in place

            def _mean_world_point(Wd, colors): #get a robust reference position from current points
                arr = np.stack([Wd[c] for c in colors], axis=0)
                return arr.mean(axis=0)

            if ANCHOR_FROM_CAM0 is None and camera_id == "cam_0" and len(avail) >= ANCHOR_MIN_POINTS:
                ANCHOR_FROM_CAM0 = _mean_world_point(Wdict, avail)
                print(" [shared-origin] anchor set from cam_0:", np.round(ANCHOR_FROM_CAM0, 6))

            if ANCHOR_FROM_CAM0 is not None:
                for c in avail:
                    Wdict[c] = Wdict[c] - ANCHOR_FROM_CAM0

            # Build matched sets (use all 4 if present, otherwise the first 3 by COLOR_ORDER)
            use_colors = avail if len(avail) >= 4 else avail[:3]
            B = np.stack([MODEL[c] for c in use_colors], axis=0)   # (N,3)
            W = np.stack([Wdict[c] for c in use_colors], axis=0)   # (N,3)

            # ---- solve pose ----
            q, R, t = horn_quaternion(B, W)
            print("  used colors:", use_colors)
            print("  q (w,x,y,z):", np.round(q, 6))
            print("  t:", np.round(t, 6))

            # ---- reprojection error (per-point & RMS) ----
            pred = (R @ B.T).T + t
            errs = np.linalg.norm(pred - W, axis=1)
            rms = float(np.sqrt((errs**2).mean()))
            print("  per-point err:", np.round(errs, 6), "  RMS:", f"{rms:.6f}")

            # ---- build processed record for JSONL log ----
            record = {
                "camera_id": camera_id,
                "schema_version": data.get("schema_version", None),
                "coord_sys": data.get("coord_sys", None),

                "ts_ns_wall": data.get("ts_ns_wall", None),
                "ts_ns_mono": data.get("ts_ns_mono", None),
                "frame_id": data.get("frame_id", None),

                "anchor_from_cam0": (
                    ANCHOR_FROM_CAM0.tolist()
                    if ANCHOR_FROM_CAM0 is not None else None
                ),

                "used_colors": use_colors,
                "quaternion_wxyz": q.tolist(),
                "rotation_matrix": R.tolist(),
                "translation_vector": t.tolist(),

                "per_point_errors": errs.tolist(),
                "rms_error": rms,

                "model_points": {c: MODEL[c].tolist() for c in use_colors},
                "world_points": {c: Wdict[c].tolist() for c in use_colors},

                "raw_payload": data,
            }

            # ---- write the record ----
            out.write(json.dumps(record) + "\n")
            out.flush()

            print("---- waiting for next payload ----")

        else:
            now = time.time()
            if now - last_heartbeat >= 1.0:
                print("waiting for payload...", flush=True)
                last_heartbeat = now

if __name__ == "__main__":
    main()
