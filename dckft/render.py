"""
render.py -- 3D swing replay window (matplotlib; runs on the Jetson with no extra installs).

Layout:
  left   3D view: bat at the current instant, sweet-spot path coloured by speed, the contact
         bat (red), and the fitted swing plane
  right  headline metrics + coaching notes
  bottom bat-speed / hand-speed vs time, with a cursor that follows playback

Controls: Play/Pause button, time slider, mouse-drag to orbit. Space = play/pause.

    python -m dckft.run process --cam sessions/x/camera.jsonl --imu sessions/x/imu.csv
    (render_swing() is called for you; use --no-show on a headless box to only write PNGs)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .config import BatConfig, RenderConfig
from .metrics import MPS_TO_MPH


def _display_basis(up: np.ndarray) -> np.ndarray:
    """Rows = display x, y, z expressed in camera coords. z = up, x = camera-right levelled."""
    z = up / np.linalg.norm(up)
    x = np.array([1.0, 0, 0]) - z * z[0]
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def _set_equal(ax, pts):
    c = (pts.max(0) + pts.min(0)) / 2
    r = (pts.max(0) - pts.min(0)).max() / 2 * 1.1
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def render_swing(result, bat: BatConfig, rcfg: RenderConfig = RenderConfig(),
                 out_dir: Optional[str] = None, title: str = "DC-KFT swing"):
    import matplotlib
    if not rcfg.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    tr, sw, m = result.traj, result.swing, result.metrics
    B = _display_basis(result.up)
    t = tr.t
    k_on = int(np.searchsorted(t, sw.t_onset))
    k_c = int(np.clip(np.searchsorted(t, m.t_contact), 0, len(t) - 1))
    origin = tr.knob[k_on]
    D = lambda p: (np.asarray(p) - origin) @ B.T
    knob = D(tr.knob)
    tip = D(tr.point(bat.length_m))
    ss = D(tr.point(bat.sweet_spot_s_m))
    speed_mph = tr.speed(bat.sweet_spot_s_m) * MPS_TO_MPH
    hand_mph = tr.speed(bat.hands_s_m) * MPS_TO_MPH
    k_show0 = max(0, k_on - int(0.15 / np.median(np.diff(t))))
    k_show1 = len(t) - 1
    trel = t - m.t_contact

    fig = plt.figure(figsize=(14, 8.5))
    fig.canvas.manager.set_window_title(title) if hasattr(fig.canvas, "manager") and fig.canvas.manager else None
    ax = fig.add_axes([0.0, 0.25, 0.62, 0.72], projection="3d")
    axm = fig.add_axes([0.63, 0.25, 0.36, 0.72]); axm.axis("off")
    axs = fig.add_axes([0.06, 0.07, 0.56, 0.15])

    # sweet-spot path, coloured by speed (onset -> contact), follow-through faded
    seg = np.stack([ss[k_on:k_c], ss[k_on + 1:k_c + 1]], axis=1)
    lc = Line3DCollection(seg, cmap="plasma", linewidths=3)
    lc.set_array(speed_mph[k_on:k_c])
    lc.set_clim(0, max(80.0, float(np.nanmax(speed_mph[k_on:k_c + 1]))))
    ax.add_collection3d(lc)
    ax.plot(*ss[k_c:k_show1].T, color="0.6", lw=1.5, ls="--")
    ax.plot(*knob[k_show0:k_show1].T, color="0.35", lw=1)
    cb = fig.colorbar(lc, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label("sweet-spot speed (mph)")

    # contact bat + swing plane
    ax.plot(*np.vstack([knob[k_c], tip[k_c]]).T, color="crimson", lw=4, alpha=0.6, label="contact")
    path = ss[k_on:k_c + 1]
    if len(path) >= 3:
        c = path.mean(0)
        _, _, vt = np.linalg.svd(path - c)
        e1, e2 = vt[0], vt[1]
        r = np.abs((path - c) @ e1).max()
        g = np.linspace(-1, 1, 2)
        P = c + r * (g[:, None, None] * e1 + 0.6 * g[None, :, None] * e2)
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color="tab:blue", alpha=0.08)

    pts = np.vstack([knob[k_show0:k_show1], tip[k_show0:k_show1]])
    _set_equal(ax, pts)
    zmin = pts[:, 2].min()
    gx, gy = np.meshgrid(np.linspace(*ax.get_xlim(), 6), np.linspace(*ax.get_ylim(), 6))
    ax.plot_wireframe(gx, gy, np.full_like(gx, zmin), color="0.85", lw=0.5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("up (m)")
    ax.view_init(elev=18, azim=-60)

    bat_line, = ax.plot([], [], [], color="saddlebrown", lw=6, solid_capstyle="round")
    knob_dot, = ax.plot([], [], [], "o", color="k", ms=6)
    time_txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=11)

    # metrics panel
    rows = [
        ("Bat speed at contact", f"{m.bat_speed_mph:.1f} mph"),
        ("Peak bat speed", f"{m.peak_bat_speed_mph:.1f} mph"),
        ("Peak hand speed", f"{m.peak_hand_speed_mph:.1f} mph"),
        ("Attack angle", f"{m.attack_angle_deg:+.1f}°"),
        ("Vertical bat angle", f"{m.vertical_bat_angle_deg:+.1f}°"),
        ("Time to contact", f"{m.time_to_contact_s * 1e3:.0f} ms"),
        ("Swing plane tilt", f"{m.swing_plane_tilt_deg:.0f}°"),
        ("Peak wrist rate", f"{m.peak_wrist_rate_dps:.0f} °/s"),
    ]
    y = 0.98
    axm.text(0, y, "Swing metrics", fontsize=15, weight="bold", va="top"); y -= 0.07
    for name, val in rows:
        axm.text(0.0, y, name, fontsize=11, va="top", color="0.3")
        axm.text(0.98, y, val, fontsize=12, va="top", ha="right", weight="bold")
        y -= 0.052
    if not m.contact_detected:
        axm.text(0, y, "(no impact detected: dry swing, contact = peak speed)", fontsize=8,
                 color="0.4", va="top"); y -= 0.04
    y -= 0.02
    axm.text(0, y, "Coaching", fontsize=15, weight="bold", va="top"); y -= 0.06
    import textwrap
    for note in result.notes:
        wrapped = textwrap.fill(note, 52)
        axm.text(0, y, "• " + wrapped, fontsize=10, va="top")
        y -= 0.035 * (wrapped.count("\n") + 1) + 0.02
    d = result.diagnostics()
    axm.text(0, 0.0, f"sync {d['time_offset_ms']:+.1f} ms · camera coverage "
                     f"{100 * d['camera_coverage']:.0f}% · fused in {d['timings_ms'].get('total', 0):.0f} ms",
             fontsize=8, color="0.45")

    # speed strip
    sl = slice(k_show0, k_show1)
    axs.plot(trel[sl] * 1e3, speed_mph[sl], label="sweet spot")
    axs.plot(trel[sl] * 1e3, hand_mph[sl], label="hands")
    axs.axvline(0, color="crimson", lw=1)
    axs.set_xlabel("time from contact (ms)"); axs.set_ylabel("mph")
    axs.legend(loc="upper left", fontsize=8); axs.grid(alpha=0.3)
    cursor = axs.axvline(trel[k_show0] * 1e3, color="k", lw=1)

    def draw(k):
        k = int(np.clip(k, k_show0, k_show1))
        bat_line.set_data_3d(*np.vstack([knob[k], tip[k]]).T)
        knob_dot.set_data_3d([knob[k, 0]], [knob[k, 1]], [knob[k, 2]])
        time_txt.set_text(f"t = {trel[k] * 1e3:+6.1f} ms   {speed_mph[k]:5.1f} mph")
        cursor.set_xdata([trel[k] * 1e3, trel[k] * 1e3])

    draw(k_c)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if rcfg.save_png:
            fig.savefig(os.path.join(out_dir, "swing.png"), dpi=110)
        if rcfg.save_mp4:
            _save_video(fig, draw, range(k_show0, k_show1, 4), os.path.join(out_dir, "swing.mp4"))

    if not rcfg.show:
        plt.close(fig)
        return None

    # interactive playback
    axsl = fig.add_axes([0.70, 0.12, 0.25, 0.03])
    axbtn = fig.add_axes([0.70, 0.05, 0.08, 0.05])
    slider = Slider(axsl, "t (ms)", trel[k_show0] * 1e3, trel[k_show1] * 1e3, valinit=0.0)
    btn = Button(axbtn, "Play")
    state = {"k": k_show0, "playing": False}
    dt_real = np.median(np.diff(t))
    step = max(1, int(round(0.03 * rcfg.playback_speed / dt_real)))

    def on_slide(v):
        state["k"] = int(np.searchsorted(trel, v / 1e3))
        draw(state["k"]); fig.canvas.draw_idle()

    slider.on_changed(on_slide)

    def toggle(_=None):
        state["playing"] = not state["playing"]
        btn.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"] and state["k"] >= k_show1 - step:
            state["k"] = k_show0

    btn.on_clicked(toggle)
    fig.canvas.mpl_connect("key_press_event", lambda e: toggle() if e.key == " " else None)
    timer = fig.canvas.new_timer(interval=30)

    def tick():
        if state["playing"]:
            state["k"] += step
            if state["k"] >= k_show1:
                state["k"] = k_show1; toggle()
            slider.eventson = False
            slider.set_val(trel[state["k"]] * 1e3)
            slider.eventson = True
            draw(state["k"]); fig.canvas.draw_idle()

    timer.add_callback(tick)
    timer.start()
    fig._dckft_refs = (slider, btn, timer)          # keep widgets alive
    toggle()                                        # auto-play once
    return fig


def _save_video(fig, draw, frames, path):
    try:
        from matplotlib.animation import FFMpegWriter
        w = FFMpegWriter(fps=30)
        with w.saving(fig, path, dpi=90):
            for k in frames:
                draw(k)
                w.grab_frame()
    except Exception as e:      # ffmpeg missing etc.
        print(f"[render] could not write {path}: {e}")
