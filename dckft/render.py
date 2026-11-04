"""
render.py -- 3D swing replay window (matplotlib; runs on the Jetson with no extra installs).

Layout: 3D bat/path plus a speed chart. Pink is camera-supported fusion;
orange is an IMU/model estimate during missing optical support.

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
    """Continuous estimated replay, separate from supported speed; stop at swing end."""
    import matplotlib
    if not rcfg.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    rp = getattr(result, "replay", None)
    tr, sw, m = (rp.traj if rp is not None else result.traj), result.swing, result.metrics
    t = tr.t
    first = int(np.clip(np.searchsorted(t, sw.t_onset-.10), 0, len(t)-1))
    last = int(np.clip(np.searchsorted(t, sw.t_end, side="right")-1, first, len(t)-1))
    if last <= first:
        raise ValueError("Swing window is too short to render")
    valid = np.ones(len(t), bool) if tr.valid is None else tr.valid.copy()
    active = (t >= sw.t_onset) & (t <= sw.t_end)
    speed = tr.speed(bat.sweet_spot_s_m) * MPS_TO_MPH
    finite = (np.isfinite(tr.knob).all(axis=1) & np.isfinite(tr.axis).all(axis=1)
              & np.isfinite(speed))
    visible = finite if rp is not None else valid & finite
    B = _display_basis(result.up)
    origin = tr.knob[first]
    D = lambda p: (np.asarray(p)-origin) @ B.T
    knob, tip = D(tr.knob), D(tr.point(bat.length_m))
    path = D(tr.point(bat.sweet_spot_s_m))
    path[~(visible & active)] = np.nan
    trel = t-sw.t_onset
    # Never replace the speed chart with the derivative of a display-only gap bridge.
    supported_speed = result.traj.speed(bat.sweet_spot_s_m) * MPS_TO_MPH
    shown_speed = np.where(valid & active & finite, supported_speed, np.nan)

    fig = plt.figure(figsize=(11, 7))
    if fig.canvas.manager and hasattr(fig.canvas.manager, "set_window_title"):
        fig.canvas.manager.set_window_title(title)
    ax = fig.add_axes([.02,.28,.64,.65], projection="3d")
    axs = fig.add_axes([.69,.36,.28,.42])
    ax.plot(*path[first:last+1].T, color="darkorange", lw=2, ls="--", label="IMU/model gap estimate")
    measured_path = path.copy()
    measured_path[~valid] = np.nan
    ax.plot(*measured_path[first:last+1].T, color="deeppink", lw=2, label="Camera-supported fusion")
    ax.legend(loc="lower left", fontsize=8)
    points = np.vstack([knob[first:last+1][visible[first:last+1]],
                        tip[first:last+1][visible[first:last+1]]])
    if not len(points):
        points = np.vstack([knob[first:last+1], tip[first:last+1]])
    _set_equal(ax, points)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("up (m)")
    ax.view_init(elev=18, azim=-60)
    line, = ax.plot([], [], [], color="0.2", lw=5)
    dots, = ax.plot([], [], [], "o", color="deeppink", ms=6)
    status = ax.text2D(.03,.95,"",transform=ax.transAxes,fontsize=11)
    axs.plot(trel[first:last+1], shown_speed[first:last+1], color="deeppink", lw=2)
    axs.set_xlabel("Time from swing start (s)"); axs.set_ylabel("Estimated barrel speed (mph)")
    axs.set_xlim(trel[first], trel[last]); axs.grid(alpha=.25)
    cursor = axs.axvline(0, color="0.3")
    speed_label = "Estimated speed at impact" if m.contact_detected else "Estimated peak barrel speed"
    headline = (speed_label + ": %.1f mph" % m.bat_speed_mph
                if m.speed_reliable else "Barrel speed: unavailable (tracking gap)")
    fig.suptitle(headline, fontsize=15, weight="bold")
    import textwrap
    footer = ("Dry swing: no impact measured. " if not m.contact_detected else "Impact detected. ")
    footer += "Orange: estimated gap motion, excluded from speed metrics. Wrist flex/drift can affect estimates. "
    footer += "Replay ends with the swing. Calibration RMS: %.1f deg. " % result.calib.rms_axis_deg
    footer += m.speed_note
    fig.text(.07,.20,textwrap.fill(footer,120),fontsize=9,color="0.35",va="top")

    def draw(k):
        k = int(np.clip(k, first, last))
        if visible[k]:
            estimated = not valid[k]
            color = "darkorange" if estimated else "deeppink"
            line.set_color(color if estimated else "0.2")
            line.set_linestyle("--" if estimated else "-")
            dots.set_color(color)
            line.set_data_3d(*np.vstack([knob[k],tip[k]]).T)
            p = np.array([D(tr.point(s))[k] for s in bat.marker_s_m.values()])
            dots.set_data_3d(*p.T)
            if estimated:
                status.set_text("%.3f s | IMU/model estimate\n%.0f ms from camera support | speed withheld" %
                                (trel[k], 1000*rp.nearest_support_s[k]))
            else:
                status.set_text("%.3f s | camera-supported %.1f mph" % (trel[k], supported_speed[k]))
        else:
            line.set_data_3d([],[],[]); dots.set_data_3d([],[],[])
            status.set_text("%.3f s | Tracking gap - pose/speed not shown" % trel[k])
        cursor.set_xdata([trel[k],trel[k]])

    at = int(np.clip(np.searchsorted(t,m.t_contact),first,last))
    if m.contact_detected:
        at = max(first,at-max(1,int(round(.0025/np.median(np.diff(t))))))
    draw(at)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if rcfg.save_png:
            fig.savefig(os.path.join(out_dir,"swing.png"),dpi=110)
        if rcfg.save_mp4:
            dt = float(np.median(np.diff(t)))
            frames = np.unique(np.r_[np.arange(first,last,max(1,int(round(dt**-1*rcfg.playback_speed/30)))),last])
            _save_video(fig,draw,frames,os.path.join(out_dir,"swing.mp4"))
            draw(at)
    if not rcfg.show:
        plt.close(fig)
        return None
    slider = Slider(fig.add_axes([.22,.07,.64,.035]),"Time (s)",
                    float(trel[first]),float(trel[last]),valinit=float(trel[first]))
    btn = Button(fig.add_axes([.07,.06,.09,.055]),"Play")
    state = {"k":first,"playing":False}
    step = max(1,int(round(.03*rcfg.playback_speed/np.median(np.diff(t)))))
    def slide(v):
        state["k"] = int(np.clip(np.searchsorted(trel,v),first,last))
        draw(state["k"]); fig.canvas.draw_idle()
    slider.on_changed(slide)
    def toggle(_=None):
        if state["k"] >= last:
            state["k"] = first
        state["playing"] = not state["playing"]
        btn.label.set_text("Pause" if state["playing"] else "Play")
    btn.on_clicked(toggle)
    fig.canvas.mpl_connect("key_press_event",lambda e: toggle() if e.key == " " else None)
    timer = fig.canvas.new_timer(interval=30)
    def tick():
        if not state["playing"]:
            return
        state["k"] = min(last,state["k"]+step)
        slider.set_val(float(trel[state["k"]]))
        if state["k"] == last:
            state["playing"] = False
            btn.label.set_text("Replay")
    timer.add_callback(tick); timer.start()
    fig._dckft_refs = (slider,btn,timer)
    draw(first); toggle()
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

