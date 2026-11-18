"""
server.py -- a small web app for recording and reviewing swings (Python standard library only).

    python3 run_app.py                 # then open http://<orin-ip>:8000 from any browser

Pages: /  (record / simulate / update, swing list)   /swing/<id>  (render + angles + downloads)
API:   GET /api/status  /api/swings          POST /api/start  /api/stop  /api/demo
       POST /api/update  /api/delete/<id>    GET /files/<id>/<name>
"""

from __future__ import annotations

import html
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from .backend import SWINGS, Backend, git_info, git_pull

# Fixed content types: on Windows, mimetypes reads the registry, which can report wrong types
TYPES = {".png": "image/png", ".mp4": "video/mp4", ".json": "application/json",
         ".csv": "text/csv; charset=utf-8", ".jsonl": "text/plain; charset=utf-8",
         ".txt": "text/plain; charset=utf-8", ".log": "text/plain; charset=utf-8"}

STYLE = """
:root{--bg:#fafafa;--fg:#1a1a1a;--muted:#666;--card:#fff;--line:#ddd;--pink:#e0187a}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--muted:#999;--card:#1e1e1e;--line:#333}}
*{box-sizing:border-box}body{margin:0;font:15px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:980px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 4px}
.muted{color:var(--muted);font-size:13px}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:14px;margin:12px 0}button{font:inherit;padding:9px 14px;border-radius:8px;
border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer;margin:4px 6px 4px 0}
button.primary{background:var(--pink);border-color:var(--pink);color:#fff}button:disabled{opacity:.45;cursor:default}
table{width:100%;border-collapse:collapse}td,th{padding:7px 6px;border-bottom:1px solid var(--line);text-align:left}
th{font-size:12px;color:var(--muted);font-weight:600}tr.row{cursor:pointer}tr.row:hover{background:rgba(224,24,122,.07)}
.big{font-size:26px;font-weight:700}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
img{max-width:100%;border-radius:8px;border:1px solid var(--line);background:#fff}a{color:var(--pink)}
pre{white-space:pre-wrap;font-size:12px;margin:0}
@media (max-width:600px){.src{display:none}td,th{padding:6px 3px;font-size:13px}}
"""

INDEX = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>DC-KFT Swings</title><style>""" + STYLE + """</style></head><body><main>
<h1>DC-KFT swing recorder</h1><div class=muted id=git></div>
<div class=card>
  <div><b>Status:</b> <span id=msg>...</span></div>
  <div class=muted id=ring></div>
  <div style="margin-top:8px">
    <label>Camera mode <select id=captureMode>
      <option value="120fps">High FPS: 960 × 600, 120 FPS</option>
      <option value="1080p60">High quality: 1920 × 1080, 60 FPS</option>
    </select></label>
    <label>Clip length <select id=captureSeconds><option value="5">5 seconds</option>
      <option value="10">10 seconds</option></select></label><br>
    <button class=primary id=start onclick="post('/api/start')">Record one swing (camera + IMU)</button>
    <button id=stop onclick="post('/api/stop')">Stop</button>
    <button id=demo onclick="post('/api/demo')">Simulate swing (no hardware)</button>
    <button id=update onclick="update()">Update from GitHub</button>
  </div>
  <div class=muted id=trigger></div><div class=muted id=cfg></div>
</div>
<div class=card><b>Swings</b> <span class=muted>(click one to view)</span>
<table><thead><tr><th>Time</th><th class=src>Source</th><th>Attack angle</th><th>Vertical bat angle</th>
<th>Bat speed</th><th>Status</th></tr></thead><tbody id=swings></tbody></table></div>
<div class=card><b>Log</b><pre id=log></pre></div>
</main><script>
const f=(v,u)=>v===null||v===undefined?'n/a':(v>0&&u=='°'?'+':'')+v.toFixed(1)+u;
async function post(u){const body=u==='/api/start'?JSON.stringify({capture_mode:captureMode.value,seconds:Number(captureSeconds.value)}):'{}';const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body});const j=await r.json();if(j.message)document.getElementById('msg').textContent=j.message;tick()}
async function update(){if(!confirm('Pull the latest code from GitHub and restart the app?'))return;
 document.getElementById('msg').textContent='Updating...';const r=await fetch('/api/update',{method:'POST'});const j=await r.json();
 alert(j.output||j.message);setTimeout(()=>location.reload(),4000)}
async function tick(){try{
 const s=await (await fetch('/api/status')).json();
 msg.textContent=s.message+(s.busy?'  (processing...)':'');ring.textContent=s.ring?('Ring buffer: '+s.ring+'  |  frames: IMU '+s.frames.imu+', camera '+s.frames.camera):'';
 trigger.textContent='Record first → detect saved frames → camera-led replay with IMU gap estimates. '+s.trigger;cfg.textContent='Config: '+s.config;
 git.textContent='Code: '+s.git.commit+' ('+s.git.date+') '+s.git.subject;
 captureMode.disabled=captureSeconds.disabled=s.mode!='idle';start.disabled=s.mode!='idle';stop.disabled=s.mode!='hardware';demo.disabled=s.mode!='idle'||s.busy;update.disabled=s.mode!='idle'||s.busy;
 log.textContent=s.log.join('\\n');
 const w=await (await fetch('/api/swings')).json();
 swings.innerHTML=w.map(x=>`<tr class=row onclick="location='/swing/${x.id}'"><td>${x.created||x.id}</td><td class=src>${x.source||''}</td>
 <td>${f(x.attack_angle_deg,'°')}</td><td>${f(x.vertical_bat_angle_deg,'°')}</td><td>${f(x.bat_speed_mph,' mph')}</td><td>${x.status||''}</td></tr>`).join('')
 ||'<tr><td colspan=6 class=muted>No swings yet. Press Start recording, or Simulate swing.</td></tr>';
}catch(e){msg.textContent='(app not reachable)'}}
tick();setInterval(tick,1500);
</script></body></html>"""


def swing_page(d: dict) -> str:
    sid = html.escape(d["id"])
    meta = d.get("meta", {})
    sj = d.get("swing", {})
    m = sj.get("metrics", {})
    diag = sj.get("diagnostics", {})

    def deg(v):
        return "n/a" if v is None else f"{v:+.1f}°"

    def num(v, unit, fmt="{:.1f}"):
        return "n/a" if v is None else fmt.format(v) + unit

    cards = [("Attack angle", deg(m.get("attack_angle_deg")), "Blast target +6 to +14°"),
             ("Vertical bat angle", deg(m.get("vertical_bat_angle_deg")), "Blast college −23 to −33°"),
             ("Bat speed", num(m.get("bat_speed_mph"), " mph"), "sweet spot at impact"),
             ("Time to contact", num(None if m.get("time_to_contact_s") is None else
                                     m["time_to_contact_s"] * 1e3, " ms", "{:.0f}"), "")]
    grid = "".join(f"<div class=card><div class=muted>{a}</div><div class=big>{b}</div>"
                   f"<div class=muted>{c}</div></div>" for a, b, c in cards)
    cam = (f"Camera-only check: attack {deg(m.get('attack_angle_cam_deg'))}, "
           f"vertical {deg(m.get('vertical_bat_angle_cam_deg'))}") if m else ""
    files = "".join(f'<a href="/files/{sid}/{html.escape(n)}">{html.escape(n)}</a> &nbsp; '
                    for n in d.get("files", []))
    img = (f'<a href="/files/{sid}/swing.png" target=_blank><img src="/files/{sid}/swing.png" '
           f'alt="swing render (click to open)"></a>' if "swing.png" in d.get("files", [])
           else "<div class=muted>No render (processing failed — see status).</div>")
    replay = (f'<div id="replay" data-url="/files/{sid}/replay.json">'
              '<canvas style="width:100%;height:420px;touch-action:none" aria-label="3D swing replay"></canvas>'
              '<p class="replay-status">Loading replay...</p><button>Play</button>'
              '<input type="range" min="0" max="1" value="0" step="1" aria-label="Replay frame" style="width:55%">'
              '<select aria-label="Playback speed"><option value="0.25">¼ speed</option>'
              '<option value="0.5">½ speed</option><option value="1">Normal speed</option></select>'
              '<p class="muted">Drag to rotate; scroll to zoom. Pink: camera-supported. '
              'Orange dashed: IMU/model estimate. Camera-supported poses include short interpolation '
              'between observations. Estimated gap motion is excluded from speed validation.</p>'
              '</div><script src="/replay.js" defer></script>'
              if 'replay.json' in d.get('files', []) else '')
    keys = ("time_offset_ms", "camera_coverage", "calib_rms_axis_deg", "sync_correlation", "imu_gaps_over_10ms")
    dtext = ", ".join(f"{k} {diag.get(k)}" for k in keys if k in diag)
    quality = meta.get('camera_quality', {})
    quality_text = ''
    if quality:
        fps = quality.get('effective_saved_fps')
        fraction = quality.get('tracking_fraction_entire_clip', 0)
        quality_text = ('Saved video: %s frames, %s FPS. Frames with raw 3D positions over the '
                        'entire clip: %.1f%%. These counts include stance and finish.' %
                        (quality.get('saved_frames'), 'n/a' if fps is None else '%.1f' % fps,
                         100*fraction))
    replay_info = diag.get('replay') or {}
    estimated = replay_info.get('active_swing_estimated_fraction')
    if estimated is not None:
        quality_text += ' Active-swing replay: %.1f%% camera-supported, %.1f%% estimated.' % (
            100*(1-estimated), 100*estimated)
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Swing {sid}</title>
<style>{STYLE}</style></head><body><main>
<a href="/">&larr; all swings</a><h1>Swing {sid}</h1>
<div class=muted>{html.escape(meta.get('created', ''))} · {html.escape(meta.get('source', ''))} ·
status {html.escape(str(meta.get('status', '')))} · code {html.escape(str(meta.get('git', '')))}</div>
<div class=grid>{grid}</div><div class=muted>{cam}</div>
<div class=card>{replay or img}</div>
<details><summary>Static render</summary>{img}</details>
<div class=card><b>Tracking quality</b><p>{html.escape(quality_text)}</p><div class=muted>{html.escape(dtext)}</div>
<div class=muted>{html.escape(str(m.get('speed_note', '')))}</div></div>
<div class=card><b>Files</b><div>{files}</div>
<button onclick="if(confirm('Delete this swing?'))fetch('/api/delete/{sid}',{{method:'POST'}}).then(()=>location='/')">Delete swing</button></div>
</main></body></html>"""


def make_handler(backend: Backend, restart):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the terminal quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj))

        def do_GET(self):
            p = unquote(self.path.split("?")[0])
            if p == "/":
                return self._send(200, INDEX, "text/html; charset=utf-8")
            if p == "/replay.js":
                with open(os.path.join(os.path.dirname(__file__), 'replay.js')) as f:
                    return self._send(200, f.read(), 'text/javascript; charset=utf-8')
            if p == "/api/status":
                return self._json({**backend.status(), "git": git_info()})
            if p == "/api/swings":
                return self._json(backend.list_swings())
            if p.startswith("/swing/"):
                d = backend.swing_detail(p.split("/", 2)[2])
                if d is None:
                    return self._send(404, "Not found", "text/plain")
                return self._send(200, swing_page(d), "text/html; charset=utf-8")
            if p.startswith("/files/"):
                parts = p.split("/")
                if len(parts) == 4:
                    sid, name = os.path.basename(parts[2]), os.path.basename(parts[3])
                    path = os.path.join(SWINGS, sid, name)
                    if os.path.isfile(path):
                        with open(path, "rb") as f:
                            return self._send(200, f.read(),
                                              TYPES.get(os.path.splitext(name)[1].lower(), "application/octet-stream"))
                return self._send(404, "Not found", "text/plain")
            return self._send(404, "Not found", "text/plain")

        def do_POST(self):
            p = unquote(self.path.split("?")[0])
            if p == "/api/start":
                try:
                    size = int(self.headers.get('Content-Length', 0))
                    if not 0 <= size <= 4096:
                        raise ValueError('Invalid request size')
                    options = json.loads(self.rfile.read(size) or b'{}')
                    ok = backend.start_hardware(options.get('capture_mode', '120fps'),
                                                options.get('seconds', 5))
                except (ValueError, TypeError, AttributeError) as e:
                    return self._json({'ok': False, 'message': str(e)}, 400)
                return self._json({"ok": ok, "message": backend.message})
            if p == "/api/stop":
                threading.Thread(target=backend.stop, daemon=True).start()
                return self._json({"ok": True, "message": "Stopping..."})
            if p == "/api/demo":
                if backend.mode != "idle" or backend.busy:
                    return self._json({"ok": False, "message": "Busy -- stop recording first."})
                threading.Thread(target=backend.demo_swing, daemon=True).start()
                return self._json({"ok": True, "message": "Simulating a swing..."})
            if p == "/api/update":
                if backend.mode != "idle" or backend.busy:
                    return self._json({"ok": False, "message": "Stop recording before updating."})
                out = git_pull()
                backend.say("git pull: " + out.splitlines()[-1] if out else "git pull done")
                self._json({"ok": True, "output": out + "\n\nRestarting the app..."})
                threading.Thread(target=restart, daemon=True).start()
                return
            if p.startswith("/api/delete/"):
                return self._json({"ok": backend.delete_swing(p.rsplit("/", 1)[1])})
            return self._send(404, "Not found", "text/plain")

    return H


def relaunch():
    """Start this app again with the same arguments (after git pull). Linux, Mac and Windows."""
    os.environ["DCKFT_APP_PULLED"] = "1"
    if os.name == "nt":               # os.execv on Windows detaches from the terminal: run a child instead
        import subprocess
        sys.exit(subprocess.call([sys.executable] + sys.argv))
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _bind(host, port, handler):
    for _ in range(20):               # the previous instance may still be releasing the port
        try:
            return ThreadingHTTPServer((host, port), handler)
        except OSError:
            time.sleep(0.5)
    return ThreadingHTTPServer((host, port), handler)


def serve(backend: Backend, host="0.0.0.0", port=8000, restart=None):
    box = {"restart": False}

    def request_restart():            # Update button: stop serving, then relaunch from the main thread
        time.sleep(1.0)
        box["restart"] = True
        box["httpd"].shutdown()

    httpd = _bind(host, port, make_handler(backend, restart or request_restart))
    box["httpd"] = httpd
    print(f"DC-KFT app on http://{host}:{port}  (Ctrl+C to quit)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        backend.stop()
        httpd.server_close()
    if box["restart"]:
        print("Restarting with the updated code ...")
        relaunch()

