"""
A clean three.js view of the offside scene — same geometry as the Plotly scene,
rendered as a self-contained HTML document embedded via <iframe srcdoc=...>.

Gradio's HTML component strips bare <script> tags, but an iframe runs its own
document, so the three.js module + scene data live inside srcdoc. This is pure
CPU: we just serialize the cached meshes; the browser does the rendering.
"""

import json
import html

import numpy as np

from .geometry import _forward_most

_PALETTE = ["#dc2626", "#2563eb", "#f59e0b", "#7c3aed",
            "#0ea5e9", "#db2777", "#10b981", "#f97316"]


def _verdict(placed, plane_x, attack_sign, defender_ids, masks, too_close=0.30):
    dset = set(int(d) for d in (defender_ids or []))
    m = masks or {}
    fmost = {i: _forward_most(placed[i], attack_sign, m.get(i)) for i in placed}
    line_fwd = None if plane_x is None else attack_sign * plane_x
    out, any_off = {}, False
    for n_, i in enumerate(placed):
        if i in dset:
            out[i] = ("#2563eb", "defender")
        elif line_fwd is None:
            out[i] = (_PALETTE[n_ % len(_PALETTE)], "")
        else:
            d = fmost[i] - line_fwd
            if d > too_close:
                out[i] = ("#dc2626", f"OFFSIDE +{d:.2f}m"); any_off = True
            elif d < -too_close:
                out[i] = ("#16a34a", f"onside {d:.2f}m")
            else:
                out[i] = ("#f59e0b", f"close {d:+.2f}m")
    return out, any_off


_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#0b1f3a;overflow:hidden;font-family:system-ui,sans-serif}
#c{display:block;width:100vw;height:100vh}
#hud{position:absolute;top:14px;left:0;right:0;text-align:center;color:#fff;font-weight:700;font-size:22px;pointer-events:none;text-shadow:0 1px 6px #000}
#hint{position:absolute;bottom:10px;left:0;right:0;text-align:center;color:#c9bcf7;font-size:12px;pointer-events:none}
</style></head><body>
<div id="hud"></div><div id="hint">drag to orbit · scroll to zoom</div><canvas id="c"></canvas>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const D = __DATA__;
document.getElementById("hud").textContent = D.title;
const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1f3a);
const cx = (D.x0 + D.x1) / 2, cy = (D.y0 + D.y1) / 2;
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
camera.up.set(0, 0, 1);
camera.position.set(cx, D.y0 - 12, 7);
const controls = new OrbitControls(camera, canvas);
controls.target.set(cx, cy, 1);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const dl = new THREE.DirectionalLight(0xffffff, 0.85);
dl.position.set(cx - 6, cy - 6, 18); scene.add(dl);
const pitch = new THREE.Mesh(
  new THREE.PlaneGeometry(D.x1 - D.x0, D.y1 - D.y0),
  new THREE.MeshStandardMaterial({color:0x2e7d4f, roughness:1}));
pitch.position.set(cx, cy, 0); scene.add(pitch);
const bp = [[D.x0,D.y0],[D.x1,D.y0],[D.x1,D.y1],[D.x0,D.y1]].map(p => new THREE.Vector3(p[0], p[1], 0.02));
scene.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(bp),
  new THREE.LineBasicMaterial({color:0xffffff})));
const faces = D.faces;
for (const p of D.players) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(p.v, 3));
  g.setIndex(faces); g.computeVertexNormals();
  scene.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial(
    {color:new THREE.Color(p.color), roughness:0.6, metalness:0.05})));
}
if (D.plane_x !== null) {
  const op = new THREE.Mesh(new THREE.PlaneGeometry(3, D.y1 - D.y0),
    new THREE.MeshBasicMaterial({color:0xff3333, transparent:true, opacity:0.32, side:THREE.DoubleSide}));
  op.rotation.y = Math.PI / 2;
  op.position.set(D.plane_x, cy, 1.5); scene.add(op);
}
function resize(){ const w=canvas.clientWidth, h=canvas.clientHeight;
  renderer.setSize(w, h, false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
window.addEventListener("resize", resize); resize();
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
</script></body></html>"""


def scene_html(placed, faces, plane_x, attack_sign, defender_ids, masks):
    """Return an <iframe> embedding a three.js render of the placed scene."""
    colors, any_off = _verdict(placed, plane_x, attack_sign, defender_ids, masks)
    allP = np.vstack(list(placed.values()))
    data = {
        "faces": [int(v) for tri in np.asarray(faces) for v in tri],
        "players": [{"v": [round(float(c), 3) for c in placed[i].reshape(-1)],
                     "color": colors[i][0], "label": f"#{i} {colors[i][1]}".strip()}
                    for i in placed],
        "x0": float(allP[:, 0].min() - 4), "x1": float(allP[:, 0].max() + 4),
        "y0": float(allP[:, 1].min() - 4), "y1": float(allP[:, 1].max() + 4),
        "plane_x": (None if plane_x is None else float(plane_x)),
        "title": "VAR — OFFSIDE" if any_off else "VAR — NO OFFSIDE",
    }
    doc = _TEMPLATE.replace("__DATA__", json.dumps(data))
    return (f'<iframe title="VAR 3D scene" '
            f'style="width:100%;height:620px;border:0;border-radius:12px" '
            f'srcdoc="{html.escape(doc, quote=True)}"></iframe>')
