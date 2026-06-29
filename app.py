"""
VAR-style offside visualizer — Gradio app.

Flow:
  upload -> scrub to the frame -> draw 2 goal-parallel lines on that frame
  -> Detect players (GPU, once/frame) -> click players to select
  -> mark defenders -> Build the 3D scene with a draggable offside plane.

The GPU is touched ONLY inside pipeline.gpu (detect_frame / reconstruct_selected);
every other callback here runs on cached numpy and stays on the CPU.
"""

import os

# headless GL + CUDA fragmentation hygiene must be set before any heavy import
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import gradio as gr

from pipeline.video import probe_video, grab_frame
from pipeline.overlay import (annotate_detections, draw_lines, draw_masks,
                              pick_box, pick_mask)
from pipeline.autolines import propose_lines
from pipeline import geometry as G

# detector radio choice -> (backend, overlay/selection style)
DETECTORS = {
    "ViTDet (boxes)": ("vitdet", "boxes"),
    "RF-DETR (boxes)": ("rfdetr", "boxes"),
    "RF-DETR (segments)": ("rfdetr", "segments"),
}


def _resolve(detector):
    return DETECTORS.get(detector, ("vitdet", "boxes"))


def _render(frame, people, selected, style):
    if style == "segments":
        return draw_masks(frame, people, selected)
    return annotate_detections(frame, people, selected)


def _pick(people, x, y, style):
    return pick_mask(people, x, y) if style == "segments" else pick_box(people, x, y)


# ============================================================================
# Stage (a): upload + frame scrubbing  (pure CPU)
# ============================================================================
def on_upload(video_path):
    """New clip: size the slider, show frame 0, reset lines/frame state."""
    if not video_path:
        return (gr.update(maximum=1, value=0), None, "Upload a clip to begin.",
                0, None, [], "")
    n, fps = probe_video(video_path)
    n_max = max(n - 1, 0)
    frame = grab_frame(video_path, 0)
    return (
        gr.update(maximum=max(n_max, 1), value=0),
        frame,
        f"{n} frames @ {fps:.1f} fps — scrub to the moment the ball is played, "
        "then auto-detect or click the 2 goal-parallel lines.",
        n_max,
        frame,        # st_frame: clean copy for line redraws
        [],           # st_lines reset
        "",           # line_status reset
        [],           # st_families reset
        0,            # st_fam_idx reset
    )


def on_scrub(video_path, idx):
    """Show the new frame and reset any half-drawn / proposed lines on it."""
    if not video_path:
        return None, None, [], "", [], 0
    frame = grab_frame(video_path, int(idx))
    return frame, frame, [], "", [], 0


def step_frame(idx, delta, n_max):
    return max(0, min(int(n_max), int(idx) + delta))


# ============================================================================
# Stage (b): draw 2 goal-parallel lines on the scrubbed frame  (CPU)
# ============================================================================
def on_line_click(line_pts, clean_frame, evt: gr.SelectData):
    """Collect 4 clicks = 2 lines; always redraw from the clean frame (no drift).

    A manual click clears any auto-proposed families (so Flip stops applying).
    """
    pts = list(line_pts) if line_pts else []
    if len(pts) >= 4:
        pts = []
    pts.append([float(evt.index[0]), float(evt.index[1])])
    img = draw_lines(clean_frame, pts)
    status = {1: "line 1: 1/2", 2: "line 1 set", 3: "line 2: 1/2",
              4: "both lines set ✓ — now Detect players"}.get(len(pts), "")
    return pts, img, status, [], 0


def on_auto_lines(clean_frame):
    """Propose the 2 goal-parallel lines from detected pitch lines."""
    if clean_frame is None:
        return [], None, "Scrub to a frame first.", [], 0
    fams = propose_lines(clean_frame)
    if not fams:
        return [], gr.update(), ("No clear pitch lines found — draw the 2 "
                                 "goal-parallel lines by hand."), [], 0
    pts = fams[0]["pts"]
    img = draw_lines(clean_frame, pts)
    status = (f"Proposed goal-parallel lines (direction 1/{len(fams)}). If the "
              "offside axis looks wrong, click **Flip line direction**, or just "
              "click the frame to redraw by hand.")
    return pts, img, status, fams, 0


def on_flip_lines(families, fam_idx, clean_frame):
    """Switch the proposal to the other detected line-family."""
    if not families:
        return gr.update(), gr.update(), "Run **Auto-detect lines** first.", gr.update()
    if len(families) < 2:
        return (families[0]["pts"], draw_lines(clean_frame, families[0]["pts"]),
                "Only one line direction was found — redraw by hand if it's wrong.", 0)
    new_idx = (int(fam_idx) + 1) % len(families)
    pts = families[new_idx]["pts"]
    return (pts, draw_lines(clean_frame, pts),
            f"Line direction {new_idx + 1}/{len(families)}.", new_idx)


# ============================================================================
# Stage (c): detect (GPU, cached) + show boxes  (CPU after the call)
# ============================================================================
def on_detect(video_path, idx, conf, detector):
    """Detect step: chosen detector (ViTDet / RF-DETR), boxes + masks. No meshes here."""
    from pipeline.gpu import detect_frame
    if not video_path:
        return None, [], [], "Upload a clip first.", gr.update(choices=[], value=[])
    backend, style = _resolve(detector)
    people = detect_frame(video_path, idx, conf, backend)
    if not people:
        return (None, [], [], "No players detected — lower the confidence slider.",
                gr.update(choices=[], value=[]))
    annotated = _render(grab_frame(video_path, idx), people, [], style)
    unit = "silhouette" if style == "segments" else "box"
    msg = (f"Detected {len(people)} players with {detector}. Click a player's "
           f"{unit} to select (click again to deselect).")
    return annotated, people, [], msg, gr.update(choices=[], value=[])


# ============================================================================
# Stage (d): click players to select  (CPU)
# ============================================================================
def on_select_player(people, selected, clean_frame, cur_def, detector, evt: gr.SelectData):
    """Toggle the clicked player; refresh highlight + defender choices (box or segment)."""
    if not people:
        return None, selected or [], "Detect players first.", gr.update()
    _, style = _resolve(detector)
    hit = _pick(people, evt.index[0], evt.index[1], style)
    sel = list(selected or [])
    if hit is not None:
        sel.remove(hit) if hit in sel else sel.append(hit)
    sel = sorted(sel)
    img = _render(clean_frame, people, sel, style)
    msg = (f"Selected players: {sel}. Mark defenders below, then Build."
           if sel else "Click a player to select.")
    keep_def = [d for d in (cur_def or []) if d in sel]
    return img, sel, msg, gr.update(choices=sel, value=keep_def)


def on_detector_change(people, selected, clean_frame, detector):
    """Re-render current detections in the new style; hint to re-Detect on backend swap."""
    if not people:
        return gr.update(), "Detector set — click Detect players to apply."
    _, style = _resolve(detector)
    return _render(clean_frame, people, selected or [], style), \
        "Re-rendered. For a different backend, click Detect players to re-run."


# ============================================================================
# Stage (e)+(f): place players + build the Plotly scene with a draggable plane
# ============================================================================
def on_build(video_path, idx, people_det, selected_ids, line_pts, flip_up,
             attack_dir, defender_ids):
    from pipeline.gpu import reconstruct_selected, get_faces
    if not selected_ids:
        return None, "Click at least one player to select.", gr.update(), None, +1, [], {}
    if not line_pts or len(line_pts) < 4:
        return None, "Draw 2 goal-parallel lines (4 points) on the frame first.", \
               gr.update(), None, +1, [], {}

    # Reconstruct ONLY the selected players' boxes (the heavy GPU step).
    selected_ids = sorted(int(i) for i in selected_ids)
    boxes = [people_det[i]["bbox"] for i in selected_ids]
    recon = reconstruct_selected(video_path, idx, boxes)
    if not recon:
        return None, "Reconstruction returned no meshes.", gr.update(), None, +1, [], {}
    # Key meshes back to their original detection ids (recon order == boxes order).
    people = {selected_ids[k]: recon[k] for k in range(len(recon))}
    faces = get_faces()
    h, w = grab_frame(video_path, idx).shape[:2]
    focal = people[selected_ids[0]]["focal_length"]
    gdir = G.goal_dir_from_lines(line_pts, focal, w, h)

    placed = G.place_players(people, selected_ids, gdir, flip_up=flip_up)
    med = float(np.median([placed[i][:, 2].max() for i in placed]))
    masks = {i: G.non_arm_mask(people[i]) for i in selected_ids}  # exclude arms/hands

    attack_sign = +1 if str(attack_dir).startswith("+X") else -1
    dset = [int(d) for d in (defender_ids or [])]
    plane_x = G.offside_plane_x(placed, attack_sign, dset, masks)
    fig = G.build_scene(placed, faces, plane_x, attack_sign, dset, masks)

    allX = np.vstack(list(placed.values()))[:, 0]
    x0, x1 = float(allX.min() - 4), float(allX.max() + 4)
    plane_update = gr.update(minimum=x0, maximum=x1, value=float(plane_x),
                             visible=True, label="Drag the offside plane (X, m)")

    warn = "  ⚠ heights look wrong — toggle 'flip up'." if med < 1.0 else ""
    return (fig, f"Median player height {med:.2f} m (expect ~1.7–1.9).{warn}",
            plane_update, placed, attack_sign, dset, masks)


def on_plane(placed, plane_x, attack_sign, defender_ids, masks):
    """Re-render the scene at a new plane X — pure CPU on the cached placement."""
    from pipeline.gpu import get_faces
    if not placed:
        return gr.update()
    return G.build_scene(placed, get_faces(), float(plane_x),
                         int(attack_sign), defender_ids or [], masks)


def on_gen3js(placed, plane_x, attack_sign, defender_ids, masks):
    """Generate a clean three.js view of the current scene (uses the current plane)."""
    from pipeline.gpu import get_faces
    from pipeline import threed
    if not placed:
        return "<p style='color:#9a8bd0'>Build a 3D scene first, then generate.</p>"
    return threed.scene_html(placed, get_faces(), float(plane_x),
                             int(attack_sign), defender_ids or [], masks)


# ============================================================================
# UI
# ============================================================================
# Roboflow-flavored theme (violet primary ≈ Roboflow purple #7C3AED) + light CSS.
RF_PURPLE = "#7C3AED"
THEME = gr.themes.Soft(primary_hue=gr.themes.colors.violet,
                       neutral_hue=gr.themes.colors.slate,
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
RF_CSS = """
.gradio-container {max-width: 1180px !important}
#rf-header {background: #7C3AED; color: #fff; padding: 18px 22px; border-radius: 14px;
            margin-bottom: 6px}
#rf-header h2 {color: #fff !important; margin: 0 0 4px 0}
#rf-header p {color: #EDE7FF !important; margin: 0; font-size: 0.92rem}
.gr-button-primary, button.primary {background: #7C3AED !important; border-color: #7C3AED !important}
"""

with gr.Blocks(title="VAR Offside Visualizer") as demo:
    gr.HTML(
        "<div id='rf-header'><h2>VAR-style Offside Visualizer</h2>"
        "<p>Upload → scrub → click 2 goal-parallel lines → Detect → "
        "click players to select → mark defenders → Build</p></div>"
    )
    gr.Markdown(
        "_Scale comes from reconstructed body height, so positions are approximate "
        "metres — good for relative offside ordering, not sub-10 cm calls._"
    )

    # session state
    st_nmax = gr.State(0)        # last valid frame index
    st_frame = gr.State(None)    # clean RGB of the current frame
    st_people = gr.State([])     # slim detections
    st_lines = gr.State([])      # clicked / proposed line points
    st_families = gr.State([])   # auto-proposed line families
    st_fam_idx = gr.State(0)     # which proposed family is active
    st_selected = gr.State([])   # player ids selected by clicking
    st_placed = gr.State(None)   # placed meshes after build
    st_attack = gr.State(+1)
    st_defenders = gr.State([])
    st_masks = gr.State({})      # per-player non-arm vertex masks

    video = gr.Video(label="1. Upload match clip")
    status = gr.Markdown()

    with gr.Row():
        frame_slider = gr.Slider(0, 1, value=0, step=1,
                                 label="2. Scrub to the offside frame")
    with gr.Row():
        prev_btn = gr.Button("◀ prev frame")
        next_btn = gr.Button("next frame ▶")

    # Stage (b): lines drawn on the scrubbed frame — auto-proposed or by hand
    frame_view = gr.Image(label="3. Goal-parallel lines: auto-detect or click 4 points",
                          interactive=True)
    with gr.Row():
        auto_lines_btn = gr.Button("✨ Auto-detect lines")
        flip_lines_btn = gr.Button("↔ Flip line direction")
    line_status = gr.Markdown()

    # Stage (c)/(d): detect, then click to select
    with gr.Row():
        thr = gr.Slider(0.0, 0.95, value=0.3, step=0.05, label="Detection confidence")
        detector = gr.Radio(list(DETECTORS.keys()), value="ViTDet (boxes)",
                            label="Detector")
        detect_btn = gr.Button("4. Detect players (GPU)", variant="primary")
    detect_view = gr.Image(label="5. Click a player to select (click again to deselect)",
                           interactive=True)
    select_status = gr.Markdown()

    # Stage (e)/(f)
    defenders = gr.CheckboxGroup(choices=[], label="6. Defenders (incl. GK) — sets the offside line")
    with gr.Row():
        flip = gr.Checkbox(False, label="flip up (if players are upside-down)")
        attack = gr.Radio(["−X  ←", "+X  →"], value="−X  ←",
                          label="Attacking direction")
    build_btn = gr.Button("7. Build 3D scene + offside line", variant="primary")

    scene = gr.Plot(label="3D scene")
    plane_slider = gr.Slider(-10, 10, value=0, step=0.05, visible=False,
                             label="Drag the offside plane (X, m)")
    build_status = gr.Markdown()

    gen3js_btn = gr.Button("🎥 Generate clean 3D scene (three.js)")
    scene3js = gr.HTML()

    # --- wiring ---
    video.change(on_upload, [video],
                 [frame_slider, frame_view, status, st_nmax, st_frame, st_lines,
                  line_status, st_families, st_fam_idx])
    frame_slider.change(on_scrub, [video, frame_slider],
                        [frame_view, st_frame, st_lines, line_status,
                         st_families, st_fam_idx])
    prev_btn.click(lambda i, m: step_frame(i, -1, m), [frame_slider, st_nmax], [frame_slider])
    next_btn.click(lambda i, m: step_frame(i, +1, m), [frame_slider, st_nmax], [frame_slider])

    frame_view.select(on_line_click, [st_lines, st_frame],
                      [st_lines, frame_view, line_status, st_families, st_fam_idx])
    auto_lines_btn.click(on_auto_lines, [st_frame],
                         [st_lines, frame_view, line_status, st_families, st_fam_idx])
    flip_lines_btn.click(on_flip_lines, [st_families, st_fam_idx, st_frame],
                         [st_lines, frame_view, line_status, st_fam_idx])

    detect_btn.click(on_detect, [video, frame_slider, thr, detector],
                     [detect_view, st_people, st_selected, select_status, defenders])
    detect_view.select(on_select_player,
                       [st_people, st_selected, st_frame, defenders, detector],
                       [detect_view, st_selected, select_status, defenders])
    detector.change(on_detector_change, [st_people, st_selected, st_frame, detector],
                    [detect_view, select_status])

    build_btn.click(
        on_build,
        [video, frame_slider, st_people, st_selected, st_lines, flip, attack, defenders],
        [scene, build_status, plane_slider, st_placed, st_attack, st_defenders, st_masks])
    plane_slider.change(on_plane,
                        [st_placed, plane_slider, st_attack, st_defenders, st_masks],
                        [scene])
    gen3js_btn.click(on_gen3js,
                     [st_placed, plane_slider, st_attack, st_defenders, st_masks],
                     [scene3js])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860,
                        theme=THEME, css=RF_CSS)
