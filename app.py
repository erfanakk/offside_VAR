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
from pipeline.overlay import draw_masks, draw_lines, pick_mask
from pipeline import geometry as G


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
        "then click 2 goal-parallel lines on the frame.",
        n_max,
        frame,        # st_frame: clean copy for line redraws
        [],           # st_lines reset
        "",           # line_status reset
    )


def on_scrub(video_path, idx):
    """Show the new frame and reset any half-drawn lines on it."""
    if not video_path:
        return None, None, [], ""
    frame = grab_frame(video_path, int(idx))
    return frame, frame, [], ""


def step_frame(idx, delta, n_max):
    return max(0, min(int(n_max), int(idx) + delta))


# ============================================================================
# Stage (b): draw 2 goal-parallel lines on the scrubbed frame  (CPU)
# ============================================================================
def on_line_click(line_pts, clean_frame, evt: gr.SelectData):
    """Collect 4 clicks = 2 lines; always redraw from the clean frame (no drift)."""
    pts = list(line_pts) if line_pts else []
    if len(pts) >= 4:
        pts = []
    pts.append([float(evt.index[0]), float(evt.index[1])])
    img = draw_lines(clean_frame, pts)
    status = {1: "line 1: 1/2", 2: "line 1 set", 3: "line 2: 1/2",
              4: "both lines set ✓ — now Detect players"}.get(len(pts), "")
    return pts, img, status


# ============================================================================
# Stage (c): detect (GPU, cached) + show boxes  (CPU after the call)
# ============================================================================
def on_detect(video_path, idx, conf):
    """Detect step: RF-DETR-Seg only (boxes + masks). No mesh reconstruction here."""
    from pipeline.gpu import detect_frame
    if not video_path:
        return None, [], [], "Upload a clip first.", gr.update(choices=[], value=[])
    people = detect_frame(video_path, idx, conf)
    if not people:
        return (None, [], [], "No players detected — lower the confidence slider.",
                gr.update(choices=[], value=[]))
    annotated = draw_masks(grab_frame(video_path, idx), people, [])
    msg = (f"Detected {len(people)} players. Click a player's silhouette to select "
           "(click again to deselect).")
    return annotated, people, [], msg, gr.update(choices=[], value=[])


# ============================================================================
# Stage (d): click players to select  (CPU)
# ============================================================================
def on_select_player(people, selected, clean_frame, cur_def, evt: gr.SelectData):
    """Toggle the player whose box was clicked; refresh highlight + defender choices."""
    if not people:
        return None, selected or [], "Detect players first.", gr.update()
    hit = pick_mask(people, evt.index[0], evt.index[1])
    sel = list(selected or [])
    if hit is not None:
        sel.remove(hit) if hit in sel else sel.append(hit)
    sel = sorted(sel)
    img = draw_masks(clean_frame, people, sel)
    msg = (f"Selected players: {sel}. Mark defenders below, then Build."
           if sel else "Click a box to select a player.")
    keep_def = [d for d in (cur_def or []) if d in sel]
    return img, sel, msg, gr.update(choices=sel, value=keep_def)


# ============================================================================
# Stage (e)+(f): place players + build the Plotly scene with a draggable plane
# ============================================================================
def on_build(video_path, idx, people_det, selected_ids, line_pts, flip_up,
             attack_dir, defender_ids):
    from pipeline.gpu import reconstruct_selected, get_faces
    if not selected_ids:
        return None, "Click at least one player to select.", gr.update(), None, +1, []
    if not line_pts or len(line_pts) < 4:
        return None, "Draw 2 goal-parallel lines (4 points) on the frame first.", \
               gr.update(), None, +1, []

    # Reconstruct ONLY the selected players' boxes (the heavy GPU step).
    selected_ids = sorted(int(i) for i in selected_ids)
    boxes = [people_det[i]["bbox"] for i in selected_ids]
    recon = reconstruct_selected(video_path, idx, boxes)
    if not recon:
        return None, "Reconstruction returned no meshes.", gr.update(), None, +1, []
    # Key meshes back to their original detection ids (recon order == boxes order).
    people = {selected_ids[k]: recon[k] for k in range(len(recon))}
    faces = get_faces()
    h, w = grab_frame(video_path, idx).shape[:2]
    focal = people[selected_ids[0]]["focal_length"]
    gdir = G.goal_dir_from_lines(line_pts, focal, w, h)

    placed = G.place_players(people, selected_ids, gdir, flip_up=flip_up)
    med = float(np.median([placed[i][:, 2].max() for i in placed]))

    attack_sign = +1 if attack_dir == "toward +X" else -1
    dset = [int(d) for d in (defender_ids or [])]
    plane_x = G.offside_plane_x(placed, attack_sign, dset)
    fig = G.build_scene(placed, faces, plane_x, attack_sign, dset)

    allX = np.vstack(list(placed.values()))[:, 0]
    x0, x1 = float(allX.min() - 4), float(allX.max() + 4)
    plane_update = gr.update(minimum=x0, maximum=x1, value=float(plane_x),
                             visible=True, label="Drag the offside plane (X, m)")

    warn = "  ⚠ heights look wrong — toggle 'flip up'." if med < 1.0 else ""
    return (fig, f"Median player height {med:.2f} m (expect ~1.7–1.9).{warn}",
            plane_update, placed, attack_sign, dset)


def on_plane(placed, plane_x, attack_sign, defender_ids):
    """Re-render the scene at a new plane X — pure CPU on the cached placement."""
    from pipeline.gpu import get_faces
    if not placed:
        return gr.update()
    return G.build_scene(placed, get_faces(), float(plane_x),
                         int(attack_sign), defender_ids or [])


# ============================================================================
# UI
# ============================================================================
with gr.Blocks(title="VAR Offside Visualizer") as demo:
    gr.Markdown(
        "## VAR-style Offside Visualizer\n"
        "Upload → scrub → **click 2 goal-parallel lines** → Detect → "
        "**click players to select** → mark defenders → Build.\n\n"
        "_Scale comes from reconstructed body height, so positions are approximate "
        "metres — good for relative offside ordering, not sub-10 cm calls._"
    )

    # session state
    st_nmax = gr.State(0)        # last valid frame index
    st_frame = gr.State(None)    # clean RGB of the current frame
    st_people = gr.State([])     # slim detections
    st_lines = gr.State([])      # clicked line points
    st_selected = gr.State([])   # player ids selected by clicking
    st_placed = gr.State(None)   # placed meshes after build
    st_attack = gr.State(+1)
    st_defenders = gr.State([])

    video = gr.Video(label="1. Upload match clip")
    status = gr.Markdown()

    with gr.Row():
        frame_slider = gr.Slider(0, 1, value=0, step=1,
                                 label="2. Scrub to the offside frame")
    with gr.Row():
        prev_btn = gr.Button("◀ prev frame")
        next_btn = gr.Button("next frame ▶")

    # Stage (b): lines are drawn directly on the scrubbed frame
    frame_view = gr.Image(label="3. Click 2 goal-parallel lines here (4 points)",
                          interactive=True)
    line_status = gr.Markdown()

    # Stage (c)/(d): detect, then click boxes to select
    with gr.Row():
        thr = gr.Slider(0.0, 0.95, value=0.4, step=0.05, label="Detection confidence")
        detect_btn = gr.Button("4. Detect players (GPU)", variant="primary")
    detect_view = gr.Image(label="5. Click a player's box to select (click again to deselect)",
                           interactive=True)
    select_status = gr.Markdown()

    # Stage (e)/(f)
    defenders = gr.CheckboxGroup(choices=[], label="6. Defenders (incl. GK) — sets the offside line")
    with gr.Row():
        flip = gr.Checkbox(False, label="flip up (if players are upside-down)")
        attack = gr.Radio(["toward +X", "toward -X"], value="toward +X",
                          label="Attacking direction")
    build_btn = gr.Button("7. Build 3D scene + offside line", variant="primary")

    scene = gr.Plot(label="3D scene")
    plane_slider = gr.Slider(-10, 10, value=0, step=0.05, visible=False,
                             label="Drag the offside plane (X, m)")
    build_status = gr.Markdown()

    # --- wiring ---
    video.change(on_upload, [video],
                 [frame_slider, frame_view, status, st_nmax, st_frame, st_lines, line_status])
    frame_slider.change(on_scrub, [video, frame_slider],
                        [frame_view, st_frame, st_lines, line_status])
    prev_btn.click(lambda i, m: step_frame(i, -1, m), [frame_slider, st_nmax], [frame_slider])
    next_btn.click(lambda i, m: step_frame(i, +1, m), [frame_slider, st_nmax], [frame_slider])

    frame_view.select(on_line_click, [st_lines, st_frame],
                      [st_lines, frame_view, line_status])

    detect_btn.click(on_detect, [video, frame_slider, thr],
                     [detect_view, st_people, st_selected, select_status, defenders])
    detect_view.select(on_select_player, [st_people, st_selected, st_frame, defenders],
                       [detect_view, st_selected, select_status, defenders])

    build_btn.click(
        on_build,
        [video, frame_slider, st_people, st_selected, st_lines, flip, attack, defenders],
        [scene, build_status, plane_slider, st_placed, st_attack, st_defenders])
    plane_slider.change(on_plane, [st_placed, plane_slider, st_attack, st_defenders],
                        [scene])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
