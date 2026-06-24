"""
VAR-style offside visualizer — Gradio app.

Pipeline (matches the working Colab notebook):
  upload video -> scrub to the offside frame -> detect players (GPU, once/frame)
  -> select players -> click 2 goal-parallel lines -> place 3D poses on a field
  -> Plotly scene with a draggable offside plane.

The GPU is touched ONLY inside pipeline.gpu.reconstruct_frame(); every other
callback in this file runs on cached numpy and stays on the CPU.
"""

import os

# headless GL + CUDA fragmentation hygiene must be set before any heavy import
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import gradio as gr

from pipeline.video import probe_video, grab_frame
from pipeline.overlay import annotate_detections, draw_lines
from pipeline import geometry as G


# ============================================================================
# Stage (a): upload + frame scrubbing  (pure CPU)
# ============================================================================
def on_upload(video_path):
    """New clip: size the slider, show frame 0, report length, remember max idx."""
    if not video_path:
        return gr.update(maximum=1, value=0), None, "Upload a clip to begin.", 0
    n, fps = probe_video(video_path)
    n_max = max(n - 1, 0)
    return (
        gr.update(maximum=max(n_max, 1), value=0),
        grab_frame(video_path, 0),
        f"{n} frames @ {fps:.1f} fps — scrub to the moment the ball is played.",
        n_max,
    )


def on_scrub(video_path, idx):
    if not video_path:
        return None
    return grab_frame(video_path, int(idx))


def step_frame(idx, delta, n_max):
    """Clamp prev/next within [0, slider max]."""
    return max(0, min(int(n_max), int(idx) + delta))


# ============================================================================
# Stage (b)+(c): detect (GPU, cached) + populate selection  (CPU after the call)
# ============================================================================
def on_detect(video_path, idx, bbox_thr):
    """The one GPU step. Imported lazily so the app boots without the model."""
    from pipeline.gpu import reconstruct_frame
    if not video_path:
        return None, gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), [], "Upload a clip first."
    people = reconstruct_frame(video_path, idx, bbox_thr)
    if not people:
        return (None, gr.update(choices=[], value=[]), gr.update(choices=[], value=[]),
                [], "No players detected — try a lower threshold.")
    annotated = annotate_detections(grab_frame(video_path, idx), people)
    choices = list(range(len(people)))
    msg = (f"Detected {len(people)} players. Tick the ones to analyze, mark defenders, "
           "then click 2 goal-parallel lines on the image above.")
    return (annotated,
            gr.update(choices=choices, value=[]),
            gr.update(choices=choices, value=[]),
            people, msg)


# ============================================================================
# Stage (d): click two goal-parallel lines (4 points)  (CPU)
# ============================================================================
def on_line_click(line_pts, frame_rgb, evt: gr.SelectData):
    """Collect 4 clicks = 2 lines; redraw dots + segments. Resets after a full set."""
    pts = list(line_pts) if line_pts else []
    if len(pts) >= 4:
        pts = []
    pts.append([float(evt.index[0]), float(evt.index[1])])
    img = draw_lines(frame_rgb, pts)
    status = {1: "line 1: 1/2", 2: "line 1 set", 3: "line 2: 1/2",
              4: "both lines set — build the scene"}.get(len(pts), "")
    return pts, img, status


# ============================================================================
# Stage (e)+(f): place players + build the Plotly scene with a draggable plane
# ============================================================================
def on_build(video_path, idx, bbox_thr, selected_ids, line_pts, flip_up,
             attack_dir, defender_ids):
    from pipeline.gpu import reconstruct_frame, get_faces
    if not selected_ids:
        return None, "Select at least one player.", gr.update(), None, +1, []
    if not line_pts or len(line_pts) < 4:
        return None, "Click 2 goal-parallel lines (4 points) first.", gr.update(), None, +1, []

    people = reconstruct_frame(video_path, idx, bbox_thr)
    faces = get_faces()
    h, w = grab_frame(video_path, idx).shape[:2]
    focal = people[selected_ids[0]]["focal_length"]
    gdir = G.goal_dir_from_lines(line_pts, focal, w, h)

    placed = G.place_players(people, list(selected_ids), gdir, flip_up=flip_up)
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
    msg = f"Median player height {med:.2f} m (expect ~1.7–1.9).{warn}"
    return fig, msg, plane_update, placed, attack_sign, dset


def on_plane(placed, plane_x, attack_sign, defender_ids):
    """Re-render the scene at a new plane X — pure CPU on the cached placement."""
    from pipeline.gpu import get_faces
    if not placed:
        return gr.update()
    fig = G.build_scene(placed, get_faces(), float(plane_x),
                        int(attack_sign), defender_ids or [])
    return fig


# ============================================================================
# UI
# ============================================================================
with gr.Blocks(title="VAR Offside Visualizer") as demo:
    gr.Markdown(
        "## VAR-style Offside Visualizer\n"
        "Upload a clip, scrub to the moment the ball is played, then "
        "detect → select → click 2 goal-parallel lines → build the 3D scene.\n\n"
        "_Scale comes from reconstructed body height, so positions are approximate "
        "metres — good for relative offside ordering, not sub-10 cm calls._"
    )

    # session state
    st_nmax = gr.State(0)        # last valid frame index
    st_people = gr.State([])     # slim detections for the current frame
    st_lines = gr.State([])      # clicked line points
    st_placed = gr.State(None)   # placed meshes (field frame) after build
    st_attack = gr.State(+1)     # attack_sign
    st_defenders = gr.State([])  # defender ids used for the verdict

    # --- stage (a) ---
    video = gr.Video(label="1. Upload match clip")
    status = gr.Markdown()
    with gr.Row():
        frame_slider = gr.Slider(0, 1, value=0, step=1,
                                 label="2. Scrub to the offside frame")
    with gr.Row():
        prev_btn = gr.Button("◀ prev frame")
        next_btn = gr.Button("next frame ▶")
    frame_view = gr.Image(label="Current frame", interactive=False)

    # --- stage (b)/(c) ---
    with gr.Row():
        thr = gr.Slider(0.3, 0.95, value=0.85, step=0.05, label="Detection confidence")
        detect_btn = gr.Button("3. Detect players (GPU)", variant="primary")
    detect_view = gr.Image(
        label="Detected players — click 2 goal-parallel lines here (4 points)",
        interactive=True)
    line_status = gr.Markdown()
    sel = gr.CheckboxGroup(choices=[], label="4. Select players to analyze (indices)")

    # --- stage (e)/(f) controls ---
    with gr.Row():
        flip = gr.Checkbox(False, label="flip up (if players are upside-down)")
        attack = gr.Radio(["toward +X", "toward -X"], value="toward +X",
                          label="Attacking direction")
    defenders = gr.CheckboxGroup(
        choices=[], label="Defenders (incl. GK) — sets the offside line")
    build_btn = gr.Button("5. Build 3D scene + offside line", variant="primary")

    scene = gr.Plot(label="3D scene")
    plane_slider = gr.Slider(-10, 10, value=0, step=0.05, visible=False,
                             label="Drag the offside plane (X, m)")
    build_status = gr.Markdown()

    # --- wiring ---
    video.change(on_upload, [video], [frame_slider, frame_view, status, st_nmax])
    frame_slider.change(on_scrub, [video, frame_slider], [frame_view])
    prev_btn.click(lambda i, m: step_frame(i, -1, m),
                   [frame_slider, st_nmax], [frame_slider])
    next_btn.click(lambda i, m: step_frame(i, +1, m),
                   [frame_slider, st_nmax], [frame_slider])

    detect_btn.click(on_detect, [video, frame_slider, thr],
                     [detect_view, sel, defenders, st_people, status])
    detect_view.select(on_line_click, [st_lines, detect_view],
                       [st_lines, detect_view, line_status])

    build_btn.click(
        on_build,
        [video, frame_slider, thr, sel, st_lines, flip, attack, defenders],
        [scene, build_status, plane_slider, st_placed, st_attack, st_defenders])
    plane_slider.change(on_plane, [st_placed, plane_slider, st_attack, st_defenders],
                        [scene])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
