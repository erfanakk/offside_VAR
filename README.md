---
title: VAR Offside Visualizer
emoji: 🥅
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# VAR-style Offside Visualizer

Reconstruct selected players in 3D from a **single broadcast clip** and draw a
VAR-style offside line you can rotate and inspect — no multi-camera rig, no pitch
calibration.

<img width="600" height="338" alt="EgyptIran (1)" src="https://github.com/user-attachments/assets/b5a096dc-6085-4f6a-b586-7bbb5150e11e" />
<img width="600" height="338" alt="Iran Belg" src="https://github.com/user-attachments/assets/9090e30e-32d7-43cc-b6eb-c0b1a1b8a53a" />

📝 **The story, the concept, and a walkthrough:**
[*Build a 3D Soccer Offside (VAR) System* — Roboflow blog](https://blog.roboflow.com/build-a-3d-soccer-offside-var-system/).
This README covers the setup, architecture, configuration, and technical detail the
blog intentionally leaves out.

---

## Pipeline (what each step actually does)

1. **Upload** a clip (`gr.Video`).
2. **Scrub** to the frame the ball is played — frames are seeked on demand (no bulk extract).
3. **Goal-parallel lines** on that frame: **✨ Auto-detect** (OpenCV: grass/white-line
   masks → Hough → vanishing-point RANSAC, proposes the two families; **↔ Flip** swaps
   them) or click 4 points by hand. These fix the offside axis via the vanishing point.
4. **Detect players (GPU)** with the selected detector — boxes + masks, cached per frame.
5. **Select** players by clicking their box/silhouette (click again to deselect).
6. **Mark defenders** (incl. GK). The offside line is drawn at the *furthest-forward*
   marked defender's furthest body point, **arms/hands excluded** (via MHR keypoints).
7. **Build** → reconstruct only the selected players, place them on a shared field frame,
   and render a Plotly scene with a draggable offside plane + OFFSIDE / NO-OFFSIDE verdict.
   Optionally **🎥 Generate a clean three.js scene** (gridlines, goal-direction arrow,
   in-scene verdict, **Save PNG**).

## Detectors (toggle in the UI)

| Option | Backend | Notes |
|---|---|---|
| **ViTDet (boxes)** | ViTDet-H Cascade Mask R-CNN (detectron2) | most accurate on broadcast footage; heavier |
| **RF-DETR (boxes)** | RF-DETR-Seg (Roboflow) | fast, Roboflow-native |
| **RF-DETR (segments)** | RF-DETR-Seg (Roboflow) | click silhouettes instead of boxes |

Each backend loads **lazily** — you only pay VRAM for the one you use. 3D reconstruction
is always **SAM 3D Body** (`facebook/sam-3d-body-dinov3`; DINOv3 backbone + MHR body model
+ MoGe2 FOV estimator).

## GPU / CPU boundary (the cost design)

The GPU is touched in **exactly two places** — detection and reconstruction — both in
`pipeline/gpu.py`, both cached per frame. Detection runs on the whole frame; the heavy
mesh reconstruction runs **only on the players you selected** (~3, not ~30). Everything
else (scrubbing, line geometry, placement, both renderers, the draggable plane) is pure
CPU on cached NumPy, so a dedicated GPU only ever does the heavy lifting.

## Code layout

```
app.py            Gradio UI + event wiring (CPU)
pipeline/
  video.py        frame seek / probe (CPU)
  gpu.py          detectors + SAM-3D reconstruction  ← the ONLY GPU code
  autolines.py    pitch-line detection + VP-RANSAC proposal (CPU, OpenCV)
  overlay.py      detection boxes / masks + line-click drawing (CPU)
  geometry.py     vanishing point, ground/up fit, field frame, offside, Plotly scene (CPU)
  threed.py       self-contained three.js scene (iframe srcdoc, CPU)
```

## Deploy (Docker SDK Space)

1. **Hardware:** a **GPU tier is required** (CPU fails at `.to("cuda")`). Comfortable
   minimum ≈ **L4 / A10G (24 GB)**; tested on **A100 (40 GB)**. VRAM is the limiter
   (detector + SAM-3D held together) — using RF-DETR and selecting few players lowers it.
2. **Secret `HF_TOKEN`:** a token for an account with **approved access to the gated
   `facebook/sam-3d-body-dinov3`**. Without it the weight download 401s.
3. First boot builds the image (compiles detectron2) and downloads ~7 GB of weights —
   give it time. The model then stays warm until you pause the Space.

**Cost control:** dedicated GPU bills continuously with no auto-shutoff — **pause the
Space** when not in use.

**Config env vars:** `HF_TOKEN` (required secret) · `SAM3D_REPO_ID`
(default `facebook/sam-3d-body-dinov3`) · `RFDETR_SIZE` (default `large`; `nano/small/medium`).

### Why not ZeroGPU?

ZeroGPU allocates the GPU per call, caps duration, enforces a daily quota, cold-loads the
~7 GB stack each time, and requires the Gradio SDK (not Docker) — all a poor fit for an
interactive scrubbing session.

## Accuracy & honesty

- Scale comes from **reconstructed body height**, so positions are approximate metres —
  good for relative offside ordering and a convincing visual, **not** sub-10 cm calls.
- **Level is onside** (offside law): any positive margin flags OFFSIDE, tagged **"(tight)"**
  when within the ±0.30 m band; orange = level / too-close-to-call on the onside side.
- Line detection **proposes** — you confirm/flip/redraw. A full metric homography was
  tried and rejected as the default (it can be confidently wrong on sparse frames).

## Explicitly later (see `TODO.md`)

Automatic pass-instant detection (ball tracking) · jersey/team auto-coloring · a
soccer-trained detector + field-keypoint homography (Roboflow) · three.js realism
(shadows / HDRI / GLB export).
