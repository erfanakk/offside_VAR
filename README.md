---
title: VAR Offside Visualizer
emoji: 🥅
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# VAR-style Offside Visualizer

Upload a match clip, scrub to the moment the ball is played, reconstruct selected
players in 3D with SAM 3D Body, and place them on a virtual pitch with a draggable
offside plane.

## Pipeline

1. **Upload** a video clip.
2. **Scrub** to the offside frame (slider + prev/next, frames seeked on demand).
3. **Detect** players on that frame — the only GPU step, cached per (video, frame, threshold).
4. **Select** the players to analyze and mark the defenders (incl. GK).
5. **Click two goal-parallel lines** (4 points) on the detected frame to fix the offside axis.
6. **Build** the 3D scene; drag the offside plane and read the OFFSIDE / NO-OFFSIDE verdict.

The GPU runs once per frame (`pipeline/gpu.py`). Scrubbing, line geometry,
placement, plotting, and the draggable plane are all CPU on the cached result.

## Code layout

```
app.py            Gradio UI + event wiring (CPU)
pipeline/
  video.py        frame seek/probe (CPU)
  gpu.py          model load + reconstruct_frame  ← the ONLY GPU code
  geometry.py     vanishing point, ground fit, field frame, scene (CPU)
  overlay.py      detection boxes + line-click drawing (CPU)
```

Isolating the GPU in `pipeline/gpu.py` means moving inference to a serverless
backend (Modal / ZeroGPU) later only touches `reconstruct_frame`.

## Deploy

This is a **Docker SDK** Space for **dedicated GPU hardware** (A100 recommended):

1. Set hardware to an A100 tier.
2. Add a secret `HF_TOKEN` — a read token for an account with approved access to
   the gated **`facebook/sam-3d-body-dinov3`**. (Override the repo with the
   `SAM3D_REPO_ID` env var if you use a different checkpoint.)
3. First boot builds the image and downloads ~7 GB of weights — give it time.
   After that, the model stays warm until you pause the Space.

**Cost control:** dedicated GPU bills while the Space is running, with no
auto-shutoff. Pause the Space from its settings when you are not using it.

### Why not ZeroGPU?

ZeroGPU allocates the GPU per call, caps call duration, enforces a daily quota,
and cold-loads the ~7 GB model stack on each allocation — a poor fit for an
interactive video-scrubbing session, and it requires the Gradio SDK (not Docker).

## Notes / limits

- Scale comes from the reconstructed body height, so positions are approximate
  metres — good for relative offside ordering, **not** sub-10 cm officiating calls.
  The verdict surfaces a "too close to call" band rather than implying false precision.
- The offside point currently uses the forward-most body vertex **including arms**;
  excluding arms (via MHR body-part labels) is planned — see `TODO.md`.
- "Find the offside moment" is manual scrubbing; automatic pass-instant detection
  (ball tracking) is future work.
