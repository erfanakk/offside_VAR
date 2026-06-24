"""VAR offside visualizer pipeline.

Module layout keeps the GPU strictly isolated:
  video.py     — frame seek/probe (CPU)
  gpu.py       — model load + reconstruct_frame (the ONLY GPU code)
  geometry.py  — vanishing point, ground fit, field frame, scene (CPU)
  overlay.py   — cv2 detection boxes + line-click drawing (CPU)
"""
