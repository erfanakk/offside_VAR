"""
The ONLY module that touches the GPU.

`get_estimator()` loads SAM 3D Body once at first use and keeps it warm on the
GPU. `reconstruct_frame()` runs `process_one_image` for a single frame and caches
the slimmed result per (video, frame, threshold). Everything downstream consumes
that cached numpy on the CPU and must never call back into this module's GPU path.

Isolating the GPU here is deliberate: to move inference to a serverless backend
(Modal / ZeroGPU) later, only `reconstruct_frame` has to change.
"""

import os
import sys
import functools

import numpy as np

# The sam-3d-body repo is cloned here by the Dockerfile and added to sys.path.
SAM3D_DIR = os.environ.get("SAM3D_DIR", "/app/sam-3d-body")
if SAM3D_DIR not in sys.path:
    sys.path.insert(0, SAM3D_DIR)

HF_REPO_ID = os.environ.get("SAM3D_REPO_ID", "facebook/sam-3d-body-dinov3")

_ESTIMATOR = None
_FACES = None


def get_estimator():
    """Lazy-load the SAM 3D Body estimator a single time; returns (estimator, faces).

    Verified against facebookresearch/sam-3d-body:
      - setup_sam_3d_body -> load_sam_3d_body_hf -> load_sam_3d_body(ckpt, mhr_path),
        which sidesteps the model_config.yaml path bug in the demo.py route.
      - estimator.faces == model.head_pose.faces (numpy).
      - The "missing keys" warning at load is benign (MHR rig buffers).
    """
    global _ESTIMATOR, _FACES
    if _ESTIMATOR is None:
        from huggingface_hub import login
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
        # Imported here so module import never fails before the repo is on sys.path.
        from notebook.utils import setup_sam_3d_body
        _ESTIMATOR = setup_sam_3d_body(hf_repo_id=HF_REPO_ID)
        _FACES = np.asarray(_ESTIMATOR.faces)
    return _ESTIMATOR, _FACES


def get_faces():
    """Triangle faces for the body mesh (CPU-only consumers use this)."""
    return get_estimator()[1]


@functools.lru_cache(maxsize=8)
def _reconstruct_cached(video_path, idx, bbox_thr):
    """GPU call, memoized per (video, frame, threshold).

    Returns a list of slim, picklable dicts (one per detected person) holding only
    the numpy fields used downstream. Returns None if the frame can't be read.
    """
    from .video import grab_frame  # local import keeps this module import-light

    est, _ = get_estimator()
    frame_rgb = grab_frame(video_path, idx)
    if frame_rgb is None:
        return None

    # process_one_image accepts an RGB array and a bbox_thr kwarg (verified).
    # It returns a LIST, one dict per person, keys include:
    #   bbox, pred_vertices, pred_cam_t, focal_length, pred_keypoints_2d, mask
    people = est.process_one_image(frame_rgb, bbox_thr=bbox_thr)

    slim = []
    for p in people:
        slim.append({
            "bbox": np.asarray(p["bbox"]).reshape(-1)[:4].astype(float),
            "pred_vertices": np.asarray(p["pred_vertices"], dtype=np.float32),
            "pred_cam_t": np.asarray(p["pred_cam_t"], dtype=np.float32).reshape(3),
            "focal_length": float(np.asarray(p["focal_length"]).reshape(-1)[0]),
        })
    return slim


def reconstruct_frame(video_path, idx, bbox_thr=0.85):
    """CPU-cheap wrapper: normalizes the cache key and returns cached people."""
    return _reconstruct_cached(str(video_path), int(idx), round(float(bbox_thr), 3))
