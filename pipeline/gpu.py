"""
The ONLY module that touches the GPU. Two lazily-loaded models, kept separate so
each stage pays only for what it needs:

  detect_frame()        -> RF-DETR-Seg: boxes + per-player masks (the Detect step).
  reconstruct_selected()-> SAM 3D Body: meshes for ONLY the selected boxes (Build).

Splitting them means Detect runs a light detector (fast, every frame) while the
heavy mesh reconstruction runs once, at Build, for ~3 players instead of ~30.
Both results are cached so re-clicking never re-hits the GPU.
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
RFDETR_SIZE = os.environ.get("RFDETR_SIZE", "medium").lower()

COCO_PERSON = 1   # rfdetr/COCO is 1-indexed: person == 1

_DETECTOR = None
_ESTIMATOR = None
_FACES = None


# ----------------------------------------------------------------------------
# Detector — RF-DETR-Seg (boxes + masks). Loaded on the first Detect.
# ----------------------------------------------------------------------------
def get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        from rfdetr import (RFDETRSegNano, RFDETRSegSmall,        # type: ignore
                            RFDETRSegMedium, RFDETRSegLarge)
        sizes = {"nano": RFDETRSegNano, "small": RFDETRSegSmall,
                 "medium": RFDETRSegMedium, "large": RFDETRSegLarge}
        Model = sizes.get(RFDETR_SIZE, RFDETRSegMedium)
        _DETECTOR = Model()
        try:
            _DETECTOR.optimize_for_inference()
        except Exception:
            pass
    return _DETECTOR


@functools.lru_cache(maxsize=16)
def _detect_cached(video_path, idx, conf):
    """Run RF-DETR-Seg on one frame; keep person boxes + masks. Cached per frame."""
    from .video import grab_frame
    frame_rgb = grab_frame(video_path, idx)
    if frame_rgb is None:
        return None
    det = get_detector().predict(frame_rgb, threshold=conf)
    masks = getattr(det, "mask", None)
    people = []
    for k in range(len(det.xyxy)):
        if int(det.class_id[k]) != COCO_PERSON:
            continue
        x1, y1, x2, y2 = [float(v) for v in det.xyxy[k]]
        people.append({
            "bbox": np.array([x1, y1, x2, y2], dtype=float),
            "score": float(det.confidence[k]),
            "mask": (np.asarray(masks[k], dtype=bool) if masks is not None else None),
        })
    return people


def detect_frame(video_path, idx, conf=0.4):
    """CPU-cheap wrapper around the cached RF-DETR detection."""
    return _detect_cached(str(video_path), int(idx), round(float(conf), 3))


# ----------------------------------------------------------------------------
# Reconstructor — SAM 3D Body. Loaded on the first Build.
# ----------------------------------------------------------------------------
def get_estimator():
    """Lazy-load the SAM 3D Body estimator once; returns (estimator, faces)."""
    global _ESTIMATOR, _FACES
    if _ESTIMATOR is None:
        from huggingface_hub import login
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
        from notebook.utils import setup_sam_3d_body
        _ESTIMATOR = setup_sam_3d_body(hf_repo_id=HF_REPO_ID)
        _FACES = np.asarray(_ESTIMATOR.faces)
    return _ESTIMATOR, _FACES


def get_faces():
    return get_estimator()[1]


@functools.lru_cache(maxsize=16)
def _reconstruct_cached(video_path, idx, boxes_key):
    """Reconstruct ONLY the given boxes. boxes_key is a hashable tuple of int xyxy."""
    from .video import grab_frame
    est, _ = get_estimator()
    frame_rgb = grab_frame(video_path, idx)
    if frame_rgb is None:
        return None
    boxes = np.array(boxes_key, dtype=np.float32).reshape(-1, 4)
    # Providing bboxes skips RF-DETR's role inside SAM-3D and reconstructs exactly
    # these people (in order); the FOV estimator still runs for focal_length.
    people = est.process_one_image(frame_rgb, bboxes=boxes)
    slim = []
    for p in people:
        slim.append({
            "bbox": np.asarray(p["bbox"]).reshape(-1)[:4].astype(float),
            "pred_vertices": np.asarray(p["pred_vertices"], dtype=np.float32),
            "pred_cam_t": np.asarray(p["pred_cam_t"], dtype=np.float32).reshape(3),
            "focal_length": float(np.asarray(p["focal_length"]).reshape(-1)[0]),
        })
    return slim


def reconstruct_selected(video_path, idx, boxes):
    """Reconstruct meshes for the selected boxes (list of [x1,y1,x2,y2])."""
    boxes_key = tuple(int(round(v)) for b in boxes for v in b[:4])
    return _reconstruct_cached(str(video_path), int(idx), boxes_key)
