"""
The ONLY module that touches the GPU. Two lazily-loaded models, kept separate so
each stage pays only for what it needs:

  detect_frame()        -> ViTDet-H Cascade Mask R-CNN: boxes + per-player masks.
  reconstruct_selected()-> SAM 3D Body: meshes for ONLY the selected boxes (Build).

Splitting them means Detect runs only the detector (every frame) while the heavy
mesh reconstruction runs once, at Build, for ~3 players instead of ~30. Both
results are cached so re-clicking never re-hits the GPU.

The detector is the SAME ViTDet-H that SAM-3D-Body bundles (detector_name="vitdet")
— its bundled helper returns boxes only, so here we run it directly and KEEP the
Mask R-CNN instance masks for silhouette selection.
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
DET_PERSON = 0   # detectron2 COCO is 0-indexed: person == 0

_DETECTOR = None
_ESTIMATOR = None
_FACES = None


# ----------------------------------------------------------------------------
# Detector — ViTDet-H Cascade Mask R-CNN (boxes + masks). Loaded on first Detect.
# ----------------------------------------------------------------------------
def get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        import torch
        from tools.build_detector import load_detectron2_vitdet
        d = load_detectron2_vitdet()   # loads config + COCO checkpoint
        _DETECTOR = d.to("cuda").eval() if torch.cuda.is_available() else d.eval()
    return _DETECTOR


@functools.lru_cache(maxsize=16)
def _detect_cached(video_path, idx, conf):
    """Run ViTDet on one frame; keep person boxes + masks. Cached per frame."""
    import torch
    import detectron2.data.transforms as T
    from .video import grab_frame

    frame_rgb = grab_frame(video_path, idx)
    if frame_rgb is None:
        return None
    det = get_detector()
    img_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])   # detectron2 wants BGR
    h, w = img_bgr.shape[:2]

    aug = T.ResizeShortestEdge(short_edge_length=1024, max_size=1024)
    img_t = aug(T.AugInput(img_bgr)).apply_image(img_bgr)
    img_t = torch.as_tensor(img_t.astype("float32").transpose(2, 0, 1))
    with torch.no_grad():
        out = det([{"image": img_t, "height": h, "width": w}])
    inst = out[0]["instances"].to("cpu")

    boxes = inst.pred_boxes.tensor.numpy()
    classes = inst.pred_classes.numpy()
    scores = inst.scores.numpy()
    masks = inst.pred_masks.numpy() if inst.has("pred_masks") else None

    people = []
    for k in range(len(boxes)):
        if int(classes[k]) != DET_PERSON or scores[k] < conf:
            continue
        m = None
        if masks is not None:
            mk = np.asarray(masks[k]).astype(bool)
            m = mk[0] if mk.ndim == 3 else mk      # [1,H,W] or [H,W]
        people.append({"bbox": boxes[k].astype(float),
                       "score": float(scores[k]), "mask": m})
    people.sort(key=lambda p: (p["bbox"][0], p["bbox"][1]))   # stable L->R order
    return people


def detect_frame(video_path, idx, conf=0.4):
    """CPU-cheap wrapper around the cached ViTDet detection."""
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
        kp = p.get("pred_keypoints_3d")
        slim.append({
            "bbox": np.asarray(p["bbox"]).reshape(-1)[:4].astype(float),
            "pred_vertices": np.asarray(p["pred_vertices"], dtype=np.float32),
            "pred_cam_t": np.asarray(p["pred_cam_t"], dtype=np.float32).reshape(3),
            "focal_length": float(np.asarray(p["focal_length"]).reshape(-1)[0]),
            # keypoints (MHR-70 joints) drive arm/hand exclusion for the offside line
            "pred_keypoints_3d": (None if kp is None
                                  else np.asarray(kp, dtype=np.float32)),
        })
    return slim


def reconstruct_selected(video_path, idx, boxes):
    """Reconstruct meshes for the selected boxes (list of [x1,y1,x2,y2])."""
    boxes_key = tuple(int(round(v)) for b in boxes for v in b[:4])
    return _reconstruct_cached(str(video_path), int(idx), boxes_key)
