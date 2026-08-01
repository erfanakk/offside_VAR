"""
The ONLY module that touches the GPU. Models are lazily loaded so each stage
pays only for what it needs, and only the detector you actually pick is loaded:

  detect_frame(..., backend) -> ViTDet / RF-DETR / YOLO / SAM3.
  reconstruct_selected()     -> SAM 3D Body: meshes for ONLY the selected boxes.

All detectors return boxes + optional per-player masks, so the UI can show/select
by box or by segment. Build still reconstructs only the selected players.
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
RFDETR_SIZE = os.environ.get("RFDETR_SIZE", "large").lower()
YOLO_HF_REPO_ID = os.environ.get("YOLO_HF_REPO_ID", "Ultralytics/YOLO11")
YOLO_HF_FILENAME = os.environ.get("YOLO_HF_FILENAME", "yolo11n.pt")
SAM3_PROMPT = os.environ.get("SAM3_PROMPT", "person")

VITDET_PERSON = 0   # detectron2 COCO is 0-indexed
RFDETR_PERSON = 1   # rfdetr/COCO is 1-indexed

_DETECTORS = {}     # backend -> loaded model
_ESTIMATOR = None
_FACES = None


def _to_numpy(value):
    """Convert torch or NumPy model output without requiring either at import time."""
    if not hasattr(value, "detach"):
        return np.asarray(value)
    tensor = value.detach().cpu()
    try:
        return tensor.numpy()
    except TypeError as exc:
        if "BFloat16" not in str(exc):
            raise
        return tensor.float().numpy()


# ----------------------------------------------------------------------------
# Detectors (loaded on first use of that backend).
# ----------------------------------------------------------------------------
def _get_vitdet():
    if "vitdet" not in _DETECTORS:
        import torch
        from tools.build_detector import load_detectron2_vitdet
        d = load_detectron2_vitdet()
        _DETECTORS["vitdet"] = d.to("cuda").eval() if torch.cuda.is_available() else d.eval()
    return _DETECTORS["vitdet"]


def _get_rfdetr():
    if "rfdetr" not in _DETECTORS:
        from rfdetr import (RFDETRSegNano, RFDETRSegSmall,        # type: ignore
                            RFDETRSegMedium, RFDETRSegLarge)
        sizes = {"nano": RFDETRSegNano, "small": RFDETRSegSmall,
                 "medium": RFDETRSegMedium, "large": RFDETRSegLarge}
        m = sizes.get(RFDETR_SIZE, RFDETRSegLarge)()
        try:
            m.optimize_for_inference()
        except Exception:
            pass
        _DETECTORS["rfdetr"] = m
    return _DETECTORS["rfdetr"]


def _get_yolo():
    if "yolo" not in _DETECTORS:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        weights = hf_hub_download(repo_id=YOLO_HF_REPO_ID,
                                  filename=YOLO_HF_FILENAME,
                                  token=os.environ.get("HF_TOKEN"))
        _DETECTORS["yolo"] = YOLO(weights)
    return _DETECTORS["yolo"]


def _get_sam3():
    if "sam3" not in _DETECTORS:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model()
        # Keep processor filtering disabled; the UI threshold is applied below.
        _DETECTORS["sam3"] = Sam3Processor(model, confidence_threshold=0.0)
    return _DETECTORS["sam3"]


def _detect_vitdet(frame_rgb, conf):
    import torch
    import detectron2.data.transforms as T
    det = _get_vitdet()
    img_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
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
        if int(classes[k]) != VITDET_PERSON or scores[k] < conf:
            continue
        m = None
        if masks is not None:
            mk = np.asarray(masks[k]).astype(bool)
            m = mk[0] if mk.ndim == 3 else mk
        people.append({"bbox": boxes[k].astype(float),
                       "score": float(scores[k]), "mask": m})
    return people


def _detect_rfdetr(frame_rgb, conf):
    det = _get_rfdetr().predict(frame_rgb, threshold=conf)
    masks = getattr(det, "mask", None)
    people = []
    for k in range(len(det.xyxy)):
        if int(det.class_id[k]) != RFDETR_PERSON:
            continue
        x1, y1, x2, y2 = [float(v) for v in det.xyxy[k]]
        people.append({"bbox": np.array([x1, y1, x2, y2], dtype=float),
                       "score": float(det.confidence[k]),
                       "mask": (np.asarray(masks[k], dtype=bool) if masks is not None else None)})
    return people


def _detect_yolo(frame_rgb, conf):
    from PIL import Image

    result = _get_yolo().predict(source=Image.fromarray(frame_rgb),
                                 conf=conf, verbose=False)[0]
    boxes = result.boxes
    people = []
    for k, cls in enumerate(boxes.cls.int().tolist()):
        if int(cls) != 0:  # COCO person
            continue
        mask = None
        if result.masks is not None:
            mask = _to_numpy(result.masks.data[k]).astype(bool)
        people.append({"bbox": _to_numpy(boxes.xyxy[k]).astype(float),
                       "score": float(_to_numpy(boxes.conf[k])), "mask": mask})
    return people


def _detect_sam3(frame_rgb, conf):
    import contextlib
    import torch
    from PIL import Image

    processor = _get_sam3()
    precision = (torch.autocast("cuda", dtype=torch.bfloat16)
                 if torch.cuda.is_available() else contextlib.nullcontext())
    with precision:
        state = processor.set_image(Image.fromarray(frame_rgb))
        output = processor.set_text_prompt(state=state, prompt=SAM3_PROMPT)
    boxes = _to_numpy(output["boxes"])
    scores = _to_numpy(output["scores"])
    masks = output.get("masks")
    if masks is not None:
        masks = _to_numpy(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
    return [{"bbox": np.asarray(boxes[k], dtype=float),
             "score": float(scores[k]),
             "mask": (None if masks is None else masks[k].astype(bool))}
            for k in range(len(boxes)) if float(scores[k]) >= conf]


@functools.lru_cache(maxsize=16)
def _detect_cached(video_path, idx, conf, backend):
    """Run the chosen detector on one frame; person boxes + masks. Cached per key."""
    from .video import grab_frame
    frame_rgb = grab_frame(video_path, idx)
    if frame_rgb is None:
        return None
    detectors = {"vitdet": _detect_vitdet, "rfdetr": _detect_rfdetr,
                 "yolo": _detect_yolo, "sam3": _detect_sam3}
    people = detectors.get(backend, _detect_vitdet)(frame_rgb, conf)
    people.sort(key=lambda p: (p["bbox"][0], p["bbox"][1]))   # stable L->R order
    return people


def detect_frame(video_path, idx, conf=0.3, backend="vitdet"):
    """CPU-cheap wrapper around the cached detection for the chosen backend."""
    return _detect_cached(str(video_path), int(idx), round(float(conf), 3), backend)


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
    # Providing bboxes reconstructs exactly these people (in order); the FOV
    # estimator still runs for focal_length.
    people = est.process_one_image(frame_rgb, bboxes=boxes)
    slim = []
    for p in people:
        kp = p.get("pred_keypoints_3d")
        slim.append({
            "bbox": np.asarray(p["bbox"]).reshape(-1)[:4].astype(float),
            "pred_vertices": np.asarray(p["pred_vertices"], dtype=np.float32),
            "pred_cam_t": np.asarray(p["pred_cam_t"], dtype=np.float32).reshape(3),
            "focal_length": float(np.asarray(p["focal_length"]).reshape(-1)[0]),
            "pred_keypoints_3d": (None if kp is None
                                  else np.asarray(kp, dtype=np.float32)),
        })
    return slim


def reconstruct_selected(video_path, idx, boxes):
    """Reconstruct meshes for the selected boxes (list of [x1,y1,x2,y2])."""
    boxes_key = tuple(int(round(v)) for b in boxes for v in b[:4])
    return _reconstruct_cached(str(video_path), int(idx), boxes_key)
