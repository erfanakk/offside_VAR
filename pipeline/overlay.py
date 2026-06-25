"""
cv2 overlays — pure CPU. Draws detection boxes and the two goal-parallel lines
the user clicks. Works on RGB arrays in, RGB arrays out (Gradio Image is numpy/RGB).
"""

import cv2
import numpy as np


def annotate_detections(frame_rgb, people, selected_ids=()):
    """Draw numbered boxes; selected players get a thick yellow box, others thin green."""
    sel = set(int(i) for i in selected_ids)
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for i, p in enumerate(people):
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        if i in sel:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 255), 3)   # yellow, thick
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)     # green, thin
        cv2.putText(img, str(i), (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pick_box(people, x, y):
    """Return the index of the smallest box containing (x, y), or None."""
    hit, best_area = None, None
    for i, p in enumerate(people):
        x1, y1, x2, y2 = p["bbox"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            if best_area is None or area < best_area:
                best_area, hit = area, i
    return hit


# palette for per-player mask tints (BGR)
_MASK_COLORS = [(60, 60, 220), (220, 140, 60), (60, 200, 220), (200, 80, 200),
                (80, 200, 120), (220, 200, 80), (140, 100, 240), (90, 180, 60)]


def draw_masks(frame_rgb, people, selected_ids=()):
    """Translucent colored silhouettes; selected players brighter + outlined.

    Falls back to a box for any detection lacking a mask.
    """
    sel = set(int(i) for i in selected_ids)
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for i, p in enumerate(people):
        m = p.get("mask")
        col = _MASK_COLORS[i % len(_MASK_COLORS)]
        is_sel = i in sel
        if m is not None:
            alpha = 0.55 if is_sel else 0.30
            tint = np.zeros_like(img); tint[m] = col
            img[m] = (img[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
            if is_sel:  # outline the selected silhouette
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (0, 230, 255), 3)
        else:
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2),
                          (0, 230, 255) if is_sel else (0, 255, 0), 3 if is_sel else 2)
        x1, y1 = int(p["bbox"][0]), int(p["bbox"][1])
        cv2.putText(img, str(i), (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pick_mask(people, x, y):
    """Index of the player whose mask covers (x, y); ties -> smallest mask.

    Falls back to box hit-testing for detections without a mask.
    """
    x, y = int(x), int(y)
    hit, best_area = None, None
    for i, p in enumerate(people):
        m = p.get("mask")
        if m is not None:
            if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x]:
                area = int(m.sum())
                if best_area is None or area < best_area:
                    best_area, hit = area, i
        else:
            x1, y1, x2, y2 = p["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best_area is None or area < best_area:
                    best_area, hit = area, i
    return hit


def draw_lines(frame_rgb, pts):
    """Draw the clicked points and the (up to two) line segments.

    pts is a list of [x, y]; points 0-1 form line 1 (red), 2-3 form line 2 (yellow).
    """
    if frame_rgb is None:
        return None
    img = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    cols = [(0, 0, 255), (0, 255, 255)]  # BGR: line1 red, line2 yellow
    for k, (px, py) in enumerate(pts):
        cv2.circle(img, (int(px), int(py)), 6, cols[k // 2], -1)
    if len(pts) >= 2:
        cv2.line(img, tuple(map(int, pts[0])), tuple(map(int, pts[1])), cols[0], 2)
    if len(pts) >= 4:
        cv2.line(img, tuple(map(int, pts[2])), tuple(map(int, pts[3])), cols[1], 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
