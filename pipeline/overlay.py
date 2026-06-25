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
