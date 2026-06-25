"""
Auto-PROPOSE the two goal-parallel lines (CPU, OpenCV only).

Fully-automatic goal-direction detection is unreliable on broadcast frames
(a full pitch homography can score "confident" yet be degenerate), so this only
*proposes*: it detects the white pitch lines, clusters them by vanishing point
into the two pitch line-families, and returns the 2 longest lines of each. The UI
shows one family, lets the user flip to the other, or redraw by hand.

Line detection (field/white masks + Hough merge) is ported from the project's
backend pitch-registration code, which is already proven on this footage.
"""

import numpy as np
import cv2


# ----------------------------- masks -------------------------------------- #
def field_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = ((h > 30) & (h < 95) & (s > 40) & (v > 40)).astype(np.uint8) * 255
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(green)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return mask


def white_mask(bgr, fmask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    w = ((v > 150) & (s < 60)).astype(np.uint8) * 255
    w = cv2.bitwise_and(w, fmask)
    return cv2.morphologyEx(w, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


# ----------------------------- line tools --------------------------------- #
def _seg_angle(seg):
    x1, y1, x2, y2 = seg
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0


def _line_from_seg(seg):
    x1, y1, x2, y2 = seg
    l = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
    return l / (np.hypot(l[0], l[1]) or 1.0)


def _intersect(l1, l2):
    p = np.cross(l1, l2)
    return None if abs(p[2]) < 1e-9 else np.array([p[0] / p[2], p[1] / p[2]])


def merge_lines(wmask):
    """Cluster Hough segments into merged pitch lines (angle + signed distance)."""
    segs = cv2.HoughLinesP(wmask, 1, np.pi / 180, threshold=70,
                           minLineLength=70, maxLineGap=30)
    if segs is None:
        return []
    segs = segs[:, 0]
    used = np.zeros(len(segs), bool)
    merged = []
    for i in range(len(segs)):
        if used[i]:
            continue
        ai, li = _seg_angle(segs[i]), _line_from_seg(segs[i])
        group = [segs[i]]; used[i] = True
        for j in range(i + 1, len(segs)):
            if used[j]:
                continue
            da = abs(ai - _seg_angle(segs[j]))
            if min(da, 180 - da) > 6:
                continue
            mid = np.array([(segs[j][0] + segs[j][2]) / 2,
                            (segs[j][1] + segs[j][3]) / 2, 1.0])
            if abs(li @ mid) < 14:
                group.append(segs[j]); used[j] = True
        pts = np.array([[s[0], s[1]] for s in group] +
                       [[s[2], s[3]] for s in group], dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        l = _line_from_seg((x0, y0, x0 + vx, y0 + vy))
        proj = (pts - [x0, y0]) @ np.array([vx, vy])
        a = np.array([x0, y0]) + proj.min() * np.array([vx, vy])
        b = np.array([x0, y0]) + proj.max() * np.array([vx, vy])
        merged.append(dict(line=l, a=a, b=b, length=float(np.hypot(*(b - a)))))
    return merged


# ----------------------- vanishing-point families ------------------------- #
def _line_dir(l):
    d = np.array([-l[1], l[0]])
    return d / (np.linalg.norm(d) + 1e-9)


def _vp_inliers(lines, vp, thr_deg=2.5):
    inl = []
    for k, m in enumerate(lines):
        mid = (m["a"] + m["b"]) / 2.0
        to_vp = vp - mid
        n = np.linalg.norm(to_vp)
        if n < 1e-6:
            inl.append(k); continue
        cosang = abs(_line_dir(m["line"]) @ (to_vp / n))
        if np.degrees(np.arccos(min(cosang, 1.0))) < thr_deg:
            inl.append(k)
    return inl


def _best_family(lines):
    """VP supported by the most total line-length; returns (inlier_idxs, vp)."""
    best = None
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            vp = _intersect(lines[i]["line"], lines[j]["line"])
            if vp is None:
                continue
            inl = _vp_inliers(lines, vp)
            score = sum(lines[k]["length"] for k in inl)
            if best is None or score > best[0]:
                best = (score, inl, vp)
    return (best[1], best[2]) if best else (None, None)


def _family_points(fam):
    """4 points (2 longest lines, 2 endpoints each): [l1a, l1b, l2a, l2b]."""
    fam = sorted(fam, key=lambda m: -m["length"])[:2]
    pts = []
    for m in fam:
        pts += [[float(m["a"][0]), float(m["a"][1])],
                [float(m["b"][0]), float(m["b"][1])]]
    return pts


def propose_lines(frame_rgb):
    """Return up to 2 families, each as {"pts": [[x,y]*4], "vp": [x,y], "n": int}.

    families[0] is the larger (by total length); the goal-parallel one may be
    either — the UI lets the user flip. Empty list if too few lines are found.
    """
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    merged = merge_lines(white_mask(bgr, field_mask(bgr)))
    longl = sorted([m for m in merged if m["length"] > 80],
                   key=lambda m: -m["length"])[:14]
    if len(longl) < 2:
        return []

    fams = []
    inl, vp = _best_family(longl)
    if inl is None:
        return []
    fam1 = [longl[k] for k in inl]
    fams.append({"pts": _family_points(fam1), "vp": vp.tolist(), "n": len(fam1)})

    rest = [longl[k] for k in range(len(longl)) if k not in inl]
    if len(rest) >= 2:
        inl2, vp2 = _best_family(rest)
        if inl2 is not None:
            fam2 = [rest[k] for k in inl2]
            if len(fam2) >= 2:
                fams.append({"pts": _family_points(fam2),
                             "vp": vp2.tolist(), "n": len(fam2)})
    return fams
