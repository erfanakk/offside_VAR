"""
Video helpers — pure CPU. Frames are seeked on demand (no bulk extraction),
so scrubbing a long clip stays cheap and never touches the GPU.
"""

import cv2


def probe_video(video_path):
    """Return (frame_count, fps) for a clip. fps falls back to 25 if unknown."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return n, fps


def grab_frame(video_path, idx):
    """Seek to an exact frame index and return it as an RGB array (or None)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
