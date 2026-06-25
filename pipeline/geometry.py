"""
Placement geometry — pure CPU, copied verbatim from the verified Colab notebook
(sections 4 & the placement cell). Do NOT rewrite this math:

  world_verts = pred_vertices + pred_cam_t            (shared camera frame)
  goal dir    = back-projected vanishing point of two clicked parallel lines
  ground      = SVD on the lowest ~3% feet vertices (camera Y is down)
  field frame = ez=up, ey=goal dir in-plane, ex=ey x ez (the offside axis)
  per player  : drop each player's own feet to Z=0 (not one global shift)
  offside     : constant-X plane, parallel to the goal line by construction
"""

import numpy as np
import plotly.graph_objects as go


def goal_dir_from_lines(line_pts, focal, w, h):
    """Two clicked goal-parallel lines -> goal-line direction in camera coords."""
    def homog_line(p, q):
        return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])
    lp = np.asarray(line_pts, dtype=float)
    l1 = homog_line(lp[0], lp[1])
    l2 = homog_line(lp[2], lp[3])
    vp = np.cross(l1, l2)
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]])
    d_img = (np.array([vp[0], vp[1], 0.0]) if abs(vp[2]) < 1e-9
             else np.array([vp[0] / vp[2], vp[1] / vp[2], 1.0]))
    g = np.linalg.inv(K) @ d_img
    return g / np.linalg.norm(g)


def world_verts(p):
    """Vertices in the shared camera frame = local mesh + per-player cam translation."""
    return np.asarray(p["pred_vertices"]) + np.asarray(p["pred_cam_t"]).reshape(1, 3)


def place_players(people, selected_ids, goal_dir_cam, flip_up=False):
    """Place selected meshes on a common field frame: X=offside axis, Y=goal line, Z=up."""
    def feet_pts(Vc, frac=0.03):
        thr = np.quantile(Vc[:, 1], 1 - frac)   # camera Y is down -> feet = largest Y
        return Vc[Vc[:, 1] >= thr]

    def head_pts(Vc, frac=0.03):
        thr = np.quantile(Vc[:, 1], frac)       # camera Y is down -> head = smallest Y
        return Vc[Vc[:, 1] <= thr]

    # Up axis = the players' OWN averaged vertical (each player's feet->head),
    # which is what actually makes them stand upright. A plane fit through the
    # feet *positions* (the notebook's original approach) tilts whenever feet land
    # unevenly across a clip, leaving players leaning with one foot pinned at Z=0.
    ups = []
    for i in selected_ids:
        Vc = world_verts(people[i])
        u = head_pts(Vc).mean(0) - feet_pts(Vc).mean(0)
        ups.append(u / (np.linalg.norm(u) + 1e-9))
    n = np.mean(ups, 0)
    n /= np.linalg.norm(n) + 1e-9
    if flip_up:
        n = -n

    feet_all = np.vstack([feet_pts(world_verts(people[i])) for i in selected_ids])
    o = feet_all.mean(0)                         # origin at the feet centroid

    g = goal_dir_cam - (goal_dir_cam @ n) * n
    g /= np.linalg.norm(g)
    ez, ey = n, g
    ex = np.cross(ey, ez); ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    Rwf = np.stack([ex, ey, ez], axis=1)

    placed = {i: (world_verts(people[i]) - o) @ Rwf for i in selected_ids}
    for i in placed:                       # drop each player's feet to Z=0 individually
        placed[i][:, 2] -= placed[i][:, 2].min()
    return placed


def offside_plane_x(placed, attack_sign, defender_ids):
    """X of the offside line = 2nd-last defender's forward-most point.

    Forward-most is measured in the ATTACK direction (fwd = attack_sign * X), the
    same quantity build_scene uses for the per-player verdict, so the drawn plane
    and the colors always agree — matches the notebook. Falls back to the only
    defender, then to the scene's median X if no defenders are labeled.
    """
    dset = set(int(d) for d in (defender_ids or []))
    fmost = {i: float((attack_sign * placed[i][:, 0]).max()) for i in placed}
    dfwd = sorted([fmost[i] for i in placed if i in dset], reverse=True)
    if len(dfwd) >= 2:
        return attack_sign * dfwd[1]
    if dfwd:
        return attack_sign * dfwd[0]
    return float(np.median(np.vstack(list(placed.values()))[:, 0]))


def build_scene(placed, faces, plane_x, attack_sign, defender_ids, too_close=0.30):
    allP = np.vstack(list(placed.values()))
    x0, x1 = allP[:, 0].min() - 4, allP[:, 0].max() + 4
    y0, y1 = allP[:, 1].min() - 4, allP[:, 1].max() + 4

    def fwd(x):
        return attack_sign * x

    fmost = {i: float(fwd(placed[i][:, 0]).max()) for i in placed}

    fig = go.Figure()
    # pitch
    fig.add_trace(go.Mesh3d(x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[0, 0, 0, 0],
                            i=[0, 0], j=[1, 2], k=[2, 3], color="seagreen",
                            opacity=0.5, showlegend=False))
    # offside plane
    if plane_x is not None:
        fig.add_trace(go.Mesh3d(x=[plane_x] * 4, y=[y0, y1, y1, y0], z=[0, 0, 3, 3],
                                i=[0, 0], j=[1, 2], k=[2, 3], color="red",
                                opacity=0.3, showlegend=False))
    # players
    palette = ["crimson", "royalblue", "gold", "darkorange", "mediumpurple",
               "deepskyblue", "hotpink", "mediumspringgreen", "tomato", "slateblue"]
    dset = set(int(d) for d in (defender_ids or []))
    any_off = False
    line_fwd = fwd(plane_x) if plane_x is not None else None
    for n_, i in enumerate(placed):
        if i in dset:
            col, lab = "royalblue", "defender"
        elif line_fwd is None:
            col, lab = palette[n_ % len(palette)], ""
        else:
            m = fmost[i] - line_fwd
            if m > too_close:
                col, lab, any_off = "red", f"OFFSIDE +{m:.2f}m", True
            elif m < -too_close:
                col, lab = "seagreen", f"onside {m:.2f}m"
            else:
                col, lab = "orange", f"close {m:+.2f}m"
        V = placed[i]
        fig.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2],
                                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                color=col, opacity=1.0, name=f"#{i} {lab}"))

    title = "VAR — OFFSIDE" if any_off else "VAR — NO OFFSIDE"
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=22, color="white")),
        paper_bgcolor="#0b1f3a", height=640, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(aspectmode="data", bgcolor="#0b1f3a",
                   xaxis_title="X offside axis (m)", yaxis_title="Y goal line (m)",
                   zaxis_title="Z (m)",
                   camera=dict(eye=dict(x=0, y=-2.2, z=1.2), up=dict(x=0, y=0, z=1))))
    return fig
