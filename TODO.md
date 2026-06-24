# Parked — explicitly out of scope for now (SPEC §7)

These are deliberately NOT built yet. Listed so they aren't lost.

- **Automatic pass-instant detection** — needs ball tracking to replace manual
  frame scrubbing.
- **Exclude arms from the forward-most point** — needs MHR body-part vertex
  labels; today `build_scene`/`offside_plane_x` use the forward-most body vertex
  including arms.
- **three.js broadcast frontend** (fog/bloom/camera moves) replacing Plotly.
- **Team identification by jersey colour** — currently the user labels defenders
  manually.
- **Multi-frame tracking.**

# Known nuances carried from the notebook

- `flip up` toggle exists because the ground-normal sign can invert; if median
  player height < 1.0 m the scene warns to toggle it.
- The offside line uses the **2nd-last** defender's forward-most point (last is
  usually the GK). With only one labeled defender it falls back to that defender;
  with none, to the scene's median X (then drag the plane).
