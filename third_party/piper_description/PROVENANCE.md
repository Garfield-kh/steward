# Piper description (vendored)

Upstream: https://github.com/agilexrobotics/piper_ros
Pinned commit (shallow tip): `ac41fcbcdda598f01b51cf6175ed9a24d0dacadc` (2026-10-30)
Copied subset: `src/piper_description/urdf/` only.

Meshes (`src/piper_description/meshes/`, ~60 MB of STL/DAE) are NOT vendored.
Fetch them when mesh-accurate visuals are needed:

    git clone --depth 1 https://github.com/agilexrobotics/piper_ros /tmp/piper_ros
    cp -r /tmp/piper_ros/src/piper_description/meshes third_party/piper_description/

Kinematic facts (from `piper_description_v100_camera.urdf`, verified 2026-10-30):

- 6 revolute joints (`joint1..joint6`), effort limit 100 N·m each;
  e.g. joint1 ±2.618 rad, joint2 0..3.14, joint3 -2.697..0, joint4 ±1.832,
  joint5 ±1.22, joint6 ±3.14.
- Parallel gripper: two prismatic finger joints (`joint7` 0..0.05 m,
  `joint8` -0.05..0 m) — position-controlled, intermediate aperture valid.
- `hand_camera_joint` (fixed): official wrist-camera mount; variants
  `piper_description_with_camera*.xacro` and `*_v100_camera.urdf` ship the
  camera link.
- The `*_left.xacro` / `*_right.xacro` variants are the two installed arms.

Used by the `agilex` branch as the kinematic source of truth for the
Cobot Magic embodiment twin (`steward/embodiments/cobot-magic.yaml`).
