---
kind: upgrade-guide
description: "Cobot raw lift coordinates use displacement from the lowest position and USD-derived arm mounts."
---

# Cobot lift coordinates and arm mounts

English | [中文](guide.zh.md)

## Change

Newly generated `cobot-magic` models use manufacturer USD arm/lift kinematics. Raw `joints.lift` and the Cobot `lift` skill's `height` parameter change from carriage height above the chassis top, `[0.35, 1.05]` m, to displacement above the lowest position, `[0, 0.60]` m. The initial value changes from `0.70` to `0`. The Ego planar arm's virtual `lift` remains TCP height.

## Migration

1. Recreate saved Cobot entities; embedded models are not updated automatically. World and Ego Runtime must use the same model.
2. Update raw lift commands to `[0, 0.60]`. For arm-root height `H`, use `H - 0.9732298` metres if within range. This floor offset is inferred from wheel geometry, not hardware calibration. Leave virtual TCP-height commands unchanged.
3. Check the new joint ranges and mounts before reusing trajectories. The [embodiment profile](../../../../steward/embodiments/cobot-magic.yaml) records mappings and assumptions.
