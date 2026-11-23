"""Shared geometric scale for the retained wrist return policy; no release gate."""
import math


def geometry(controller):
    ctx = getattr(controller, '_registered_alignment', {}) or {}
    reg = ctx.get('registration') or {}
    tip = ctx.get('tip') or {}
    hole = reg.get('measuredHole') or next((h for h in reg.get('layout', [])
        if h.get('bodyHole') == reg.get('hole') and h.get('measured', True)), {})
    radius = tip.get('shaftRadiusM', tip.get('radiusM'))
    hole_radius = hole.get('radiusM')
    scales = [float(r) for r in (radius, hole_radius) if r is not None and math.isfinite(r) and r > 0]
    # Quarter of the smallest available diameter: a local mouth-height condition,
    # independent of global coordinates, catalogue identity and scoring thresholds.
    height = min(scales)/2 if scales else None
    return dict(tip=tip, hole=hole, registration=reg, heightLimitM=height)


def limits(controller, report):
    g = geometry(controller)
    bounds = dict(heightLimitM=g['heightLimitM'], lateralLimitM=float(report.get('toleranceM', .002)),
                  axisLimitDeg=8., heightBasis='quarter_smallest_available_shaft_or_mouth_diameter')
    return bounds
