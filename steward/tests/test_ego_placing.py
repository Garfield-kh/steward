"""ESim placing: margins from the footprint of the load as it is turned."""

from __future__ import annotations

import math

import numpy as np
import pytest


def test_a_load_turned_45_degrees_is_kept_wholly_on_the_support():
    """D2: keep = max(hx, hy) is not a turned load's half width: at 45 deg a 0.20 m half-wide load
    passed a 0.28 m half-wide support and overhung it by up to 8 cm."""
    from steward.assets.generators import steward_bot
    from steward.ego.esim import Esim, PredictionError, entity_box
    from steward.ego.estimator import EgoEstimator, Track

    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)

    def track(oid, cat, pos, size):
        return Track(id=oid, category_counts={cat: 5.0}, pos=np.array(pos, float), var=np.full(3, 1e-10), yaw=0.0,
                     yaw_var=1e-6, size=np.array(size, float), observations=10, first_seen=0.0, last_seen=0.0, existence=1.0)

    est.tracks = {"e:4": track("e:4", "table", [0.8, 0.0, 0.375], [0.56, 0.56, 0.75]),
                  "e:2": track("e:2", "box", [0.3, -0.3, 0.8], [0.40, 0.40, 0.05])}
    es = est.to_scene()
    table, box = es.find("e:4"), es.find("e:2")
    esim = Esim(model)
    with pytest.raises(PredictionError):
        esim._choose_spot(es, table, entity_box(table)[0], box, entity_box(box)[0], [0.0, 0.0], math.pi / 4, 0.0)
    spot, _ = esim._choose_spot(es, table, entity_box(table)[0], box, entity_box(box)[0], [0.0, 0.0], 0.0, 0.0)
    assert spot == pytest.approx((0.8, 0.0))


def test_the_place_risk_uses_the_turned_footprint_too():
    """D2 (review): the spot check used the turned footprint but the receipt's edge risk still took
    max(hx, hy): a 0.20 m half-wide load turned 45 deg, centred on a 0.31 m half-wide support, is
    2.7 cm from the edge, and was rated as 11 cm."""
    from steward.ego.esim import Esim
    from steward.ego.planner import Box2D

    sbox = Box2D(id="t", cx=0.0, cy=0.0, yaw=0.0, hx=0.31, hy=0.31)
    load = Box2D(id="l", cx=0.0, cy=0.0, yaw=0.0, hx=0.20, hy=0.20)
    ex, ey = Esim._turned_half(load, math.pi / 4, sbox.yaw)
    assert ex == pytest.approx(0.2828, abs=1e-3) and ey == pytest.approx(0.2828, abs=1e-3)
    assert Esim._edge_margin(sbox, (0.0, 0.0), ex, ey) == pytest.approx(0.0272, abs=1e-3)
    # unturned, a long thin load is as far from the long edges as its short side allows
    thin = Box2D(id="n", cx=0.0, cy=0.0, yaw=0.0, hx=0.25, hy=0.02)
    assert Esim._edge_margin(sbox, (0.0, 0.0), *Esim._turned_half(thin, 0.0, 0.0)) == pytest.approx(0.06)
