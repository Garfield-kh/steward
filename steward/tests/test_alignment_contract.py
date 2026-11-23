"""An explicit geometric constraint must not silently become an ordinary placement."""
from types import SimpleNamespace

import pytest

from steward.ego.esim import Esim, PredictionError


def alignment(held_holes=(), mate_holes=()):
    held, mate = SimpleNamespace(id="load"), SimpleNamespace(id="support")
    esim = Esim.__new__(Esim)
    esim._screw_holes = lambda obj: list(held_holes if obj is held else mate_holes)
    scene = SimpleNamespace(find=lambda eid: mate if eid == mate.id else None)
    return lambda params: esim._place_hole_align(scene, held, mate, params)


@pytest.mark.parametrize("params", [{"alignHoles": True}, {"alignTo": "support"}, {"hole": "socket"}])
def test_missing_load_features_reject_explicit_alignment(params):
    with pytest.raises(PredictionError, match="RSI_ALIGN_REQUIRED.*load"):
        alignment()(params)


@pytest.mark.parametrize("params", [{"alignHoles": True}, {"hole": "socket"}])
def test_missing_mate_features_reject_explicit_alignment(params):
    with pytest.raises(PredictionError, match="RSI_ALIGN_REQUIRED.*support"):
        alignment(held_holes=[{"id": "socket", "p": [0, 0, 0]}])(params)


def test_unspecified_alignment_keeps_ordinary_placement():
    assert alignment()({}) == (None, None, None, [])


def test_explicit_opt_out_preserves_temporary_placement():
    assert alignment()({"alignHoles": False, "alignTo": "missing", "hole": "socket"}) == (None, None, None, [])
