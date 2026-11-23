"""Transport must not select the opposite member of a symmetric multi-hole fit."""
import math
from types import SimpleNamespace
import pytest
from steward.ego.esim import Esim
from steward.ego.hole_registration import register_hole_template
from steward.math3d import wrap_angle
from steward.scene.schema import Est
from test_hole_registration import entity, pair, template


@pytest.mark.parametrize('transport', [0., 1.9, -2.4, math.pi])
@pytest.mark.parametrize('before', [.2, -2.9])
def test_pair_chooses_nearest_pretransport_axis_frame(transport, before):
    holes, _ = register_hole_template(pair(), template((.48, .12)), [.48, .12, .03])
    held = entity('part', holes, before + transport)
    held.state_vars = {'orientation': Est(value={'graspWorldYawRad': before})}
    mate = entity('mate', holes)
    sim = Esim.__new__(Esim)
    _, _, yaw, issues = sim._place_hole_align(None, held, mate, {'alignHoles': True})
    expected = 0. if abs(before) < math.pi/2 else math.pi
    assert wrap_angle(yaw - expected) == pytest.approx(0.)
    assert any('RSI_MATING_FRAME' in x for x in issues)
