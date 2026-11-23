import math
import numpy as np
import pytest
from steward.ego.hole_registration import multiview_interface_box
from steward.ego.priors import screw_holes, SCREW_HOLE_FRAMES


def views(angle=0):
    rng = np.random.default_rng(4)
    points = rng.uniform([-.09,-.0225,0], [.09,.0225,.08], (1200,3))
    # Include real outline edges, as complementary views do.
    points[:200,1] = -.0225; points[200:400,1] = .0225
    points[400:500,0] = -.09; points[500:600,0] = .09
    c,s = math.cos(angle), math.sin(angle)
    points[:,:2] = points[:,:2] @ np.array([[c,s],[-s,c]])
    points += [1,2,.75]
    return [dict(points=points.copy(), camera=[1+math.cos(a),2+math.sin(a),1.3],
                 stamp=i+1, range=1.1, truncated=False, invalidDepthFraction=0)
            for i,a in enumerate((0,.5,-.5))]


def fit(rows, **kw):
    return multiview_interface_box(rows, SCREW_HOLE_FRAMES['airplane_fuselage'],
                                  screw_holes('airplane_fuselage'), **kw)


@pytest.mark.parametrize('angle',[0,.31,1.3,-1.5])
def test_world_rotated_multiview_fit_no_world_axis_snapping(angle):
    result = fit(views(angle))
    assert result is not None
    assert abs((result[2]-angle+math.pi/2)%math.pi-math.pi/2) < math.radians(2)
    assert result[-1]['views']==3
    assert result[-1]['directedYawObserved'] is False


def test_repeated_camera_is_not_multiview():
    rows=views()
    for r in rows: r['camera']=rows[0]['camera']
    assert fit(rows) is None


@pytest.mark.parametrize('bad',['old','far','cut','moving'])
def test_bad_registration_not_frozen(bad):
    rows=views()
    if bad=='old': assert fit(rows,after=10) is None; return
    if bad=='far': rows[0]['range']=3
    if bad=='cut': rows[0]['truncated']=True
    if bad=='moving': rows[0]['points'] += [.2,0,0]
    assert fit(rows) is None
