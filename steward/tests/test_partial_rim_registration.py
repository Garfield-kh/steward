"""Self-contained assembly regressions using bundled inputs and synthetic fixtures."""
from copy import deepcopy
from pathlib import Path
import os
import math
import numpy as np
import pytest
from steward.ego import stationary_entrance_layers as p91,component_registration as cr,support_holes as sh
from steward.ego import support_hole_edges as edge,inhand
CONTEXT=dict(kind='existing_inhand_interface',measuredCentresRequired=2)

@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv('STEWARD_ASSEMBLY_PARTIAL_RIM_REGISTRATION','1');monkeypatch.setenv('STEWARD_ASSEMBLY_STATIONARY_REACQUISITION','1')


def synthetic(defect):
    from steward.math3d import quat_from_matrix
    cam=dict(p=np.array([0.,0.,1.]),q=quat_from_matrix(np.array([[0.,1.,0.],[0.,0.,-1.],[-1.,0.,0.]])))
    height,width=240,320;y,x=np.indices((height,width))
    rgb=np.full((height,width,3),180,np.uint8);depth=np.full((height,width),.5)
    if defect=='shadow':rgb[:]=[180,90,45]
    for cx in (width//2-30,width//2+30):
        core=(x-cx)**2+(y-height//2)**2<12**2;rgb[core]=30
        if defect=='raised':depth[core]=.48
        if defect=='shadow':rgb[core]=[60,30,15]
    if defect=='invalid':depth[:]=0
    raw=dict(rgb=rgb,depth=depth,fovy=60.,header=dict(odometry=dict(sigma=dict(pos=0.))))
    return raw,cam,np.ones((height,width),bool)

@pytest.mark.parametrize('defect',['flat','raised','shadow','invalid'])
def test_flat_shadows_caps_invalid_depth_still_rejected_in_context(defect):
    raw,cam,region=synthetic(defect);diagnostics={}
    candidates=sh.detect_region(raw,cam,region,diagnostics,calibrated=True,measurement_context=CONTEXT)
    assert candidates==[],diagnostics
    assert edge.extract(raw,cam,region,candidates,diagnostics)==[]

def test_one_anchor_never_invents_an_unseen_centre():
    raw,cam,region=synthetic('flat');raw['rgb'][:]=180
    anchor=dict(point=[0.,0.,.5],normal=[0.,0.,1.],radiusM=.01,planeNoiseM=.001)
    assert edge.extract(raw,cam,region,[anchor],{})==[]

@pytest.mark.parametrize('defect',['flat','raised','shadow','invalid'])
def test_partial_edge_negatives_with_a_real_measured_anchor(defect):
    raw,cam,region=synthetic('flat')
    height,width=raw['depth'].shape;y,x=np.indices((height,width))
    first=(x-(width//2-30))**2+(y-height//2)**2<12**2
    second=(x-(width//2+30))**2+(y-height//2)**2<12**2
    raw['depth'][:]=.1;raw['depth'][first]=.11
    if defect=='raised':raw['depth'][second]=.08
    if defect=='shadow':
        raw['rgb'][:]=[180,90,45];raw['rgb'][first]=30;raw['rgb'][second]=[60,30,15]
    if defect=='invalid':
        raw['depth'][(x-(width//2+30))**2+(y-height//2)**2<18**2]=0
    diagnostics={}
    anchors=sh.detect_region(raw,cam,region,diagnostics,calibrated=True,measurement_context=CONTEXT)
    assert len(anchors)==1,diagnostics
    assert anchors[0]['depthEvidence']=='resolved_recess'
    assert edge.extract(raw,cam,region,anchors,diagnostics)==[],diagnostics
