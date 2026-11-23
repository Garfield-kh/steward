"""Solve a vertical wrist lift against calibrated frustum inequalities."""
import math
import numpy as np
from steward.math3d import matrix_from_quat


def lift_interval(cam, sensor, target):
    r=matrix_from_quat(cam['q']);a=(np.asarray(target)-cam['p']) @ r
    b=np.array([0.,0.,1.]) @ r
    intr=sensor['intrinsics'];f=intr['height']/(2*math.tan(math.radians(intr['fovy'])/2))
    # Ten percent image margin leaves room around the selected centre.
    tx=.4*intr['width']/f;ty=.4*intr['height']/f
    near,far=sensor.get('depthRangeM',[0.,math.inf])
    # Each row is c + d*h >= 0, with local target a - b*h.
    terms=[(a[0]-near,-b[0]),(far-a[0],b[0])]
    for j,t in ((1,tx),(2,ty)):
        for sign in (-1,1):terms.append((t*a[0]+sign*a[j],-t*b[0]-sign*b[j]))
    lo,hi=0.,math.inf
    for c,d in terms:
        if abs(d)<1e-12:
            if c<0:return None
        elif d>0:lo=max(lo,-c/d)
        else:hi=min(hi,-c/d)
    return (lo,hi) if hi>=lo else None


def choose(cam,sensor,target,boxes,known):
    interval=lift_interval(cam,sensor,target)
    row=dict(status='unknown',reason='no_vertical_frustum_solution',liftM=0.)
    if interval is None:return row
    target=np.asarray(target);clearance=0.
    if known:
        length=known['lengthM'];radius=known['radiusM']
        for centre,half,rotation in boxes:
            extent=abs(np.asarray(rotation)) @ np.asarray(half)
            if np.all(abs(np.asarray(centre)[:2]-target[:2])<=extent[:2]+radius):
                clearance=max(clearance,float(target[2]+length+.003-(centre[2]-extent[2])))
    h=max(interval[0],clearance)
    if h>interval[1]:
        h=interval[0];row['reason']='frustum_before_full_clearance'
    else:row['reason']='fk_clearance_in_frustum'
    row.update(status='planned',liftM=float(h),clearanceLiftM=float(clearance),
        feasibleLiftIntervalM=[float(v) if math.isfinite(v) else None for v in interval],
        cameraHeightM=float(cam['p'][2]+h),knownGeometry=known)
    return row
