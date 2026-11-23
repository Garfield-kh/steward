"""Raised contact heads and exposed distal tips from one native RGB-D frame.

No support-plane prerequisite: a long carried body can occlude that plane.
Only a visible narrow distal section is a tip; a housing edge is not one.
"""
import math
import numpy as np
from steward.ego import wrist_observation as wo
from steward.ego.support_holes import components


def convex_boundary(points):
    """Monotone hull in the horizontal plane; invariant to world yaw."""
    points=sorted(set(map(tuple,points)))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def half(items):
        out=[]
        for p in items:
            while len(out)>=2 and cross(out[-2],out[-1],p)<=0:out.pop()
            out.append(p)
        return out
    return np.asarray(half(points)[:-1]+half(reversed(points))[:-1])


def measure(controller, sample, target, opening, diagnostics, *, contact=False):
    raw, cam = sample
    region = wo.window(raw, cam, target, opening)
    est = controller.rt.estimator; h = raw['header']
    grip = np.asarray(est.gripper_position(est.held_gripper, h['joints'], h['odometry']))
    region |= wo.window(raw, cam, grip, opening)
    depth = wo.local_median(raw['depth'], region)
    raw = dict(raw, depth=np.where(raw['depth'] > 0, depth, 0))
    p, y, x = wo.cloud(raw, cam, region)
    if len(p) < 48: return None, []
    focal = depth.shape[0]/(2*math.tan(math.radians(raw['fovy'])/2))
    noise = 2*float(np.median(depth[y,x]))/focal
    xyz = np.full((*depth.shape,3), np.nan); xyz[y,x] = p
    with controller.rt._lock:
        previous = est.joints, est.odometry
        try:
            est.joints, est.odometry = h['joints'], h['odometry']
            bodies = list(est.body_boxes())
        finally: est.joints, est.odometry = previous
    valid = np.ones(len(p), bool)
    for c,half,r in bodies:
        valid &= ~np.all(abs((p-c) @ r) <= np.asarray(half)+noise,axis=1)
    mask = np.zeros(depth.shape,bool); mask[y[valid],x[valid]]=True
    # Horizontal height bands retain tilted/noisy disk tops that pixel normal
    # thresholding would fragment. Reject curved sides by a local plane fit.
    # A disk radius must span four depth/pixel uncertainty units to resolve
    # both its plateau and boundary; tiny horizontal edge fragments stay unknown.
    caps=[]; cap_candidates=[]
    ztarget=float(target[2])
    for z in np.arange(ztarget-opening/2,ztarget+opening/2,noise):
        band=mask & (abs(xyz[:,:,2]-z)<=noise)
        for pixels in components(band):
            if len(pixels)<24:continue
            q=xyz[pixels[:,1],pixels[:,0]]
            centre=np.mean(q,axis=0); xy=q[:,:2]-centre[:2]
            _,sv,_=np.linalg.svd(xy,full_matrices=False)
            if sv[1]<=0 or sv[0]>1.8*sv[1]:continue
            radius=float(np.quantile(np.linalg.norm(xy,axis=1),.95))
            if radius<4*noise or radius>opening/4:continue
            slope=np.linalg.lstsq(np.c_[xy,np.ones(len(xy))],q[:,2]-centre[2],rcond=None)[0]
            if np.linalg.norm(slope[:2])>math.tan(math.pi/6):continue
            # Require resolved circular boundary curvature, not a small
            # horizontal slice of a polygon or the edge of a support plate.
            boundary=convex_boundary(q[:,:2])
            from steward.ego.entry_layout import circle
            fitted=circle(boundary,noise)
            if fitted is None:continue
            centre[:2],radius,_=fitted
            if radius<4*noise or radius>opening/4:continue
            # Reject a height slice through a wall: its surface continues just
            # above the putative cap inside the measured footprint.
            radial=np.linalg.norm(p[:,:2]-centre[:2],axis=1)
            above=(radial<radius)&(p[:,2]>centre[2]+2*noise)&(p[:,2]<centre[2]+radius)
            if np.count_nonzero(above)>len(q)/4:continue
            radial=np.linalg.norm(p[:,:2]-centre[:2],axis=1)
            ring=p[(radial>radius+noise)&(radial<2*radius+noise)&(p[:,2]<centre[2]+noise)]
            if len(ring)<24 or np.quantile(ring[:,2],.25)>=centre[2]-2*noise:continue
            normal=np.r_[-slope[:2],1.];normal/=np.linalg.norm(normal)
            cap_candidates.append((len(q),dict(point=centre.tolist(),normal=normal.tolist(),radiusM=radius,
                uncertaintyM=2*noise,source='same_frame_raised_plateau')))
    for _,cap in sorted(cap_candidates,key=lambda x:-x[0]):
        if not any(np.linalg.norm(np.asarray(cap['point'])-h['point'])<cap['radiusM']+h['radiusM'] for h in caps):
            caps.append(cap)
    # Depth-connected foreground components. Allow long loads outside the jaw
    # sphere, but require a component to approach the measured gripper region.
    edges=np.zeros_like(mask)
    for axis in (0,1):
        for shift in (-1,1):
            edges |= np.linalg.norm(xyz-np.roll(xyz,shift,axis=axis),axis=2)>3*noise
    loadmask=mask & ~edges
    groups=[]
    for pixels in components(loadmask):
        if len(pixels)<24:continue
        q=xyz[pixels[:,1],pixels[:,0]]
        lo,hi=np.quantile(q[:,2],[.02,.98])
        groups.append(dict(q=q,pixels=pixels,lo=lo,hi=hi,
            xy=np.mean(q[:,:2],axis=0),width=float(2*np.quantile(np.linalg.norm(q[:,:2]-q[:,:2].mean(0),axis=1),.98))))
    candidates=[]; diagnostics['tipCandidates']=[]
    for body in groups:
        q=body['q'];span=body['hi']-body['lo']
        if span<8*noise or span<body['width']:continue
        if np.min(np.linalg.norm(q-grip,axis=1))>opening:continue
        # A detached image component may still be a protruding bit: visible
        # housing hides its junction. Require axial overlap and a gap smaller
        # than the measured section, never extrapolate the unseen endpoint.
        for end in groups:
            if end is body:continue
            gap=body['lo']-end['hi'];length=end['hi']-end['lo']
            if not (length>3*noise and -noise<gap<max(length,body['width']/2)):continue
            if end['width']>=body['width']/2:continue
            if np.linalg.norm(end['xy']-body['xy'])>body['width']/2:continue
            if end['lo']<ztarget+(-2 if contact else 2)*noise:continue
            bottom=end['q'][end['q'][:,2]<=end['lo']+2*noise]
            tip=np.r_[np.mean(bottom[:,:2],axis=0),end['lo']]
            pixels=end['pixels']
            if np.any(pixels.min(0)<2) or np.any(pixels.max(0)>=np.array(depth.shape[::-1])-2):continue
            if any(np.all(abs((tip-c)@r)<=np.asarray(half)+3*noise) for c,half,r in bodies):continue
            diagnostics['tipCandidates'].append(dict(spanM=float(span),sectionLengthM=float(length),gapM=float(gap)))
            candidates.append(dict(tip=tip.tolist(),head=np.r_[body['xy'],body['hi']].tolist(),
                uncertaintyM=2*noise,shaftRadiusM=end['width']/2,
                source='same_frame_exposed_distal_section'))
    diagnostics.update(noiseM=noise,occupiedHeadCandidates=len(caps),resolvedTips=len(candidates),heads=caps)
    feature=candidates[0] if len(candidates)==1 else None
    if feature:
        caps=[c for c in caps if c['point'][2]<feature['tip'][2]+(1 if contact else -1)*feature['uncertaintyM']]
    return feature,caps


def observe(controller, sample, target, opening, registration):
    from steward.ego import entry_layout as el, tip_entry_servo as servo
    diagnostics={}
    feature,caps=measure(controller,sample,target,opening,diagnostics,contact=registration.get('contactPhase',False))
    # Preserve the round-shaft path for a load whose exposed shaft and body
    # form one image component. It still requires a measured supporting plane.
    if feature is None:
        _,plane=el.surface_arcs(sample,target,opening,{})
        if plane:
            feature=el.short_lower(controller,sample,plane,opening,{})
            if feature is None:feature=servo.lower_end(controller,sample,plane,opening)
            if feature:
                caps=caps if registration.get('contactPhase') and caps else el.heads(controller,sample,plane,opening,feature)
    correspondence={}
    if registration.get('layout'):
        entry=el.associate([],caps,registration,opening,correspondence)
    else:
        # Missing historical layout: nearest raised head in XY is explicitly
        # a prior association, not a recovered semantic hole identity.
        ranked=sorted((float(np.linalg.norm(np.asarray(c['point'])[:2]-np.asarray(target)[:2])),i,c) for i,c in enumerate(caps))
        entry=None
        if ranked and ranked[0][0]<opening:
            if len(ranked)==1 or ranked[1][0]-ranked[0][0]>ranked[0][2]['uncertaintyM']:
                entry=dict(ranked[0][2],association='nearest_head_prior',targetFromLayout=False)
    diagnostics['correspondence']=correspondence
    if entry is None:return dict(bothVisible=False,reason='head_unseen_or_ambiguous',diagnostics=diagnostics)
    if registration.get('contactPhase'):
        contact=bit_head_seam(controller,sample,entry,opening)
        if contact:
            return dict(bothVisible=False,reason=None,entry=entry,visualContact=contact,
                stamp=sample[0]['header']['stamp'],diagnostics=diagnostics)
    if feature is None:return dict(bothVisible=False,entry=entry,
        reason='distal_tip_unseen_or_ambiguous',diagnostics=diagnostics)
    delta=np.asarray(entry['point'])-feature['tip']
    return dict(bothVisible=True,reason=None,entry=entry,lowerEnd=feature,deltaM=delta.tolist(),
        residualM=float(np.linalg.norm(delta[:2])),uncertaintyM=entry['uncertaintyM']+feature['uncertaintyM'],
        targetFromLayout=False,source='same_frame_head_minus_distal_tip',
        diagnostics=diagnostics,stamp=sample[0]['header']['stamp'])


def bit_head_seam(controller, sample, entry, opening):
    """Visible narrow bit continues to a measured head and to the held body.

    This is surface contact evidence, never a reconstructed hidden endpoint.
    An air gap, broad housing, or absent head cannot satisfy the depth bands.
    """
    from steward.ego.entry_layout import circle
    raw,cam=sample
    centre=np.asarray(entry['point']); radius=entry['radiusM']
    region=wo.window(raw,cam,centre,opening)
    p,_,_=wo.cloud(raw,cam,region)
    if len(p)<48:return None
    noise=entry['uncertaintyM']/2
    if noise<=0 or radius<=4*noise:return None
    grip=np.asarray(controller.rt.estimator.gripper_position(
        controller.rt.estimator.held_gripper,raw['header']['joints'],raw['header']['odometry']))
    height=p[:,2]-centre[2]
    local=np.linalg.norm(p[:,:2]-centre[:2],axis=1)<radius-noise
    fits=[]
    for lo,hi in ((noise,3*noise),(3*noise,5*noise)):
        band=p[local & (height>=lo) & (height<=hi)]
        if len(band)<12:return None
        fitted=circle(band[:,:2],noise)
        if fitted is None:return None
        xy,r,error=fitted
        if r>=radius/2 or r<=noise/2:return None
        fits.append((xy,r,error))
    if np.linalg.norm(fits[0][0]-fits[1][0])>noise:return None
    if abs(fits[0][1]-fits[1][1])>noise:return None
    xy=(fits[0][0]+fits[1][0])/2; bit_radius=(fits[0][1]+fits[1][1])/2
    if np.linalg.norm(xy-centre[:2])+bit_radius>radius-noise:return None
    radial=np.linalg.norm(p[:,:2]-xy,axis=1)
    seam=p[(abs(radial-bit_radius)<=noise)&(height>0)&(height<=noise)]
    if len(seam)<12:return None
    # A separate standing pin below a hovering tool is not contact of the held
    # tool: require visible axial continuity into the encoder grip envelope.
    upper=grip[2]-centre[2]
    if upper<=5*noise:return None
    for low in np.arange(5*noise,upper,2*noise):
        band=p[(height>=low)&(height<low+2*noise)&(radial<radius+noise)]
        if len(band)<6:
            # The housing may hide a short axial interval. Accept only when the
            # current depth ray actually hits that foreground near the grip.
            from steward.ego.support_holes import project, unproject
            intr=dict(width=raw['depth'].shape[1],height=raw['depth'].shape[0],fovy=raw['fovy'])
            uv,z=project([[*xy,centre[2]+low+noise]],cam,intr)
            x,y=np.rint(uv[0]).astype(int)
            if not (0<=x<intr['width'] and 0<=y<intr['height']):return None
            depth=float(raw['depth'][y,x])
            if not 0<depth<z[0]-noise:return None
            foreground=unproject([[x,y]],[depth],cam,intr)[0]
            if (foreground[2]<=centre[2]+low or np.linalg.norm(foreground-grip)>opening
                    or np.linalg.norm(foreground[:2]-xy)>radius+noise):return None
    return dict(source='same_frame_bit_head_seam',observed=True,tipVisible=False,
        uncertaintyM=2*noise,seamPoints=len(seam),shaftRadiusM=float(bit_radius),
        headRadiusM=float(radius),axisCentreM=xy.tolist())
