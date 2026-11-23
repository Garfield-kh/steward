"""Require a measured upward destination after transporting a support assembly."""
from steward.ego import capabilities
import math
import numpy as np

ENABLED = capabilities.get('STEWARD_ASSEMBLY_POSTFLIP_DESTINATION') != '0'


def assess(rt, plan):
    target=plan.get('placementTarget') or {};ref=target.get('mateReference') or {}
    if (not ENABLED or plan.get('seatTarget') or not target.get('holePairs')
            or ref.get('ownReleaseAt') is None or not
            (ref.get('faceCheck') or 'rotationBefore' in ref or ref.get('source')=='commanded_release')):
        return None
    row=dict(status='unknown',passed=None,body=ref['entity'],source='post_transport_destination',
        reason='observe_post_transport_face',nextAction='observe_body_face_then_predict')
    if plan.get('support')!=ref['entity']:
        return dict(row,status='failed',passed=False,reason='postflip_support_substituted')
    surface=plan.get('placementSurface') or target.get('placementSurface')
    from steward.ego.place_surface import valid, height_at
    if (not surface or not surface.get('observed') or not surface.get('patch')
            or surface.get('entity',surface.get('body'))!=ref['entity']
            or not valid(surface,ref,rt.estimator.stamp)):
        return dict(row,reason='no_fresh_postflip_body_surface')
    normal=np.asarray(surface['patch']['normal'])
    row.update(observedAt=surface['observedAt'],normal=normal.tolist())
    # A fitted plane points upward by construction; semantic interface access
    # must also agree, otherwise the visible face might be the other one.
    from steward.ego.assembly_status import interface_access
    scene=rt.ego_state();body=scene.find(ref['entity'])
    selected={p['bodyHole'] for p in target['holePairs']}
    access=interface_access(rt.esim,body,selected) if body else dict(status='unknown')
    row['face']=access
    if normal[2]<=0 or access['status']!='estimated_accessible':
        return dict(row,reason='target_face_not_confirmed_upward')
    yaw=ref.get('yaw',0.);c,s=math.cos(yaw),math.sin(yaw)
    rotation=np.array([[c,-s],[s,c]])
    source={h['id']:h for h in target.get('releasedFrame',{}).get('holes',[])}
    if any(p['partHole'] not in source for p in target['holePairs']):
        return dict(row,reason='component_hole_frame_unavailable')
    yaw=target['yawRad'];c,s=math.cos(yaw),math.sin(yaw);part_rotation=np.array([[c,-s],[s,c]])
    points=[np.asarray(plan['spot'])[:2]+part_rotation @ np.asarray(source[p['partHole']]['p'])[:2]
            for p in target['holePairs']]
    local=(np.asarray(points)-np.asarray(ref['position'])[:2]) @ rotation
    margin=np.asarray(ref['size'])[:2]/2-abs(local)
    row['minimumBodyMarginM']=float(margin.min())
    if margin.min()<0:
        return dict(row,status='failed',passed=False,reason='target_holes_outside_bare_body')
    table=scene.find(ref.get('support'))
    if table is None:
        return dict(row,reason='outer_support_unobserved')
    from steward.ego.planner import entity_box
    heights=[height_at(surface['patch'],p) for p in points]
    clearance=min(heights)-entity_box(table)[2]
    row['surfaceAboveOuterSupportM']=float(clearance)
    if clearance<=surface.get('uncertaintyM',0.):
        return dict(row,status='failed',passed=False,reason='target_surface_indistinguishable_from_table')
    row.update(status='satisfied',passed=True,reason='fresh_upward_body_face_contains_target_holes',nextAction=None)
    return row


def require(rt, plan, job=None):
    row=assess(rt,plan)
    if row is None: return
    plan['postflipDestination']=row
    if job is not None: job.postcondition['postflipDestination']=row
    if row['passed'] is not True:
        from steward.ego.skills import SkillFailed
        raise SkillFailed('postflip destination needs observation; load remains held: '+row['reason'])
