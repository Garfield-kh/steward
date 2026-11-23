"""One support edge, balanced contact reserves, and in-place assembly (ASSEMBLY_EDGE_RETENTION)."""
from copy import deepcopy
from itertools import product
import math
import numpy as np
from steward.ego import capabilities
from steward.math3d import matrix_from_quat, quat_from_yaw


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_EDGE_RETENTION') != '0' and capabilities.get('STEWARD_ASSEMBLY_EDGE_GEOMETRY') != '0'


def clip_polygon(points, axis, limit, sign=1.):
    points = np.asarray(points, float); out = []
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        da, db = limit-sign*a[axis], limit-sign*b[axis]
        if da >= 0: out.append(a)
        if (da >= 0) != (db >= 0): out.append(a+(b-a)*da/(da-db))
    return np.asarray(out).reshape(-1, points.shape[1])


def contact(record, table, dim, sign, angle, offset, *, inward=False):
    from steward.ego.assembly_edge import entity_box, support_polygon
    from steward.ego.edge_result import rotated
    from steward.ego.edge_geometry import signed_margin
    shape = rotated(record['edgeGeometry'], angle)
    box, _, top = entity_box(table)
    tr = matrix_from_quat(quat_from_yaw(box.yaw))[:2, :2]
    half = np.array([box.hx, box.hy]); offset = np.asarray(offset)
    protrusion = np.asarray(shape['protrusions'])[:, :2] @ tr
    env = np.asarray(shape['envelope'])[:, :2] @ tr
    com = np.asarray(shape['centreOfMass'])[:2] @ tr
    # A body's bearing is the actual intersection with the support, not the
    # artificial strip ending at the estimated protrusion's inner boundary.
    bearing = np.asarray(shape['core'] if shape['bearingRole']=='body_face' else shape['bearingPolygon'])[:, :2] @ tr
    polygon = support_polygon(np.c_[bearing+offset, np.zeros(len(bearing))])
    for axis, side in product(range(2), (-1, 1)):
        if len(polygon): polygon = clip_polygon(polygon, axis, half[axis], side)
    bearing_margin = signed_margin(polygon, offset+com) if len(polygon)>=3 else -math.inf
    table_margin = float(np.min(half-abs(offset+com)))
    clearance = float(np.min(sign*(protrusion[:,dim]+offset[dim]))-half[dim])
    inside = float(np.min(half-abs(bearing+offset)))
    if (clearance <= 0 and not inward) or min(table_margin, bearing_margin) <= 0:
        return None
    if np.max(abs(env[:,1-dim]+offset[1-dim])) > half[1-dim]+1e-8:
        return None
    line = half[dim]-sign*offset[dim]
    inner = float(np.min(sign*protrusion[:,dim])); cp = float(sign*com[dim])
    spot = np.array([box.cx,box.cy])+tr @ offset
    normal = sign*tr[:,dim]
    from steward.ego import placement_body_frame
    angle_deg = 90.
    if placement_body_frame.enabled():
        axis = matrix_from_quat(quat_from_yaw(angle)) @ placement_body_frame.long_axis(record)
        angle_deg = round(math.degrees(math.atan2(abs(float(axis[:2] @ normal)),
            abs(float(axis[0]*normal[1]-axis[1]*normal[0])))), 12)
    return dict(offset=offset.tolist(),spot=spot.tolist(),edgeAxis=dim,edgeSign=sign,
        yaw=record['reference']['yaw']+angle,yawDeltaRad=angle,
        overhangM=float(np.max(sign*(env[:,dim]+offset[dim]))-half[dim]),
        protrusionClearanceM=clearance,comMarginM=min(table_margin,bearing_margin),
        balancedReserveM=min(clearance,table_margin,bearing_margin),
        tableComMarginM=table_margin,bearingComMarginM=bearing_margin,
        bearingInsideMarginM=float(np.min(half-abs(polygon))), fullBearingInsideMarginM=inside,
        supportWindowM=inner-cp,edgeLineBodyM=line,protrusionInnerM=inner,
        bearingOuterM=float(np.max(sign*bearing[:,dim])),longAxisEdgeAngleDeg=angle_deg if placement_body_frame.enabled() else 90.,
        edgeNormalWorld=normal.tolist(),protrusionSpanM=float(np.ptp(protrusion[:,dim])),
        requiredComMarginM=0.,comUncertaintyM=shape.get('comUncertaintyM',0.),
        bodyCentreZM=top-shape['bearingZM'],bearingZM=shape['bearingZM'],
        bodyGraspZM=top+(shape['upperBodyZM']-shape['bearingZM'])/2,
        comOffsetWorldM=shape['centreOfMass'][:2],massSource=shape['massSource'],
        bearingRole=shape['bearingRole'],contactWindow=True,geometryConflict=False,
        geometryConfirmed=shape.get('geometryConfirmed',False),allMarginsNegative=False,
        unconfirmed=not shape.get('geometryConfirmed',False),
        bearingPolygonWorld=np.c_[polygon @ tr.T+[box.cx,box.cy],np.full(len(polygon),top)].tolist(),
        estimatedTiltDeg=0.,observed=False)


def candidates(record, table, sigma=0., diagnostics=None):
    from steward.ego.assembly_edge import entity_box
    from steward.ego.edge_result import rotated
    box, _, _ = entity_box(table); tr=matrix_from_quat(quat_from_yaw(box.yaw))[:2,:2]
    half=np.array([box.hx,box.hy]); shape=record['edgeGeometry']; result=[]
    from steward.ego import placement_support_margin, placement_body_frame
    target=record.get('edgeCorrectionTarget')
    if target and placement_body_frame.enabled():
        result = placement_body_frame.correction_rows(record, table, sigma)
    elif target:
        if table.id == target['support']:
            offset=(np.asarray(target['spot'])-[box.cx,box.cy]) @ tr
            from steward.ego import directed_orientation
            angle=target.get('yawDeltaRad',0.) if directed_orientation.enabled() else 0.
            c=contact(record,table,target['edgeAxis'],target['edgeSign'],angle,offset,inward=placement_support_margin.enabled())
            if c:
                if placement_support_margin.enabled(): c['inwardOnly']=True
                result.append(c)
            # If the planned outward move is mechanically rejected after grasp,
            # the only recovery candidate is the original release position.
            original=(np.asarray(record['reference']['position'][:2])-[box.cx,box.cy]) @ tr
            restore=contact(record,table,target['edgeAxis'],target['edgeSign'],0.,original,inward=placement_support_margin.enabled())
            if restore:
                restore['correctionRecovery']=True
                if placement_support_margin.enabled(): restore['inwardOnly']=True
                result.append(restore)
    else:
        axis=(placement_body_frame.outward_axis(record) if placement_body_frame.enabled() else np.asarray(shape['outwardAxis']))[:2]; heading=math.atan2(axis[1],axis[0])
        for dim,sign in product(range(2),(-1,1)):
            angle=(box.yaw+dim*math.pi/2+(math.pi if sign<0 else 0.)-heading+math.pi)%(2*math.pi)-math.pi
            s=rotated(shape,angle); com=np.asarray(s['centreOfMass'])[:2] @ tr
            protrusion=np.asarray(s['protrusions'])[:,:2] @ tr
            bearing=np.asarray(s['bearingPolygon'])[:,:2] @ tr
            env=np.asarray(s['envelope'])[:,:2] @ tr
            inner=float(np.min(sign*protrusion[:,dim])); cp=float(sign*com[dim])
            line=(inner+cp)/2
            uncertainty=max(0.,float(sigma))+float(shape.get('comUncertaintyM',0.))
            inward=placement_support_margin.enabled() and inner>cp
            if inward:
                line=placement_support_margin.inward_line(inner,cp,uncertainty)
            tangent=1-dim; lo=-half[tangent]-env[:,tangent].min(); hi=half[tangent]-env[:,tangent].max()
            if lo>hi:continue
            root=(np.asarray(record['reference']['position'])[:2]-[box.cx,box.cy]) @ tr
            alongs=[float(np.clip(-com[tangent],lo,hi)),float(np.clip(root[tangent],lo,hi))]
            if placement_body_frame.enabled() and record.get('edgeNeighbourSearch'):
                stride=float(np.ptp(env[:,tangent]))
                alongs.extend([float(np.clip(alongs[-1]-stride,lo,hi)),
                               float(np.clip(alongs[-1]+stride,lo,hi)),float(lo),float(hi)])
            for along in dict.fromkeys(alongs):
                offset=np.zeros(2);offset[dim]=sign*(half[dim]-line);offset[tangent]=along
                c=contact(record,table,dim,sign,angle,offset,inward=inward)
                if c:
                    c['geometryUncertaintyM']=float(sigma+shape.get('comUncertaintyM',0.))
                    if inward:
                        c.update(inwardOnly=True, centreMarginM=line,
                            inwardBiasM=line-(inner+cp)/2,
                            protrusionUnconfirmed=c['protrusionClearanceM']<=0)
                    c['reserveCoversUncertainty']=c['balancedReserveM']>c['geometryUncertaintyM']
                    c['unconfirmed'] |= not c['reserveCoversUncertainty']
                    result.append(c)
    if not target:
        result.sort(key=lambda c:(-round(c['balancedReserveM']/max(record['reference']['size']),12),
            np.linalg.norm(np.asarray(c['spot'])-record['reference']['position'][:2])))
    from steward.ego import directed_orientation
    if directed_orientation.enabled():
        for c in result:
            c['protrusionEndSource']=deepcopy(shape.get('protrusionEndSource', dict(source='registered_end_best_estimate',confidence='unconfirmed')))
            c['plannedYawRad']=c['yaw']
    if diagnostics is not None:diagnostics.update(candidateCount=len(result),contactWindow=True,rejected={},allMarginsNegative=False)
    return result


def correction(record, report):
    if report.get('level') is not False or not report.get('observed') or report.get('estimatedTiltDeg') is None or not report.get('used'):
        return False
    anchor=record.get('edgeAnchor')
    if not anchor:return False
    reserve=min(report.get('comMarginM',0.),anchor.get('comMarginM',0.))
    from steward.ego import placement_support_margin
    if placement_support_margin.enabled() and anchor.get('inwardOnly'):
        reserve=min(reserve,anchor.get('centreMarginM',reserve))-anchor.get('geometryUncertaintyM',0.)
    from steward.ego import placement_body_frame
    span=anchor.get('protrusionSpanM',0.)
    if placement_body_frame.enabled() and span <= 0: span=reserve
    step=min(span/2,reserve/2)
    if step<=0:return False
    normal=np.asarray(anchor['edgeNormalWorld']); start=np.asarray(anchor['spot'] if placement_body_frame.enabled() else record['reference']['position'][:2])
    record['edgeCorrectionTarget']=dict(support=record['reference']['support'],
        spot=(start+normal*step).tolist(),edgeAxis=anchor['edgeAxis'],edgeSign=anchor['edgeSign'])
    if placement_body_frame.enabled(): record['edgeCorrectionTarget']['yaw']=anchor['yaw']
    record['edgeCorrectionDirection']=normal.tolist()
    record['edgeCorrectionStepM']=step
    return True


def retain(record, report):
    if report.get('used') or report.get('observed') and report.get('level') is True:
        record['edgeRetained']=dict(reference=deepcopy(record['reference']),
            edgePlacement=deepcopy(report),reason='assemble_remaining_interfaces_in_place')
    record.pop('edgeCorrectionTarget',None)


def retained_record(records, body):
    candidates=[r for key,r in records.items() if r.get('edgeRetained') and
        (key==body or body in r.get('aliases', []))]
    return candidates[0] if len(candidates)==1 else None


def retained_result(job, record):
    saved=record['edgeRetained']; edge=deepcopy(saved['edgePlacement'])
    from steward.ego import placement_support_margin
    if placement_support_margin.enabled():
        job.postcondition.update(placement_support_margin.guidance(record))
        job.message=job.postcondition['guidance']
    job.postcondition.update(executed=False,releaseExecuted=False,bodyRetained=True,
        body=record['reference']['entity'],reference=deepcopy(record['reference']),
        edgePlacement=edge,flipped=False,canContinue=True,actionStatus='executed',
        confirmation='unverified' if edge.get('unconfirmed') else 'confirmed',
        reason='edge_body_retained_assemble_in_place')


def recovery_record(controller, job, record, table, candidate):
    """A completed recovery release invalidates any earlier measured tilt."""
    from steward.ego.assembly_motion import released_record, restore_track
    from steward.ego.assembly_edge import rebase_height
    stamp=controller.ex._state()['stamp']
    delta=matrix_from_quat(quat_from_yaw(candidate['yawDeltaRad']))
    new=released_record(record,candidate['spot'],table,delta,stamp)
    rebase_height(new,record,delta,candidate)
    new['edgeAnchor']=deepcopy(candidate);new.pop('edgeCorrectionTarget',None)
    with controller.rt._lock:
        controller.rt.estimator.stationary_assemblies.records[record['reference']['entity']]=new
        restore_track(controller.rt.estimator,new)
    job.postcondition.update(reference=deepcopy(new['reference']),
        edgePlacement=dict(candidate,used=True,estimatedTiltDeg=None,level=None,
            observed=False,unconfirmed=True,reason='recovered_on_same_edge',levelReason='recovery_release_not_measured',
            extraObservations=0))
