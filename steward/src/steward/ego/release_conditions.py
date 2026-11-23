"""Passive release receipts and posterior repair, behind the existing release_conditions switch.

The admitted endpoint owns motion. In particular, an encoder position is not a
loaded actuator setpoint: re-commanding it removes the existing compensation.
No extra camera acquisition, correction, dwell or withdrawal runs here.
"""
from copy import deepcopy
import math
import uuid
import numpy as np
from steward.ego import capabilities
from steward.ego.fasten_geometry import transform_feature


def enabled():
    return capabilities.get('STEWARD_ASSEMBLY_RELEASE_CONDITIONS') != '0'


def unit(v):
    v = np.asarray(v, float)
    norm = np.linalg.norm(v)
    if v.shape != (3,) or not np.isfinite(v).all() or norm <= np.finfo(float).eps:
        raise ValueError('unresolved_axis')
    return v / norm


def classify(g):
    flags = []
    gap = g.get('gapM')
    noise = g.get('uncertaintyM', 0.)
    if gap is not None:
        if gap > noise: flags.append('suspended')
        if gap < -noise: flags.append('penetrating')
    if g.get('tilted'): flags.append('tilted')
    return flags


def estimate(controller, current):
    """Read the already admitted target and grasp; never replace either one.

    A surrounding depth patch is not necessarily the entrance-bearing surface.
    Cached geometry stays labelled as an estimate, even with a fresh encoder.
    """
    est = controller.rt.estimator
    state = controller.ex._state()
    target = current.get('seatTarget') or {}
    track = est.tracks.get(current.get('object'))
    local = (getattr(track, 'held_tool_feature', None) or
             getattr(track, 'held_grasp_geometry', None))
    if not local or not target.get('point') or est.held_id != current.get('object'):
        return dict(sameFrame=False, axisMeasured=False, reason='endpoint_geometry_missing')
    feature = transform_feature(local,
        est.gripper_rotation(est.held_gripper, state['joints'], state['odometry']),
        est.gripper_position(est.held_gripper, state['joints'], state['odometry']))
    normal = unit(target.get('normal', [0., 0., 1.]))
    if normal[2] < 0: normal = -normal
    axis = unit(feature['axis'])
    gap = float((np.asarray(feature['tip']) - target['point']) @ normal)
    noise = max(float(target.get('planeNoiseM', 0.)), float(feature.get('uncertaintyM', 0.)))
    length = float(np.linalg.norm(np.asarray(feature['head']) - feature['tip']))
    tilt = math.acos(float(np.clip(axis @ normal, -1., 1.)))
    return dict(gapM=gap, tip=feature['tip'], planePoint=deepcopy(target['point']),
        axis=axis.tolist(), normal=normal.tolist(), tiltDeg=math.degrees(tilt),
        tilted=tilt > math.atan2(noise, max(length, noise)), uncertaintyM=noise,
        sameFrame=False, axisMeasured=False, observedAt=state['stamp'],
        planeObservedAt=target.get('observedAt', target.get('stamp')),
        planeSource=target.get('source', target.get('targetSource', 'admitted_entry_estimate')),
        endpointSource=local.get('source', 'grasp_geometry') + '_plus_fk',
        contactMeasured=None)


def prepare(controller, current, report):
    """Record the retained endpoint without issuing commands or consuming ticks."""
    if not enabled(): return
    row = dict(schema='release_conditions.v1', id=uuid.uuid4().hex, executed=True,
        policy='retained_endpoint_no_added_motion', viewChanges=0, corrections=[],
        unconfirmed=True, allConfirmed=False, addedMotion=False,
        settle=dict(stationary=None, stationaryS=None, addedDurationS=0.,
                    source='ordinary_release_settle_not_remeasured'))
    try:
        geometry = estimate(controller, current)
        row.update(before=geometry, after=deepcopy(geometry))
        state = controller.ex._state()
        row['setpointResiduals'] = {k:float(v-state['joints'][k])
            for k,v in getattr(controller.ex, '_setpoints', {}).items()
            if k in state['joints']}
    except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        row['reason'] = 'best_estimate_release:' + type(exc).__name__
    height = current.get('releaseHeight', {})
    if height.get('policy') == 'observed_plane_height_only':
        from steward.ego.release_plane import summary as height_summary
        row['height'] = height_summary(height)
        for stage in ('before', 'after'):
            if height.get(stage):
                row.setdefault(stage, {}).update({k:deepcopy(v) for k,v in height[stage].items()
                    if k in ('gapM', 'tip', 'planePoint', 'normal', 'planeSource',
                             'endpointSource', 'observedAt', 'planeObservedAt')})
        row.update(policy='observed_plane_height_only', addedMotion=bool(height.get('motions')))
        row['corrections'] = deepcopy(height.get('motions', []))
    row['initialIssues'] = classify(row.get('before', {}))
    row['suspectedCauses'] = list(row['initialIssues'])
    row.update(object=current.get('object'), body=current.get('seatBody'), hole=current.get('seatHole'))
    current['releaseConditions'] = row
    report['releaseConditions'] = deepcopy(row)


def opened(current):
    """Called after the ordinary open, including paths with cameras disabled."""
    if not enabled() or not current.get('releaseConditions'): return
    current['releaseConditions'].update(releasedAt=current.get('releasedAt'), releaseExecuted=True)


def held_open(ex, current):
    """Timestamp the existing wait; do not add another wait or reset setpoints."""
    if not enabled() or not current.get('releaseConditions'): return
    row = current['releaseConditions']
    row['postOpenHold'] = dict(durationS=max(0., ex._state()['stamp']-current['releasedAt']),
        extraDurationS=0., source='ordinary_open_wait_sensor_clock')


def after_open(controller, current, report):
    if not enabled() or not current.get('releaseConditions'): return
    report['releaseConditions'] = deepcopy(current['releaseConditions'])


def withdrawing(ex, current):
    if not enabled() or not current.get('releaseConditions'): return
    row = current['releaseConditions']
    delta = current['seatEndpoint']['safeLift'] - ex._state()['joints']['lift']
    row['withdrawal'] = dict(direction=[0., 0., float(np.sign(delta))],
        startedAt=ex._state()['stamp'], collision=None, addedMotion=False,
        source='ordinary_lift_then_retract_command_not_contact_measurement')


def withdrawn(ex, current):
    if not enabled() or not current.get('releaseConditions'): return
    row = current['releaseConditions'].get('withdrawal')
    if row is not None:
        row['durationS'] = max(0., ex._state()['stamp'] - row['startedAt'])


def landing(feature,normal,target,noise):
    """Shape-relative outcome, separate from the hole's occupancy verdict."""
    axis=unit(np.asarray(feature['head'])-feature['tip']);normal=unit(normal)
    length=float(np.linalg.norm(np.asarray(feature['head'])-feature['tip']))
    along=float(axis@normal);lateral=float(np.linalg.norm(np.cross(axis,normal))*length)
    uncertainty=max(noise,float(feature.get('uncertaintyM',0.)))
    if abs(along)*length<=2*uncertainty:state='fallen'
    elif lateral>2*uncertainty:state='tilted'
    elif along<0:state='inverted_upright'
    else:
        delta=np.asarray(feature['tip'])-target['point']
        distance=float(np.linalg.norm(delta-(delta@normal)*normal))
        state='upright_beside' if distance>target.get('radiusM',0.)+uncertainty else 'upright_at_entry'
    return dict(outcome=state,axisTiltDeg=math.degrees(math.acos(float(np.clip(along,-1.,1.)))),
        headM=feature['head'],tipM=feature['tip'],uncertaintyM=uncertainty,
        observedAt=feature.get('stamp'),source='independent_same_identity_rgbd_endpoints')


def outcome(controller,current,report):
    if not enabled() or not current.get('releaseConditions'):return
    row=current['releaseConditions'];check=(current.get('_holeSeatReport') or {}).get('check',{})
    state=check.get('landingState')
    row['outcome']='fallen_or_tilted' if state=='tilted_head' else 'upright_at_entry' if check.get('passed') else 'unknown'
    try:
        from steward.ego.fasten_supply import measured_fastener
        scene=controller.rt.ego_state();entity=scene.find(current['object'])
        feature=measured_fastener(entity,scene.stamp) if entity is not None else None
        from steward.ego import released_object
        object_checked = released_object.enabled() and check.get('source') == released_object.SOURCE
        if feature and current['releasedAt']<feature['stamp']<=scene.stamp and not object_checked:
            observed=landing(feature,row.get('after',{}).get('normal',[0.,0.,1.]),
                current['seatTarget'],row.get('after',{}).get('uncertaintyM',0.))
            row['landingGeometry']=observed;row['outcome']=observed['outcome']
            if observed['outcome']!='upright_at_entry':
                check=dict(observed,status='failed',passed=False,landingState=observed['outcome'],
                    reason='released_same_object_'+observed['outcome'],source='rgbd_endpoint_width',
                    freshEndpointObserved=True,attachmentObserved=False)
                if current.get('_holeSeatReport'):current['_holeSeatReport']['check']=deepcopy(check)
    except (AttributeError,KeyError,TypeError,ValueError):
        pass
    row['postReleaseCheck']=deepcopy(check)
    row['failureCauses']=list(row.get('suspectedCauses',[])) if check.get('passed') is False else []
    if row['outcome'] in ('fallen','tilted','fallen_or_tilted'):
        if row.get('allConfirmed'):row['failureCauses'].append('conditions_correct_still_fallen')
        elif not row['failureCauses']:row['failureCauses'].append('unconfirmed_release_conditions')
    row['causeConfirmed']=False
    report['releaseConditions']=deepcopy(row)


def repair_now(attempt):
    """A fresh observed fall enters the existing same-object assembly_outcome replay."""
    from steward.ego.released_object import recovery_evidence
    if recovery_evidence(attempt, attempt.get('seatCheck') or {}): return True
    if not enabled():return False
    row=attempt.get('releaseConditions',{})
    check=attempt.get('seatCheck',{})
    return bool(row.get('outcome') in ('fallen','tilted','fallen_or_tilted','inverted_upright') and
        check.get('passed') is False and (check.get('observedAt') or -1)>attempt.get('seatedAt',math.inf))


def record_release(current,report):
    if not enabled() or not current.get('releaseConditions'):return
    row=current['releaseConditions']
    row['releaseExecuted']=bool(current.get('releaseExecuted'))
    if row['releaseExecuted']:row['releasedAt']=current.get('releasedAt')
    report['releaseConditions']=deepcopy(row)
    from steward.ego.capability_usage import record
    record('release_conditions',event_id=row['id'],stage='released' if row['releaseExecuted'] else 'prepared',
        entity=current.get('object'),stamp=row.get('releasedAt'),action=row['releaseExecuted'],
        measured=row.get('allConfirmed',False),reason=row.get('reason'),evidence=row)


def summary(row):
    """Small idempotent receipt; the full keyed event keeps every sensor sample."""
    if row.get('executed') is False:
        # No new controller ran. Its reason remains in the keyed event and
        # session ledger; the existing unconfirmed release receipt suffices.
        return {}
    result={k:deepcopy(row[k]) for k in ('id','unconfirmed','releasedAt') if k in row}
    if row.get('outcome') not in (None,'unknown'):result['outcome']=row['outcome']
    if row.get('failureCauses'):
        result.update(failureCauses=deepcopy(row['failureCauses']),causeConfirmed=False)
    if 'after' in row:
        result['after']={k:deepcopy(v) for k,v in row['after'].items() if k in
            ('gapM','tiltDeg')}
    if 'settle' in row:
        result['settle']={k:v for k,v in row['settle'].items() if k in ('stationary','stationaryS')}
    if 'withdrawal' in row:
        result['withdrawal']={k:v for k,v in row['withdrawal'].items() if k in
            ('direction','speedMps')}
    def rounded(v):
        if isinstance(v,float):return round(v,6)
        if isinstance(v,dict):return {k:rounded(x) for k,x in v.items()}
        if isinstance(v,list):return [rounded(x) for x in v]
        return v
    return rounded(result)
