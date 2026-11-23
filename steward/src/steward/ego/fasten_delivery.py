"""One cached-frame tip correction, then a low encoder-checked release."""
from steward.ego import capabilities
from steward.ego.phase_timing import timed
from copy import deepcopy
import math
import time
import numpy as np
from steward.ego.fasten_geometry import directed_tool, transform_feature
from steward.ego.fasten_plan import value, tool_estimate

ENABLED = capabilities.get('STEWARD_ASSEMBLY_FASTEN_DELIVERY') != '0'
RELEASE_HEIGHT_M = .002


def planned_height(held, stamp, entry_z, fallback):
    if not ENABLED:
        return fallback
    feature = tool_estimate(held, stamp)
    if not feature or 'grip' not in feature:
        return fallback
    return float(entry_z+RELEASE_HEIGHT_M-(feature['tip'][2]-feature['grip'][2]))


@timed("hover_observation")
def measure_hover(controller, current, report):
    """Only already available primary RGB-D, tied to this grasp and stationary FK."""
    if not ENABLED or report.get('_deliveryHoverSampled'):
        return
    report['_deliveryHoverSampled'] = True
    report['_deliveryHoverStarted'] = time.monotonic()
    row = report.setdefault('seatDelivery', {})
    row.update(measuredCorrectionM=None, correctionApplied=False, measurement='unknown')
    ctx = controller._registered_alignment; reg = ctx.get('registration')
    if not reg:
        row['measurementReason'] = 'no_registered_target'
        return
    est = controller.rt.estimator; scene = controller.rt.ego_state()
    ent = scene.find(current['object']); view = value(ent, 'contactView')
    if (not view or view.get('source') != 'rgbd_segment' or view.get('camera') != est.primary_camera
            or not view.get('nativeResolution') or view.get('truncated', True)
            or view.get('invalidDepthFraction', 1) > .05
            or not est.held_since < view.get('stamp', -1) <= scene.stamp
            or scene.stamp-view['stamp'] > 2.):
        row['measurementReason'] = 'no_fresh_native_held_segment'
        return
    state = controller.ex._state(); joints=view.get('sensorJoints'); odom=view.get('sensorOdometry')
    if (joints is None or odom is None or
            any(abs(state['joints'].get(k,v)-v) >= .001 for k,v in joints.items()) or
            any(abs(state['odometry'][k]-odom[k]) >= .001 for k in ('x','y','yaw'))):
        row['measurementReason'] = 'frame_not_synchronized_with_hover'
        return
    feature = directed_tool(view['points'], ent.category.value, view['stamp'])
    if not feature:
        row['measurementReason'] = 'tip_or_polarity_occluded'
        return
    from steward.ego.fasten_target import spacing
    separation = spacing(reg.get('layout', reg.get('holes', [])))
    if separation is None:
        row['measurementReason'] = 'hole_spacing_unknown'
        return
    target = np.asarray(reg.get('executionPoint', reg['point']))
    delta = target[:2]-np.asarray(feature['tip'])[:2]
    length = float(np.linalg.norm(delta)); bound=separation/2
    correction = delta*min(1., bound/max(length, 1e-12))
    from steward.ego.assembly_precision import calibrate
    if calibrate(controller, current['object'], feature=feature) is None:
        row['measurementReason'] = 'grasp_changed'
        return
    ctx.update(tip=deepcopy(est.tracks[current['object']].held_tool_feature),
               tipHeldSince=est.held_since, tipOffsetSource='measured')
    intr = est.camera_intrinsics(view['camera'])
    cam = est.camera_pose(odom, view['camera'])
    distance = float(np.linalg.norm(np.asarray(feature['tip'])-cam['p']))
    metres_per_pixel = 2*distance*math.tan(math.radians(intr['fovy'])/2)/view['nativeResolution'][1]
    # An endpoint fit is bounded at least by two native pixels, not hole-centre precision.
    ctx['tip']['uncertaintyM'] = max(float(feature.get('uncertaintyM', 0.)), 2*metres_per_pixel)
    ctx['tipOffsetMeasuredCount'] += 1
    correction_point = target.copy(); correction_point[:2]=np.asarray(feature['tip'])[:2]+correction
    ctx['correctionPoint'] = correction_point.tolist()
    report.update(tipOffsetMeasured=True, tipOffsetSource='measured',
                  tipOffsetMeasuredCount=ctx['tipOffsetMeasuredCount'], preReleaseResidualM=length,
                  alignmentBasis='registered_hole_fk')
    row.update(measurement='measured_tip', measurementReason=None,
        measuredCorrectionM=np.round(correction,6).tolist(), correctionBoundM=bound,
        measuredResidualM=length, uncertaintyM=reg['uncertaintyM']+ctx['tip']['uncertaintyM'])
    report.setdefault('observations', []).append(dict(status='estimated', residualM=length,
        source='native_held_tip_to_registered_target', observedAt=view['stamp'],
        uncertaintyM=row['uncertaintyM']))


@timed("descent")
def low_release(controller, current, report):
    """Remove bbox inflation/uncertainty from the final tip height, using actual FK."""
    if current.get('_observedReleaseEndpoint'):
        report.setdefault('seatDelivery',{})['releaseHeightSource']='same_frame_visible_gap'
        return
    if not ENABLED or not current.get('seatTarget'):
        return
    row = report.setdefault('seatDelivery', {})
    est=controller.rt.estimator; track=est.tracks.get(current['object'])
    feature=getattr(track,'held_tool_feature',None) or getattr(track,'held_grasp_geometry',None)
    if not feature or est.held_id != current['object']:
        row.update(releaseHeightM=None, releaseHeightSource='unknown')
        return
    def endpoint():
        state=controller.ex._state()
        return state, transform_feature(feature,
            est.gripper_rotation(est.held_gripper,state['joints'],state['odometry']),
            est.gripper_position(est.held_gripper,state['joints'],state['odometry']))
    state,f=endpoint(); entry=current['seatTarget']['point'][2]
    from steward.ego.release_height import geometry
    bound = geometry(controller, current)
    floor = bound['floorM'] if bound else entry
    height=float(f['tip'][2]-entry)
    release_height = RELEASE_HEIGHT_M
    release_height += max(0., floor-entry)
    desired=float(state['joints']['lift']+release_height-height)
    row.update(releaseHeightM=height, releaseHeightSource='measured_offset_fk' if
               feature.get('source')=='rgbd_endpoint_width' else 'inferred_offset_fk')
    if abs(height-release_height) > min(.001, release_height/2):
        from steward.ego.skills import SkillFailed
        start=dict(state['joints']); end=dict(start,lift=desired)
        path=[dict(start,lift=float(h)) for h in np.linspace(start['lift'],desired,
              max(2,int(abs(desired-start['lift'])/.002)+1))]
        if not controller.rt.esim.arm.reachable_path(path):
            row['lowReleaseWarning']='lowering_path_unavailable'
            return
        try:
            from steward.ego.fasten_seat import lift_to
            lift_to(controller.ex,desired,ramp=False)
        except SkillFailed as exc:
            controller.check()
            row['lowReleaseWarning']=str(exc)[:120]
        _,f=endpoint(); row['releaseHeightM']=float(f['tip'][2]-entry)
    if report.get('_deliveryHoverStarted') is not None:
        row['hoverToReleaseWallS'] = round(time.monotonic()-report['_deliveryHoverStarted'],3)
    current['releaseGapM']=row['releaseHeightM']
    row['releaseHeightM']=round(row['releaseHeightM'],6)
