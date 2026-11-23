import numpy as np
from steward.assets.generators import steward_bot
from steward.ego.estimator import EgoEstimator
from steward.ego.kinematics import _link_pose, _matrix
from steward.ego.support_holes import project
from steward.math3d import pose_compose


def test_switch_removes_sensor_and_preserves_head_rig(monkeypatch):
    monkeypatch.setenv('STEWARD_WRIST_CAMERA', '1')
    enabled = steward_bot().to_dict()
    monkeypatch.setenv('STEWARD_WRIST_CAMERA', '0')
    disabled = steward_bot().to_dict()
    grip = next(l for l in enabled['body']['links'] if l['id']=='gripper')
    camera = grip['sensors'].pop()
    assert camera['id']=='wrist_camera' and camera['onDemand']
    grip.pop('sensors')
    assert enabled == disabled


def test_work_envelope_visible_and_camera_follows_encoders(monkeypatch):
    monkeypatch.setenv('STEWARD_WRIST_CAMERA', '1')
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    link, sensor = est.cameras['wrist_camera']
    # Robot geometry only: centre, finger tips and below-grasp operating envelope.
    points = np.array([[0,0,0], [0,0,-.025], [0,0,-.05], [0,0,-.15], [.03,.045,-.025]])
    uv, depth = project(points, sensor['mount'], sensor['intrinsics'])
    assert ((uv > 0) & (uv < [848,480])).all()
    assert ((depth>.07) & (depth<.5)).all()
    first=est.camera_pose(camera='wrist_camera')
    est.joints.update(lift=.9, reach=.4, wrist_roll=.4, wrist_yaw=.7)
    second=est.camera_pose(camera='wrist_camera')
    assert not np.allclose(first['p'],second['p'])
    expected=pose_compose(est.link_pose(link),sensor['mount'])
    assert np.allclose(second['p'],expected['p'])


def test_periodic_perception_excludes_wrist(monkeypatch):
    from types import SimpleNamespace as NS
    from steward.ego.runtime import EgoRuntime
    monkeypatch.setenv('STEWARD_WRIST_CAMERA','1')
    est=EgoEstimator(robot_model=steward_bot().to_dict())
    rt=NS(estimator=est,config=NS(cameras=None),_missing_cameras=frozenset())
    frame=dict(cameras=[dict(id=c) for c in est.cameras])
    assert EgoRuntime._cameras(rt,frame)==['head_camera','wide_camera']


def test_calibration_tracks_wrist_but_gripper_to_wrist_remains_rigid(monkeypatch):
    from steward.ego.kinematics import camera_rig, gripper_pose, CAMERA_AXES
    monkeypatch.setenv('STEWARD_WRIST_CAMERA','1')
    model=steward_bot().to_dict()
    first=dict(lift=.6,reach=.3,wrist_yaw=0.,wrist_roll=0.)
    second=dict(lift=.8,reach=.5,wrist_yaw=.7,wrist_roll=.4)
    rigs=[camera_rig(model,j) for j in (first,second)]
    assert not np.allclose(rigs[0]['wrist_camera']['camera_to_reference'],rigs[1]['wrist_camera']['camera_to_reference'])
    assert rigs[0]['wide_camera']==rigs[1]['wide_camera']
    poses=[gripper_pose(model,dict(stamp=1.,reference_camera='head_camera',calibration_source='rig_calibration',
        camera_axes=CAMERA_AXES,joint_encoders=j,views=rig)) for j,rig in zip((first,second),rigs)]
    assert np.allclose(poses[0]['gripper_to_camera']['wrist_camera'],poses[1]['gripper_to_camera']['wrist_camera'])


def test_real_typescript_calibration_preserves_moving_view_and_timestamp(monkeypatch):
    import json,subprocess
    from pathlib import Path
    from steward.ego.kinematics import camera_rig, CAMERA_AXES
    monkeypatch.setenv('STEWARD_WRIST_CAMERA','1')
    model=steward_bot().to_dict();j=dict(lift=.6,reach=.3,wrist_yaw=.7,wrist_roll=.4)
    packet=dict(stamp=1.,reference_camera='head_camera',calibration_source='rig_calibration',
        camera_axes=CAMERA_AXES,joint_encoders=j,views=camera_rig(model,j))
    for v in packet['views'].values():v.update(width=848,height=480,K=[[433,0,424],[0,433,240],[0,0,1]])
    packet['views']['wrist_camera'].update(renderNearM=.02, depthRangeM=[.02,.5], housingSizeM=[.016,.016,.012],tipPlaneClearanceM=.009)
    root=Path(__file__).resolve().parents[1]/'dsh/packages'
    script="""
import { registerHooks, stripTypeScriptTypes } from 'node:module';
import { readFileSync } from 'node:fs';
registerHooks({resolve(s,c,n){if(s==='@steward/dsh-ego')return {url:'data:text/javascript,export class EgoError extends Error {}',shortCircuit:true};return n(s,c)},
load(u,c,n){if(u.endsWith('.ts'))return {format:'module',shortCircuit:true,source:stripTypeScriptTypes(readFileSync(new URL(u),'utf8'),{mode:'transform'})};return n(u,c)}});
const {parseCameraCalibration}=await import(PARSER);
const {egoTools}=await import(TOOLS);
const p=parseCameraCalibration(JSON.parse(readFileSync(0,'utf8')));
const t=egoTools({cameraCalibration:async()=>p}).find(t=>t.name==='ego_camera_calibration');
const v=await t.execute({}, {signal:new AbortController().signal});
process.stdout.write(t.output.render({},v)[0].text);
""".replace('PARSER',json.dumps((root/'dsh-ego-runtime-client/src/sensor-packet.ts').as_uri())).replace('TOOLS',json.dumps((root/'dsh-tool-ego/src/index.ts').as_uri()))
    # The production tool's framework registration is identity-only in this offline test.
    script=script.replace("if(s==='@steward/dsh-ego')", "if(s==='@deepseek-ai/dsh-tools')return {url:'data:text/javascript,export const defineTool=x=>x',shortCircuit:true};if(s==='@steward/dsh-ego')").replace('export class EgoError extends Error {}','export class EgoError extends Error {}; export const recordingId=x=>x;')
    result=json.loads(subprocess.run(['node','--input-type=module','-e',script],input=json.dumps(packet),text=True,capture_output=True,check=True).stdout)
    assert result['stamp']==1. and result['joint_encoders']==j
    assert 'wrist_camera' in result['views']
    assert result['views']['wrist_camera']['renderNearM']==.02
    assert result['views']['wrist_camera']['depthRangeM']==[.02,.5]
    assert result['views']['wrist_camera']['tipPlaneClearanceM']==.009
    assert result['views']['wrist_camera']['housingSizeM']==[.016,.016,.012]
    assert len(json.dumps(result).encode())<20000
