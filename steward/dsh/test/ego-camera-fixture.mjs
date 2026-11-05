// Each sensor test owns its HTTP listener, workspace and provider instance.
import { createServer } from 'node:http'
import { once } from 'node:events'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import EgoRuntimeClient from '@steward/dsh-ego-runtime-client'
import { egoTools } from '@steward/dsh-tool-ego'

export async function cameraFixture(t, overrides = {}) {
  const root = await mkdtemp(join(tmpdir(), 'steward-ego-camera-'))
  const ids = ['head_camera_left', 'head_camera', 'head_camera_right']
  const bytes = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0xff, 0xd9])
  const packet = { stamp: 42.125, reference_camera: 'head_camera', calibration_source: 'rig_calibration',
    camera_axes: 'opencv_x_right_y_down_z_forward',
    views: Object.fromEntries(ids.map((id, i) => [id, { width: 320, height: 240,
      K: [[200, 0, 159.5], [0, 200, 119.5], [0, 0, 1]],
      camera_to_reference: [[1, 0, 0, (i - 1) * 0.12], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      image: 'data:image/jpeg;base64,' + bytes.toString('base64') }])) }
  const calibration = { reference_camera: packet.reference_camera, camera_axes: packet.camera_axes,
    calibration_source: packet.calibration_source,
    views: Object.fromEntries(ids.map(id => [id, Object.fromEntries(Object.entries(packet.views[id]).filter(([key]) => key !== 'image'))])) }
  packet.gripper_pose = { stamp: packet.stamp, pose_source: 'joint_encoders_and_robot_model',
    gripper_frame: 'gripper', gripper_axes: 'x_forward_y_left_z_up', camera_axes: packet.camera_axes,
    joint_positions: { lift: 0.6, reach: 0.1 },
    gripper_to_base: [[1, 0, 0, 0.36], [0, 1, 0, 0], [0, 0, 1, 0.6], [0, 0, 0, 1]],
    gripper_to_camera: Object.fromEntries(ids.map((id, i) => [id,
      [[1, 0, 0, (1 - i) * 0.12], [0, 1, 0, 0.5], [0, 0, 1, 0.7], [0, 0, 0, 1]]])) }
  const calls = [], state = { packet, status: 200 }
  const ctx = new Context()
  const server = createServer(async (req, res) => {
    const chunks = []; for await (const chunk of req) chunks.push(chunk)
    calls.push({ method: req.method, path: req.url, body: Buffer.concat(chunks).toString() })
    res.setHeader('content-type', 'application/json')
    const captureRequest = req.method === 'POST' && req.url === '/ego/cameras/capture'
    const poseRequest = req.method === 'GET' && req.url === '/ego/gripper/pose'
    const calibrationRequest = req.method === 'GET' && req.url === '/ego/cameras/calibration'
    if (!captureRequest && !poseRequest && !calibrationRequest) {
      res.writeHead(503); res.end(JSON.stringify({ error: 'Unexpected robot request in camera fixture' })); return
    }
    res.writeHead(state.status)
    res.end(JSON.stringify(state.status === 200 ? calibrationRequest ? calibration : poseRequest ? state.packet.gripper_pose : state.packet : { error: 'Camera unavailable' }))
  })
  t.after(async () => {
    try { await ctx.fiber.dispose() } finally {
      server.closeAllConnections()
      await new Promise(resolve => server.close(resolve))
      await rm(root, { recursive: true, force: true })
    }
  })
  server.listen(0, '127.0.0.1'); await once(server, 'listening')
  const config = { endpoint: `http://127.0.0.1:${server.address().port}`, timeoutMs: 5000,
    cameraCaptureDir: join(root, 'workspace/captures'), cameraMaxBytes: 1024, ...overrides }
  await ctx.plugin(EgoRuntimeClient, config)
  const tool = egoTools(ctx.ego).find(tool => tool.name === 'ego_capture_cameras')
  const capture = (signal = AbortSignal.timeout(10000)) => tool.execute({}, { name: tool.name, arguments: {}, signal })
  const poseTool = egoTools(ctx.ego).find(tool => tool.name === 'ego_gripper_pose')
  const pose = () => poseTool.execute({}, { name: poseTool.name, arguments: {}, signal: AbortSignal.timeout(10000) })
  const calibrationTool = egoTools(ctx.ego).find(tool => tool.name === 'ego_camera_calibration')
  const readCalibration = () => calibrationTool.execute({}, { name: calibrationTool.name, arguments: {}, signal: AbortSignal.timeout(10000) })
  return { root, config, capture, pose, readCalibration, calibration, ctx, calls, state, bytes, packet, ids }
}
