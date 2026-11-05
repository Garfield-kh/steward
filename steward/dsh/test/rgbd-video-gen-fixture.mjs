// Isolated HTTP fixture for the built rgbd_video_gen provider and tools.
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { createServer } from 'node:http'
import { once } from 'node:events'
import { mkdtemp, realpath, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import Client from '@steward/dsh-rgbd-video-gen-client'
import EgoRuntimeClient from '@steward/dsh-ego-runtime-client'
import { rgbdVideoGenTools } from '@steward/dsh-tool-rgbd-video-gen'

export const defaults = { num_source_views: 6, max_target_views: 10, num_inference_steps: 50,
  seed: 1234, output_fps: 4, flow_dtype: 'bf16' }
const hash = bytes => createHash('sha256').update(bytes).digest('hex')

export async function fixture(t, overrides = {}) {
  const { egoEndpoint, ...providerOverrides } = overrides
  const root = await realpath(await mkdtemp(join(tmpdir(), 'steward-rgbd-video-gen-')))
  const contexts = []
  const calls = []
  const egoCalls = []
  const keys = new Map()
  const bytes = Buffer.from('generated rgbd fixture')
  const scene = { format: 'rgbd_videogen_scene_v1', reference_camera: 'front',
    cameras: { front: { width: 32, height: 32, K: [[32, 0, 16], [0, 32, 16], [0, 0, 1]] } },
    frames: Array.from({ length: 7 }, (_, i) => ({ views: { front: {
      image: i < 6 ? 'data:image/png;base64,AA==' : null,
      camera_to_world: [[1, 0, 0, i], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    } } })) }
  const recordingId = 'rgb_' + 'c'.repeat(32)
  const evidence = { recording_id: recordingId, status: 'succeeded', fps: 2, camera_id: 'head_camera',
    input_modalities: ['rgb'], reference_frame: 'first_recorded_head_camera_opencv', target_stamp: 18.25,
    pose_source: 'robot_odometry_and_joint_encoders',
    gripper_to_reference: [[0, -1, 0, 0], [-0.5, 0, -Math.sqrt(3) / 2, 0.71], [Math.sqrt(3) / 2, 0, -0.5, 0.58], [0, 0, 0, 1]],
    frames: Array.from({ length: 37 }, (_, i) => ({ stamp: i / 2, width: 320, height: 240,
      K: [[200, 0, 159.5], [0, 200, 119.5], [0, 0, 1]],
      camera_to_reference: [[1, 0, 0, i * 0.05], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      image: 'data:image/jpeg;base64,' + Buffer.concat([Buffer.from([0xff, 0xd8, 0xff]), Buffer.from(`frame-${i}`)]).toString('base64'),
      odometry: { x: i * 0.05, y: 0, yaw: 0 }, joint_encoders: { lift: 0.59, reach: 0.1 } })) }
  const state = { job: 'running', drop: false, corrupt: false, badUrl: false, prefixed: false,
    wrongId: false, duplicate: false, postCount: 0, hold: null, readyError: false, params: { ...defaults } }
  const job = () => ({ job_id: state.wrongId ? 'j_other' : 'j_one', state: state.job,
    progress: { stage: state.job === 'succeeded' ? 'done' : 'sample', percent: 100 }, error: null })
  const manifest = () => {
    const artifacts = ['frames.zip', 'depth.npz', 'demo_pred_tgt.mp4', 'meta.json'].map((filename, i) => ({
      kind: ['frames_zip', 'depth_npz', 'rgb_video', 'meta'][i], filename,
      bytes: bytes.length, sha256: hash(bytes), media_type: 'application/octet-stream',
    }))
    if (state.targetRgb) artifacts.push({ ...artifacts[0], kind: 'rgb', filename: 'pred_03.jpg' })
    for (const filename of state.frameNames ?? []) artifacts.push({ ...artifacts[0], kind: 'frame_pred', filename })
    if (state.reservedName) artifacts[0].filename = 'manifest.json'
    if (state.duplicate) artifacts.push({ ...artifacts[0], filename: 'FRAMES.zip' })
    return { job_id: state.wrongId ? 'j_other' : 'j_one', state: 'succeeded', metrics: { src_recon_rgb_psnr: 25.5 },
      params: state.params, scene: { scene_id: 'fixture' }, meta: { target_cameras: state.targetCameras ?? [], has_target_gt: false }, artifacts,
      artifact_urls: Object.fromEntries(artifacts.map(a => [a.filename, state.badUrl ? 'http://other.invalid/secret'
        : `${state.prefixed ? '/nvs' : ''}/v1/nvs/jobs/j_one/artifacts/${a.filename}`])) }
  }
  const server = createServer(async (req, res) => {
    try {
      const json = (value, code = 200) => { res.writeHead(code, { 'content-type': 'application/json' }); res.end(JSON.stringify(value)) }
      const call = { method: req.method, path: req.url, key: req.headers['idempotency-key'], contentType: req.headers['content-type'] }
      calls.push(call)
      if (req.url.startsWith('/ego/') || req.url.startsWith('/skill/')) {
        const chunks = []; for await (const chunk of req) chunks.push(chunk)
        call.body = Buffer.concat(chunks).toString()
        egoCalls.push(call)
        if (state.recordingEnabled && req.method === 'POST' && req.url === '/ego/rgb/record/export') return json(evidence)
        return json({ error: 'Ego API disabled in RGBD synthesis fixture' }, 503)
      }
      if (req.url === '/nvs/health/ready') return json({ status: 'ready' })
      if (req.url === '/nvs/v1/capabilities') {
        if (state.readyError) return json({ error: { code: 'UNAVAILABLE', message: 'model unavailable', retryable: false } }, 503)
        return json({ input_format: scene.format, defaults, ready: true })
      }
      if (req.method === 'POST') {
        assert.match(call.key, /^[A-Za-z0-9._:-]{8,128}$/)
        const chunks = []; for await (const chunk of req) chunks.push(chunk)
        const body = Buffer.concat(chunks).toString()
        call.body = body
        if (req.url.endsWith('/cancel')) { state.job = 'cancelled'; return json(job()) }
        assert.ok(['/nvs/v1/nvs/demo', '/nvs/v1/nvs/jobs', '/nvs/v1/nvs/jobs/upload'].includes(req.url))
        if (req.url.endsWith('/upload')) assert.match(body, /name="archive"; filename="scene.zip"/)
        else {
          assert.equal(call.contentType, 'application/json')
          const parsed = JSON.parse(body)
          state.params = Object.fromEntries(Object.entries(defaults).map(([key, value]) => [key, parsed[key] ?? value]))
        }
        if (!keys.has(call.key)) { keys.set(call.key, job()); state.postCount++ }
        if (state.drop) { state.drop = false; req.socket.destroy(); return }
        return json(keys.get(call.key), 202)
      }
      if (req.url === '/nvs/v1/nvs/jobs/j_one') return json(job())
      if (req.url === '/nvs/v1/nvs/jobs/j_one/result') return json(manifest())
      if (req.url.startsWith('/nvs/v1/nvs/jobs/j_one/artifacts/')) {
        if (state.hold) { state.hold(); res.writeHead(200); res.write(bytes.subarray(0, 1)); return }
        res.writeHead(200); res.end(state.corrupt ? Buffer.alloc(bytes.length) : bytes); return
      }
      json({ error: { code: 'NOT_FOUND', message: req.url, retryable: false } }, 404)
    } catch (error) { res.writeHead(500); res.end(JSON.stringify({ error: { code: 'FIXTURE', message: String(error) } })) }
  })
  t.after(async () => {
    try { await Promise.all(contexts.map(ctx => ctx.fiber.dispose())) } finally {
      server.closeAllConnections()
      await new Promise(resolve => server.close(resolve))
      await rm(root, { recursive: true, force: true })
    }
  })
  server.listen(0, '127.0.0.1'); await once(server, 'listening')
  await writeFile(join(root, 'scene.json'), JSON.stringify(scene))
  await writeFile(join(root, 'scene.zip'), 'PK fixture')
  const config = { endpoint: `http://127.0.0.1:${server.address().port}/nvs`, inputRoot: root,
    stateDir: join(root, 'state'), retryDelayMs: 1, requestTimeoutMs: 5000, pollIntervalMs: 100, ...providerOverrides }
  const create = async () => {
    const ctx = new Context(); contexts.push(ctx)
    await ctx.plugin(EgoRuntimeClient, { endpoint: egoEndpoint ?? `http://127.0.0.1:${server.address().port}` })
    await ctx.plugin(Client, config)
    const tools = Object.fromEntries(rgbdVideoGenTools(ctx.rgbdVideoGen).map(tool => [tool.name, tool]))
    return { ctx, client: ctx.rgbdVideoGen, call: (name, args, signal = AbortSignal.timeout(10000)) => tools[`rgbd_video_gen_${name}`].execute(args,
      { name: `rgbd_video_gen_${name}`, arguments: args, signal }) }
  }
  return { ...await create(), create, root, state, calls, egoCalls,
    egoEndpoint: `http://127.0.0.1:${server.address().port}`, config, scene, bytes, manifest, evidence, recordingId,
    demo: { scene: 'waymo080', target_mode: 'trajectory', request_key: 'fixture-demo-001' } }
}
