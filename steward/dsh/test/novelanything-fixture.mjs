// Isolated server for the built provider and real tool execution.
import { createServer } from 'node:http'
import { once } from 'node:events'
import { createHash } from 'node:crypto'
import { mkdtemp, realpath, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import Client from '@steward/dsh-novelanything-client'
import { novelAnythingTools } from '@steward/dsh-tool-novelanything'
import EgoRuntimeClient from '@steward/dsh-ego-runtime-client'

export async function fixture(t, overrides = {}) {
  const { egoEndpoint, ...providerOverrides } = overrides
  const root = await realpath(await mkdtemp(join(tmpdir(), 'steward-novelanything-')))
  const contexts = [], calls = []
  const bytes = Buffer.from('synthetic novel view RGB/depth')
  const state = { status: 'running', health: 'ready', drop: false, badUrl: false, badPath: false,
    corrupt: false, checksum: true, duplicate: false, hold: null, wrongId: false }
  const recordingId = 'rgb_' + 'a'.repeat(32)
  const recording = { recording_id: recordingId, status: 'recording', fps: 2, camera_id: 'head_camera', input_modalities: ['rgb'],
    captured_frames: 3, written_frames: 3, first_stamp: 0, last_stamp: 1,
    pose_source: 'robot_odometry_and_joint_encoders', clock: 'simulation_time', error: null }
  const prepared = { archive_base64: Buffer.from('PK synthetic robot RGB scene').toString('base64'),
    bytes: Buffer.byteLength('PK synthetic robot RGB scene'), num_source_views: 2, context: [0, 1], target: [2],
    metadata: { model: 'NovelAnything', recording_id: recordingId, camera_id: 'head_camera', input_modalities: ['rgb'],
      fps: 2, recorded_frames: 3, source_frame_indices: [0, 2], label: 'gripper_view_prediction',
      pose_source: 'robot_odometry_and_joint_encoders', target_stamp: 1.25 } }
  const artifacts = () => ['output/pred_rgb.npy', 'output/pred_depth.npy', 'output/pred_0.png'].map(path => ({
    path: state.badPath ? '../escape' : path,
    url: state.badUrl ? 'http://external.invalid/secret' : `api/jobs/job_one/artifacts/${path}`,
    ...(state.checksum ? { bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') } : {}),
  }))
  const job = () => ({ id: state.wrongId ? 'job_other' : 'job_one', status: state.status,
    progress: 100, error: null, ...(state.status === 'succeeded' ? { result: {
      targets: [{ frame_index: 2, fxfycxcy: [400, 400, 259, 168], calibration_source: 'first_context' }],
      metrics: {}, artifacts: state.duplicate ? [...artifacts(), artifacts()[0]] : artifacts(),
    } } : {}) })
  const server = createServer(async (req, res) => {
    const call = { method: req.method, path: req.url, authorization: req.headers.authorization,
      contentType: req.headers['content-type'] }
    calls.push(call)
    const json = (value, code = 200) => { res.writeHead(code, { 'content-type': 'application/json' }); res.end(JSON.stringify(value)) }
    try {
      if (req.url.startsWith('/novelanything/ego/') || req.url.startsWith('/novelanything/skill/')) {
        const chunks = []; for await (const chunk of req) chunks.push(chunk)
        call.body = Buffer.concat(chunks).toString()
        if (req.url.endsWith('/rgb/record/start')) return json(recording)
        if (req.url.includes('/rgb/record/status')) return json(recording)
        if (req.url.endsWith('/rgb/record/stop')) return json({ ...recording, status: 'succeeded' })
        if (req.url.endsWith('/rgb/prepare-novelanything')) return json(prepared)
        if (req.url.endsWith('/predict')) return json({ id: 'receipt_fixture', skill: 'navigate', params: { x: 0.5, y: -1.75, yaw: 1.5708 },
          risk: 0.01, maxRisk: 0.35, withinRiskLimit: true, feasible: true, issues: [], expiresAt: 100, plan: {} })
        if (req.url.endsWith('/receipt?id=receipt_fixture')) return json({ id: 'receipt_fixture', feasible: true,
          risk: 0.01, maxRisk: 0.35, withinRiskLimit: true, issues: [] })
        if (req.url.endsWith('/run') || req.url.includes('/status?job=')) return json({ id: 'job_nav', skill: 'navigate',
          receipt: 'receipt_fixture', state: req.url.endsWith('/run') ? 'running' : 'succeeded', message: '', log: [] })
        return json({ error: 'Unexpected robot fixture request' }, 503)
      }
      if (req.url === '/novelanything/api/health') return json({ status: state.health, model: 'NovelAnything', raster: [518, 336] })
      if (req.url.startsWith('/novelanything/api/samples')) return json({ samples: [{ id: 'fixture', context: [0, 1], target: [2] }] })
      if (req.method === 'POST') {
        const chunks = []; for await (const chunk of req) chunks.push(chunk)
        call.body = Buffer.concat(chunks).toString()
        if (state.drop) { req.socket.destroy(); return }
        if (req.url === '/novelanything/api/uploads') return json({ upload_id: 'upload_one', frame_count: 3 }, 201)
        if (req.url === '/novelanything/api/jobs') return json(job(), 202)
      }
      if (req.url === '/novelanything/api/jobs/job_one') return json(job())
      if (req.url === '/novelanything/api/jobs/job_one/log') { res.end('synthetic job log'); return }
      if (req.url.startsWith('/novelanything/api/jobs/job_one/artifacts/')) {
        res.writeHead(200)
        if (state.hold) { res.write(bytes.subarray(0, 1)); state.hold(); return }
        res.end(state.corrupt ? Buffer.from('broken') : bytes); return
      }
      json({ detail: 'Unknown fixture route' }, 404)
    } catch (error) { json({ detail: String(error) }, 500) }
  })
  t.after(async () => {
    try { await Promise.all(contexts.map(ctx => ctx.fiber.dispose())) } finally {
      server.closeAllConnections()
      await new Promise(resolve => server.close(resolve))
      await rm(root, { recursive: true, force: true })
    }
  })
  server.listen(0, '127.0.0.1'); await once(server, 'listening')
  await writeFile(join(root, 'scene.zip'), 'PK synthetic scene archive')
  const config = { endpoint: `http://127.0.0.1:${server.address().port}/novelanything`, inputRoot: root,
    tokenEnv: '', requestTimeoutMs: 5000, pollIntervalMs: 100, ...providerOverrides }
  async function create() {
    const ctx = new Context(); contexts.push(ctx)
    await ctx.plugin(EgoRuntimeClient, { endpoint: egoEndpoint ?? config.endpoint })
    await ctx.plugin(Client, config)
    const tools = new Map(novelAnythingTools(ctx.novelAnything).map(tool => [tool.name, tool]))
    const call = (name, args = {}, signal = AbortSignal.timeout(10000)) => {
      const tool = tools.get('novelanything_' + name)
      return tool.execute(args, { name: tool.name, arguments: args, signal })
    }
    return { ctx, call }
  }
  return { ...await create(), create, root, config, calls, state, bytes, job, prepared, recording }
}
