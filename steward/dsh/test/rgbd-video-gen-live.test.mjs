// Opt-in real-service generation and independent file-integrity verification.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, stat, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import Client from '@steward/dsh-rgbd-video-gen-client'
import EgoRuntimeClient from '@steward/dsh-ego-runtime-client'
import { rgbdVideoGenTools } from '@steward/dsh-tool-rgbd-video-gen'

test('live rgbd_video_gen demo completes and downloads verified RGB/depth artifacts', {
  skip: process.env.RGBD_VIDEO_GEN_LIVE !== '1', timeout: 900000,
}, async t => {
  const root = resolve('../var/rgbd_video_gen-validation/live')
  const ctx = new Context()
  let id
  t.after(async () => {
    try {
      if (id) {
        const status = await ctx.rgbdVideoGen.status(id, 0, AbortSignal.timeout(30000))
        if (['queued', 'running'].includes(status.state)) await ctx.rgbdVideoGen.cancel(id, 20, AbortSignal.timeout(60000))
      }
    } finally { await ctx.fiber.dispose() }
  })
  await ctx.plugin(EgoRuntimeClient, { endpoint: process.env.STEWARD_EGO_API ?? 'http://127.0.0.1:7420' })
  await ctx.plugin(Client, { endpoint: process.env.RGBD_VIDEO_GEN_URL ?? 'http://115.120.57.9/nvs', inputRoot: '..', stateDir: root })
  const tools = Object.fromEntries(rgbdVideoGenTools(ctx.rgbdVideoGen).map(tool => [tool.name, tool]))
  const trace = []
  const signal = AbortSignal.timeout(840000)
  const call = async (name, args) => {
    const output = await tools[`rgbd_video_gen_${name}`].execute(args, { name, arguments: args, signal })
    trace.push({ name, args, output })
    await mkdir(root, { recursive: true })
    await writeFile(resolve(root, 'trace.json'), JSON.stringify(trace, null, 2) + '\n')
    t.diagnostic(`${name}: ${output.state ?? output.completed ?? 'ok'}`)
    return output
  }
  await call('capabilities', {})
  const args = { scene: 'waymo080', target_mode: 'trajectory', num_inference_steps: 30,
    max_target_views: 2, request_key: 'steward-rgbd-video-gen-live-v1' }
  const accepted = await call('demo', args)
  id = accepted.job_id
  let state = await call('status', { job_id: id, wait_seconds: 20 })
  while (['queued', 'running'].includes(state.state)) state = await call('status', { job_id: id, wait_seconds: 20 })
  assert.equal(state.state, 'succeeded', JSON.stringify(state))
  const result = await call('result', { job_id: id })
  assert.ok(result.artifacts.some(a => a.kind === 'rgb_video'))
  const files = await call('download', { job_id: id, filenames: result.artifacts.map(a => a.filename) })
  for (const file of files.files) {
    assert.equal((await stat(file.path)).size, file.bytes)
    const hash = createHash('sha256')
    for await (const chunk of createReadStream(file.path)) hash.update(chunk)
    assert.equal(hash.digest('hex'), file.sha256)
  }
  assert.equal((await call('demo', args)).job_id, id)
})
