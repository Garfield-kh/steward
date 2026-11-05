// Opt-in inference against an operator-selected sample; no robot sensors or motion are used.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createHash, randomUUID } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, mkdtemp, readFile, stat, writeFile } from 'node:fs/promises'
import { resolve, join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import Client from '@steward/dsh-novelanything-client'
import { novelAnythingTools } from '@steward/dsh-tool-novelanything'
import EgoRuntimeClient from '@steward/dsh-ego-runtime-client'

test('live NovelAnything sample produces downloadable RGB/depth arrays', {
  skip: process.env.NOVELANYTHING_LIVE !== '1', timeout: 900000,
}, async t => {
  const parent = resolve(import.meta.dirname, '../../var/novelanything-validation')
  await mkdir(parent, { recursive: true })
  const root = await mkdtemp(join(parent, 'live-'))
  const ctx = new Context(); t.after(() => ctx.fiber.dispose())
  await ctx.plugin(EgoRuntimeClient, { endpoint: process.env.STEWARD_EGO_API ?? 'http://127.0.0.1:7420' })
  await ctx.plugin(Client, { endpoint: process.env.NOVELANYTHING_URL ?? 'http://115.120.57.9/novelanything',
    inputRoot: root, tokenEnv: process.env.NOVELANYTHING_LIVE_TOKEN_ENV ?? '' })
  const tools = new Map(novelAnythingTools(ctx.novelAnything).map(tool => [tool.name, tool]))
  const trace = [], signal = AbortSignal.timeout(840000)
  const call = async (name, args = {}) => {
    const tool = tools.get('novelanything_' + name)
    const output = await tool.execute(args, { name: tool.name, arguments: args, signal })
    trace.push({ name, args, output })
    await writeFile(join(root, 'trace.json'), JSON.stringify(trace, null, 2) + '\n')
    t.diagnostic(`${name}: ${output.status ?? 'ok'}`)
    return output
  }
  assert.equal((await call('health')).status, 'ready')
  const sample = process.env.NOVELANYTHING_SAMPLE ?? 'scannetpp-3f15a9266d'
  await call('samples', { sample_id: sample })
  const accepted = await call('submit', { sample_id: sample, context: [0, 10], target: [5], metrics: false,
    request_key: `live-${randomUUID()}` })
  let job = await call('status', { job_id: accepted.id, wait_seconds: 20 })
  while (['queued', 'running'].includes(job.status)) job = await call('status', { job_id: accepted.id, wait_seconds: 20 })
  assert.equal(job.status, 'succeeded', JSON.stringify(job))
  const result = await call('result', { job_id: accepted.id })
  const paths = result.result.artifacts.map(a => a.path)
  assert.ok(paths.some(path => path.endsWith('pred_rgb.npy')))
  assert.ok(paths.some(path => path.endsWith('pred_depth.npy')))
  const downloaded = await call('download', { job_id: accepted.id, artifact_paths: paths })
  for (const file of downloaded.files) {
    assert.equal((await stat(file.path)).size, file.bytes)
    const hash = createHash('sha256')
    for await (const chunk of createReadStream(file.path)) hash.update(chunk)
    assert.equal(hash.digest('hex'), file.sha256)
  }
  const manifest = JSON.parse(await readFile(downloaded.local_manifest_path, 'utf8'))
  assert.equal(manifest.input.request.sample_id, sample)
  t.diagnostic(`Retained local manifest: ${downloaded.local_manifest_path}`)
})
