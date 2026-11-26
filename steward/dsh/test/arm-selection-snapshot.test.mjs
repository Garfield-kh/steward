// Synthetic Session replay through the shipped Ego profile and HTTP provider, without a model key.
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { once } from 'node:events'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'
import { test } from 'node:test'
import { launchProfile } from './profile-test.mjs'

const workspace = resolve(import.meta.dirname, '..')
const require = createRequire(import.meta.url)

test('Ego profile replays explicit R selection without losing the model argument', { timeout: 120000 }, async t => {
  const root = await mkdtemp(join(tmpdir(), 'steward-arm-replay-'))
  t.after(() => rm(root, { recursive: true, force: true }))
  const calls = []
  const receipt = { id: 'rcpt-1', skill: 'pick', params: { target: 'e:driver', arm: 'R' },
    feasible: true, risk: 0.01, maxRisk: 0.35, withinRiskLimit: true, issues: [], expiresAt: 60,
    plan: { target: 'e:driver' } }
  const job = { id: 'job-1', skill: 'pick', receipt: 'rcpt-1', state: 'succeeded', message: 'picked', log: [] }
  const server = createServer(async (req, res) => {
    let raw = ''
    for await (const chunk of req) raw += chunk
    const body = raw ? JSON.parse(raw) : undefined
    const path = new URL(req.url, 'http://localhost').pathname
    calls.push({ path, body })
    const value = path === '/ego/predict' || path === '/ego/receipt' ? receipt
      : path === '/skill/run' || path === '/skill/wait' ? job : undefined
    res.writeHead(value === undefined ? 404 : 200, { 'content-type': 'application/json' })
    res.end(JSON.stringify(value ?? { error: 'Unexpected request' }))
  })
  server.listen(0, '127.0.0.1')
  t.after(async () => { server.closeAllConnections(); await new Promise((ok, fail) => server.close(e => e ? fail(e) : ok())) })
  await once(server, 'listening')
  const endpoint = `http://127.0.0.1:${server.address().port}`
  const env = { ...process.env, DSH_HOME: join(root, 'home'),
    DSH_SNAPSHOT_FILE: join(import.meta.dirname, 'snapshots/arm-selection/session.v4.jsonl') }
  delete env.DEEPSEEK_API_KEY
  delete env.DEEPSEEK_BASE_URL
  const launch = args => launchProfile(args, root, env)
  await launch(['--profile', 'ego', '--from-default-profile', 'headless', '--dump-default-config'])
  await launch(['plugin', '--profile', 'ego', 'add', join(workspace, 'packages/dsh-bundle-ego')])
  const patch = [
    { id: 'tools', config: { mode: 'native' } },
    { id: 'session-title-llm', disabled: true },
    { id: 'steward-ego-runtime', config: { endpoint, timeoutMs: 5000 } },
    { id: 'agent-default-model', config: { provider: 'deepseek-official', model: 'deepseek-flash' } },
    { id: 'llm-deepseek', disabled: true }, { id: 'llm-deepseek-account', disabled: true },
    { insert: [{ id: 'llm-replay', name: pathToFileURL(require.resolve('@deepseek-ai/dsh-llm-replay')).href,
      config: { providers: [{ id: 'deepseek-official', models: [{ id: 'deepseek-flash', contextWindow: 128000, reasoningEfforts: ['low'] }] }] } }] },
  ]
  const patchPath = join(root, 'arm.patch.json')
  await writeFile(patchPath, JSON.stringify(patch))
  const output = await launch(['--profile', 'ego', '--patch', patchPath, '--json', 'Pick e:driver using the right arm, then execute its receipt.'])
  const events = output.trim().split('\n').filter(line => line.startsWith('{')).map(line => JSON.parse(line))
  const results = events.filter(e => e.type === 'tool_result')
  assert.equal(results.length, 2, output)
  assert.ok(results.every(e => e.status === 'completed'), output)
  assert.equal(JSON.parse(results[0].result).arm, 'R')
  assert.deepEqual(calls.find(c => c.path === '/ego/predict').body, { skill: 'pick', params: { target: 'e:driver', arm: 'R' } })
  assert.equal(calls.filter(c => c.path === '/skill/run').length, 1)
  assert.ok(events.some(e => e.type === 'final'))
})
