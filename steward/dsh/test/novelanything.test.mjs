import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readFile, readdir, writeFile, symlink, mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { fixture } from './novelanything-fixture.mjs'
import { artifactPath } from '../packages/dsh-novelanything-client/lib/protocol.js'

test('uploads archive exact bytes and accepted request keys recover without duplicate POSTs', async t => {
  const f = await fixture(t)
  const args = { archive_path: 'scene.zip', request_key: 'upload-scene-001', dataset: 'scannetpp' }
  const uploaded = await f.call('upload', args)
  assert.deepEqual(await readFile(uploaded.local_input.archive_path), await readFile(join(f.root, 'scene.zip')))
  const post = f.calls.find(call => call.method === 'POST')
  assert.match(post.contentType, /^multipart\/form-data/)
  assert.match(post.body, /name="file"; filename="scene.zip"/)
  assert.match(post.body, /scannetpp/)
  assert.equal(post.authorization, undefined)
  await (await f.create()).call('upload', args)
  assert.equal(f.calls.filter(c => c.method === 'POST').length, 1)
  await writeFile(join(f.root, 'scene.zip'), 'PK changed')
  await assert.rejects(f.call('upload', args), e => e.code === 'KEY_CONFLICT')
})

test('recording preparation saves a bounded local ZIP and strips image bytes from model output', async t => {
  const f = await fixture(t)
  const args = { recording_id: f.recording.recording_id, num_source_views: 2 }
  const prepared = await f.call('prepare_gripper_view', args)
  assert.deepEqual(await readFile(prepared.archive_path), Buffer.from(f.prepared.archive_base64, 'base64'))
  assert.equal(prepared.metadata.camera_id, 'head_camera')
  assert.deepEqual(prepared.metadata.input_modalities, ['rgb'])
  assert.equal(prepared.label, 'gripper_view_prediction')
  assert.doesNotMatch(JSON.stringify(prepared), /archive_base64/)
  assert.equal(JSON.parse(await readFile(prepared.local_manifest_path, 'utf8')).fps, 2)
  assert.ok(f.calls.every(c => c.path.endsWith('/ego/rgb/prepare-novelanything')))
  f.prepared.metadata.camera_id = 'head_camera_left'
  await assert.rejects(f.call('prepare_gripper_view', args), e => e.code === 'INVALID_DATA')
  f.prepared.metadata.camera_id = 'head_camera'
  f.prepared.metadata.input_modalities = ['rgb', 'depth']
  await assert.rejects(f.call('prepare_gripper_view', args), e => e.code === 'INVALID_DATA')
  f.prepared.metadata.input_modalities = ['rgb']; f.prepared.bytes++
  await assert.rejects(f.call('prepare_gripper_view', args), e => e.code === 'INVALID_RECORDING')
  await assert.rejects(f.call('prepare_gripper_view', { ...args, recording_id: '../outside' }))
})

test('submission associates archived input, sends frame selection and recovers acceptance across restart', async t => {
  const f = await fixture(t)
  await f.call('upload', { archive_path: 'scene.zip', request_key: 'submit-upload-001' })
  const args = { upload_id: 'upload_one', context: [0, 1], target: [2], metrics: false, request_key: 'submit-scene-001' }
  const job = await f.call('submit', args)
  assert.equal(job.id, 'job_one')
  assert.ok(job.local_input.archive_path)
  assert.deepEqual(JSON.parse(f.calls.find(c => c.path.endsWith('/api/jobs')).body), {
    upload_id: 'upload_one', context: [0, 1], target: [2], metrics: false,
  })
  await (await f.create()).call('submit', args)
  assert.equal(f.calls.filter(c => c.method === 'POST').length, 2)
  await assert.rejects(f.call('submit', { ...args, metrics: true }), e => e.code === 'KEY_CONFLICT')
})

test('ambiguous POSTs are never repeated even after restart or concurrent calls', async t => {
  const f = await fixture(t); f.state.drop = true
  const args = { sample_id: 'fixture', request_key: 'lost-response-001' }
  const replies = await Promise.allSettled([f.call('submit', args), f.call('submit', args)])
  assert.ok(replies.every(r => r.status === 'rejected' && r.reason.code === 'UNKNOWN_OUTCOME'))
  await assert.rejects((await f.create()).call('submit', args), e => e.code === 'UNKNOWN_OUTCOME')
  assert.equal(f.calls.filter(c => c.method === 'POST').length, 1)
})

test('invalid sources, indices, readiness, paths and wait bounds send no POST', async t => {
  const f = await fixture(t)
  for (const fields of [{}, { sample_id: 'fixture', upload_id: 'upload_one' },
    { sample_id: '../outside' }, { sample_id: 'fixture', context: [0, 1] },
    { sample_id: 'fixture', context: [0, 1], target: [1] },
    { sample_id: 'fixture', context: [0, 0], target: [2] },
    { sample_id: 'fixture', context: [0, 1], target: [-1] }]) {
    await assert.rejects(f.call('submit', { request_key: 'invalid-001', ...fields }))
  }
  f.state.health = 'loading'
  await assert.rejects(f.call('submit', { sample_id: 'fixture', request_key: 'not-ready-001' }), e => e.code === 'NOT_READY')
  await assert.rejects(f.call('upload', { archive_path: '../outside.zip', request_key: 'outside-001' }))
  await assert.rejects(f.call('upload', { archive_path: 'scene.zip', dataset: 'bad', request_key: 'dataset-001' }))
  await assert.rejects(f.call('status', { job_id: 'job_one', wait_seconds: 100 }), e => e.code === 'INVALID_WAIT')
  assert.equal(f.calls.filter(c => c.method === 'POST').length, 0)
})

test('only succeeded jobs expose results; interrupted is terminal and wrong identities fail', async t => {
  const f = await fixture(t)
  assert.equal((await f.call('status', { job_id: 'job_one' })).next_action, 'novelanything_status')
  for (const status of ['queued', 'running', 'failed', 'cancelled', 'interrupted']) {
    f.state.status = status
    await assert.rejects(f.call('result', { job_id: 'job_one' }), e => e.code === 'RESULT_UNAVAILABLE')
  }
  const status = await f.call('status', { job_id: 'job_one', wait_seconds: 20 })
  assert.equal(status.next_action, 'novelanything_log')
  assert.match((await f.call('log', { job_id: 'job_one' })).log, /synthetic/)
  f.state.wrongId = true
  await assert.rejects(f.call('status', { job_id: 'job_one' }), e => e.code === 'INVALID_DATA')
})

test('downloads preserve directories, metadata and partial selections with optional server checksums', async t => {
  const f = await fixture(t); f.state.status = 'succeeded'
  const result = await f.call('result', { job_id: 'job_one' })
  const paths = result.result.artifacts.map(a => a.path)
  f.state.corrupt = true
  await assert.rejects(f.call('download', { job_id: 'job_one', artifact_paths: paths }), e => e.code === 'INTEGRITY_FAILURE')
  assert.ok(!(await readdir(join(f.root, 'outputs'), { recursive: true })).some(path => path.endsWith('.part')))
  f.state.corrupt = false
  const first = await f.call('download', { job_id: 'job_one', artifact_paths: paths.slice(0, 1) })
  f.state.checksum = false
  const last = await f.call('download', { job_id: 'job_one', artifact_paths: paths.slice(1) })
  assert.equal(first.files[0].server_checksum_verified, true)
  assert.ok(last.files.every(file => !file.server_checksum_verified))
  const manifest = JSON.parse(await readFile(last.local_manifest_path, 'utf8'))
  assert.equal(manifest.files.length, 3)
  assert.deepEqual(manifest.result.targets, result.result.targets)
  assert.deepEqual(manifest.result.metrics, {})
  for (const file of manifest.files) assert.deepEqual(await readFile(file.path), f.bytes)
  assert.equal(last.local_manifest_path, join(f.root, 'outputs/novelanything/job_one/manifest.json'))
})

test('untrusted artifact URLs, paths, duplicates and directory links cannot escape output root', async t => {
  const f = await fixture(t); f.state.status = 'succeeded'
  for (const key of ['badUrl', 'badPath', 'duplicate']) {
    f.state[key] = true
    await assert.rejects(f.call('result', { job_id: 'job_one' }), e => e.code === 'INVALID_ARTIFACT')
    f.state[key] = false
  }
  for (const path of ['../x', '/x', 'C:/x', 'output\\x', 'CON.txt', 'a/../x', 'manifest.json', 'x%2fy', 'x.']) {
    assert.throws(() => artifactPath(path))
  }
  const outside = join(f.root, 'other'); await mkdir(outside)
  const folder = join(f.root, 'outputs/novelanything/job_one'); await mkdir(folder, { recursive: true })
  // Link inside output tree stays inside inputRoot but must still stay inside this job directory.
  await symlink(outside, join(folder, 'output'), process.platform === 'win32' ? 'junction' : 'dir')
  await assert.rejects(f.call('download', { job_id: 'job_one', artifact_paths: ['output/pred_rgb.npy'] }))
  assert.deepEqual(await readdir(outside), [])
})

test('disposal waits for held downloads and removes incomplete files without remote cancellation', { timeout: 15000 }, async t => {
  const f = await fixture(t); f.state.status = 'succeeded'
  const started = Promise.withResolvers(); f.state.hold = started.resolve
  const task = f.call('download', { job_id: 'job_one', artifact_paths: ['output/pred_rgb.npy'] })
  const rejected = assert.rejects(task)
  await started.promise
  await f.ctx.fiber.dispose()
  await rejected
  assert.ok(!(await readdir(join(f.root, 'outputs'), { recursive: true })).some(path => path.endsWith('.part')))
  assert.ok(!f.calls.some(c => c.path.includes('cancel')))
})
