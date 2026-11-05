import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readFile, readdir, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fixture } from './rgbd-video-gen-fixture.mjs'

test('JSON submission preserves the proxy prefix, content type and resolved defaults', async t => {
  const f = await fixture(t)
  assert.equal((await f.call('capabilities', {})).ready.status, 'ready')
  const args = { scene_path: 'scene.json', request_key: 'json-submit-001', num_inference_steps: 30 }
  const result = await f.call('submit', args)
  assert.equal(result.job_id, 'j_one')
  const post = f.calls.find(c => c.method === 'POST')
  assert.equal(post.path, '/nvs/v1/nvs/jobs')
  const body = JSON.parse(post.body)
  assert.deepEqual(JSON.parse(await readFile(result.local_input.request_path, 'utf8')), body)
  assert.deepEqual(JSON.parse(await readFile(result.local_input.scene_path, 'utf8')), body.scene)
  assert.equal(body.num_inference_steps, 30)
  assert.equal(body.num_source_views, 6)
  assert.equal(body.scene.frames[6].views.front.image, null)
  await (await f.create()).call('submit', args)
  assert.equal(f.state.postCount, 1)
  await assert.rejects(f.call('submit', { ...args, seed: 4 }), /LOCAL_IDEMPOTENCY_CONFLICT/)
  const record = await readFile(result.request_record, 'utf8')
  assert.doesNotMatch(record, /data:image/)
})

test('flexible evidence counts submit exact frames and explicit source counts without padding', async t => {
  for (const sources of [1, 3, 6, 8]) await t.test(`${sources} evidence views plus one target`, async t => {
    const f = await fixture(t)
    f.scene.frames = Array.from({ length: sources + 1 }, (_, i) => ({ views: { front: {
      image: i < sources ? f.scene.frames[0].views.front.image : null,
      camera_to_world: [[1, 0, 0, i], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    } } }))
    await writeFile(join(f.root, 'scene.json'), JSON.stringify(f.scene))
    const args = { scene_path: 'scene.json', request_key: `flexible-source-${sources}` }
    if (sources !== 6) args.num_source_views = sources
    await f.call('submit', args)
    const body = JSON.parse(f.calls.find(c => c.method === 'POST').body)
    assert.equal(body.num_source_views, sources)
    assert.equal(body.scene.frames.length, sources + 1)
    assert.equal(body.scene.frames[sources].views.front.image, null)
    assert.deepEqual(body.scene, f.scene)
  })
})

test('short scenes require a valid evidence count, source images and at least one target', async t => {
  const f = await fixture(t)
  const original = structuredClone(f.scene)
  for (const [frames, sources, missingImage] of [[1, 1, false], [4, undefined, false], [3, 3, false],
    [10, 9, false], [2, 0, false], [2, 1, true], [65, 1, false]]) {
    const scene = structuredClone(original)
    scene.frames = Array.from({ length: frames }, () => structuredClone(original.frames[0]))
    if (missingImage) scene.frames[0].views.front.image = null
    await writeFile(join(f.root, 'scene.json'), JSON.stringify(scene))
    const args = { scene_path: 'scene.json', request_key: 'invalid-flexible-frames' }
    if (sources !== undefined) args.num_source_views = sources
    await assert.rejects(f.call('submit', args))
  }
  assert.equal(f.state.postCount, 0)
})

test('lost responses retry the same durable key; demo BEV options reach the server', async t => {
  const f = await fixture(t)
  f.state.drop = true
  const args = { ...f.demo, target_mode: 'bev', num_bev_views: 5, bev_altitude_scale: 1.2 }
  await f.call('demo', args)
  const posts = f.calls.filter(c => c.method === 'POST')
  assert.equal(posts.length, 2)
  assert.equal(posts[0].key, posts[1].key)
  assert.equal(JSON.parse(posts[1].body).target_mode, 'bev')
  await (await f.create()).call('demo', args)
  assert.equal(f.state.postCount, 1)
  await assert.rejects(f.call('demo', { ...f.demo, target_mode: 'bev' }), /requires/)
})

test('bare JPEG base64 is sent as a data URI and server-local paths are refused', async t => {
  const f = await fixture(t)
  f.scene.frames[0].views.front.image = '/9j/4AAQSkZJRg=='
  await writeFile(join(f.root, 'scene.json'), JSON.stringify(f.scene))
  await f.call('submit', { scene_path: 'scene.json', request_key: 'bare-image-001' })
  const scene = JSON.parse(f.calls.find(c => c.method === 'POST').body).scene
  assert.equal(scene.frames[0].views.front.image, 'data:image/jpeg;base64,/9j/4AAQSkZJRg==')
  assert.deepEqual(scene.frames[0].views.front.camera_to_world, f.scene.frames[0].views.front.camera_to_world)
  f.scene.frames[0].views.front.image = '/etc/passwd'
  await writeFile(join(f.root, 'scene.json'), JSON.stringify(f.scene))
  await assert.rejects(f.call('submit', { scene_path: 'scene.json', request_key: 'server-path-001' }), /INVALID_IMAGE/)
  assert.equal(f.state.postCount, 1)
})

test('ZIP upload uses multipart archive and rejects changed input on replay', async t => {
  const f = await fixture(t)
  const args = { archive_path: 'scene.zip', request_key: 'zip-upload-001' }
  const accepted = await f.call('upload', args)
  const archive = await readFile(accepted.local_input.archive_path)
  const post = f.calls.find(c => c.method === 'POST')
  assert.match(post.contentType, /^multipart\/form-data; boundary=/)
  await writeFile(join(f.root, 'scene.zip'), 'PK changed')
  await assert.rejects(f.call('upload', args), /LOCAL_IDEMPOTENCY_CONFLICT/)
  assert.deepEqual(await readFile(accepted.local_input.archive_path), archive)
  assert.equal(f.state.postCount, 1)
})

test('invalid input and generation settings make no POST', async t => {
  const f = await fixture(t)
  for (const change of [{ num_inference_steps: 101 }, { seed: -1 }, { output_fps: 0 }, { num_source_views: 7 }, { request_key: 'bad' }]) {
    await assert.rejects(f.call('submit', { scene_path: 'scene.json', request_key: 'invalid-submit-001', ...change }))
  }
  f.scene.frames[0].views.front.image = 'local.png'
  await writeFile(join(f.root, 'scene.json'), JSON.stringify(f.scene))
  await assert.rejects(f.call('submit', { scene_path: 'scene.json', request_key: 'invalid-scene-001' }), /INVALID_IMAGE/)
  const outside = join(f.root, 'state', '..', '..', 'outside.zip')
  await writeFile(join(f.root, 'wrong.txt'), 'x')
  await assert.rejects(f.call('upload', { archive_path: 'wrong.txt', request_key: 'invalid-zip-001' }), /INVALID_INPUT_PATH/)
  await assert.rejects(f.call('upload', { archive_path: outside, request_key: 'outside-zip-001' }))
  assert.equal(f.state.postCount, 0)
})

test('progress cannot imply success and remote failures retain their code', async t => {
  const f = await fixture(t)
  assert.equal((await f.call('status', { job_id: 'j_one' })).next_action, 'rgbd_video_gen_status')
  await assert.rejects(f.call('result', { job_id: 'j_one' }), /RESULT_NOT_READY/)
  f.state.job = 'failed'
  await assert.rejects(f.call('result', { job_id: 'j_one' }), /RESULT_UNAVAILABLE/)
  f.state.readyError = true
  await assert.rejects(f.call('capabilities', {}), e => e.code === 'UNAVAILABLE' && e.status === 503 && !e.retryable)
  f.state.wrongId = true
  await assert.rejects(f.call('status', { job_id: 'j_one' }), /INVALID_METADATA/)
  await assert.rejects(f.call('status', { job_id: '../escape' }), /INVALID_JOB_ID/)
})

test('only verified artifacts are published, reused and represented by the saved manifest', async t => {
  const f = await fixture(t); f.state.job = 'succeeded'; f.state.prefixed = true
  const result = await f.call('result', { job_id: 'j_one' })
  assert.equal(result.completed, true)
  assert.equal(JSON.parse(await readFile(result.local_result_path, 'utf8')).job_id, 'j_one')
  const args = { job_id: 'j_one', filenames: result.artifacts.map(a => a.filename) }
  f.state.corrupt = true
  await assert.rejects(f.call('download', args), /INTEGRITY_FAILURE/)
  assert.ok(!(await readdir(join(f.root, 'outputs', 'rgbd_video_gen'), { recursive: true })).some(p => p.endsWith('.part') || p.endsWith('frames.zip')))
  f.state.corrupt = false
  const downloaded = await f.call('download', args)
  assert.equal(downloaded.local_manifest_path, join(f.root, 'outputs', 'rgbd_video_gen', 'j_one', 'manifest.json'))
  const manifest = JSON.parse(await readFile(downloaded.local_manifest_path, 'utf8'))
  assert.equal(manifest.files.length, 4)
  assert.ok(manifest.files.every(file => file.verified && file.path.startsWith(join(f.root, 'outputs'))))
  await f.call('download', { job_id: 'j_one', filenames: ['depth.npz'] })
  assert.equal(JSON.parse(await readFile(downloaded.local_manifest_path, 'utf8')).files.length, 4)
  for (const file of downloaded.files) {
    assert.ok(file.verified && !file.reused)
    assert.deepEqual(await readFile(file.path), f.bytes)
  }
  assert.ok((await f.call('download', args)).files.every(file => file.reused))
  f.state.badUrl = true
  await assert.rejects(f.call('result', { job_id: 'j_one' }), /INVALID_METADATA/)
  f.state.badUrl = false; f.state.duplicate = true
  await assert.rejects(f.call('result', { job_id: 'j_one' }), /INVALID_METADATA/)
})

test('output metadata cannot be overwritten by a different endpoint sharing a job id', async t => {
  const f = await fixture(t); f.state.job = 'succeeded'
  const manifest = join(f.root, 'outputs', 'rgbd_video_gen', 'j_one', 'manifest.json')
  await f.call('result', { job_id: 'j_one' })
  const saved = JSON.parse(await readFile(manifest, 'utf8'))
  saved.endpoint = 'https://another.example/nvs'
  await writeFile(manifest, JSON.stringify(saved))
  await assert.rejects(f.call('download', { job_id: 'j_one', filenames: ['depth.npz'] }), /LOCAL_OUTPUT_CONFLICT/)
  assert.equal(JSON.parse(await readFile(manifest, 'utf8')).endpoint, saved.endpoint)
})

test('local manifest associates archived evidence with target RGB, depth and camera across partial selections', async t => {
  const f = await fixture(t)
  f.scene.frames = f.scene.frames.slice(0, 4)
  await writeFile(join(f.root, 'scene.json'), JSON.stringify(f.scene))
  const accepted = await f.call('submit', { scene_path: 'scene.json', request_key: 'target-mapping-001', num_source_views: 3 })
  f.state.job = 'succeeded'; f.state.targetRgb = true
  const camera = { K: f.scene.cameras.front.K, camera_to_world: f.scene.frames[3].views.front.camera_to_world }
  f.state.targetCameras = [camera]
  const first = await f.call('download', { job_id: 'j_one', filenames: ['depth.npz'] })
  let manifest = JSON.parse(await readFile(first.local_manifest_path, 'utf8'))
  assert.deepEqual(manifest.input, accepted.local_input)
  assert.equal(manifest.targets[0].rgb_path, null)
  assert.equal(manifest.targets[0].depth_key, 'v03')
  assert.deepEqual(manifest.targets[0].camera, camera)
  const rgb = await f.call('download', { job_id: 'j_one', filenames: ['pred_03.jpg'] })
  manifest = JSON.parse(await readFile(rgb.local_manifest_path, 'utf8'))
  assert.equal(manifest.files.length, 2)
  assert.equal(manifest.targets[0].rgb_path, rgb.files[0].path)
  assert.equal(rgb.files[0].local_filename, 'target_prediction_00.jpg')
  assert.ok(rgb.files[0].path.endsWith('target_prediction_00.jpg'))
  assert.equal(manifest.targets[0].depth_path, first.files[0].path)
  f.state.reservedName = true
  await assert.rejects(f.call('result', { job_id: 'j_one' }), /INVALID_METADATA/)
  assert.deepEqual(JSON.parse(await readFile(rgb.local_manifest_path, 'utf8')), manifest)
})

test('cancellation requires ownership and sends a stable idempotency header', async t => {
  const f = await fixture(t)
  await assert.rejects(f.call('cancel', { job_id: 'j_one' }), /UNKNOWN_LOCAL_TASK/)
  await f.call('demo', f.demo)
  assert.equal((await f.call('cancel', { job_id: 'j_one' })).cancelled, true)
  await (await f.create()).call('cancel', { job_id: 'j_one' })
  const calls = f.calls.filter(c => c.path.endsWith('/cancel'))
  assert.equal(calls[0].key, calls[1].key)
})

test('disposal aborts an in-flight download, removes partial files and sends no remote cancel', async t => {
  const f = await fixture(t); f.state.job = 'succeeded'
  const started = Promise.withResolvers(); f.state.hold = started.resolve
  const task = f.call('download', { job_id: 'j_one', filenames: ['frames.zip'] })
  const rejected = assert.rejects(task)
  await started.promise
  await f.ctx.fiber.dispose()
  await rejected
  assert.ok(!(await readdir(join(f.root, 'outputs', 'rgbd_video_gen'), { recursive: true })).some(p => p.endsWith('.part')))
  assert.ok(!f.calls.some(c => c.path.endsWith('/cancel')))
})


test('local frame names distinguish source reconstructions and zero-based target predictions, reusing verified legacy bytes', async t => {
  const f = await fixture(t); f.state.job = 'succeeded'
  f.state.params.num_source_views = 3
  f.state.targetCameras = [{}, {}]
  f.state.frameNames = ['pred_00.jpg', 'pred_03.jpg', 'pred_04.jpg']
  const result = await f.call('result', { job_id: 'j_one' })
  const frames = result.artifacts.filter(a => a.filename.startsWith('pred_'))
  assert.deepEqual(frames.map(a => a.local_filename), ['source_reconstruction_00.jpg', 'target_prediction_00.jpg', 'target_prediction_01.jpg'])
  const directory = join(f.root, 'outputs', 'rgbd_video_gen', 'j_one')
  await writeFile(join(directory, 'pred_00.jpg'), f.bytes)
  const downloaded = await f.call('download', { job_id: 'j_one', filenames: frames.map(a => a.filename) })
  assert.equal(downloaded.files[0].reused, true)
  assert.ok(!f.calls.some(c => c.path.endsWith('/artifacts/pred_00.jpg')))
  assert.ok(f.calls.some(c => c.path.endsWith('/artifacts/pred_03.jpg')))
  for (const file of downloaded.files) {
    assert.equal(file.path, join(directory, file.local_filename))
    assert.deepEqual(await readFile(file.path), f.bytes)
  }
  assert.ok(!(await readdir(directory)).includes('pred_00.jpg'))
  const manifest = JSON.parse(await readFile(downloaded.local_manifest_path, 'utf8'))
  assert.deepEqual(manifest.targets.map(t => t.rgb_path), downloaded.files.slice(1).map(f => f.path))
  assert.deepEqual(manifest.targets.map(t => t.depth_key), ['v03', 'v04'])
  assert.ok((await f.call('download', { job_id: 'j_one', filenames: frames.map(a => a.filename) })).files.every(f => f.reused))
})

test('ambiguous local names and prediction indices without cameras are rejected before publication', async t => {
  const f = await fixture(t); f.state.job = 'succeeded'
  for (const names of [['pred_00.jpg', 'source_reconstruction_00.jpg'], ['pred_99.jpg']]) {
    f.state.frameNames = names
    await assert.rejects(f.call('result', { job_id: 'j_one' }), /INVALID_METADATA/)
  }
  assert.ok(!f.calls.some(c => c.path.includes('/artifacts/')))
})
