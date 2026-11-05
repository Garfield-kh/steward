import assert from 'node:assert/strict'
import { readFile, readdir, mkdir, symlink } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { test } from 'node:test'
import { fixture } from './rgbd-video-gen-fixture.mjs'

const targets = [{ frame: 'gripper', position: [0, 0, 0], look_at: [1, 0, 0], up: [0, 0, 1] }]

test('recording preparation retains every frame and sends chronological evidence spanning the entire route', async t => {
  const f = await fixture(t); f.state.recordingEnabled = true
  const args = { recording_id: f.recordingId, targets }
  const prepared = await f.call('prepare_recording', args)
  assert.equal(prepared.recorded_frames, 37)
  assert.equal(prepared.all_frames_used, false)
  assert.deepEqual(prepared.source_frame_indices, [0, 5, 10, 15, 21, 26, 31, 36])
  assert.deepEqual(prepared.source_stamps, [0, 2.5, 5, 7.5, 10.5, 13, 15.5, 18])
  assert.doesNotMatch(JSON.stringify(prepared), /base64|camera_to_world/)
  const evidence = JSON.parse(await readFile(prepared.recording_manifest_path, 'utf8'))
  assert.equal(evidence.frames.length, 37)
  assert.equal((await readdir(dirname(prepared.recording_manifest_path))).filter(p => p.endsWith('.jpg')).length, 37)
  const scene = JSON.parse(await readFile(prepared.local_scene_path, 'utf8'))
  assert.equal(scene.frames.length, 9)
  for (const [i, index] of prepared.source_frame_indices.entries()) {
    assert.equal(scene.frames[i].views.head_camera.image, f.evidence.frames[index].image)
    assert.deepEqual(scene.frames[i].views.head_camera.camera_to_world, f.evidence.frames[index].camera_to_reference)
    assert.deepEqual(await readFile(join(dirname(prepared.recording_manifest_path), evidence.frames[index].image_path)),
      Buffer.from(f.evidence.frames[index].image.split(',')[1], 'base64'))
  }
  await f.call('submit', { scene_path: prepared.scene_path, num_source_views: 8, max_target_views: 1, request_key: 'all-route-frames-001' })
  const body = JSON.parse(f.calls.find(call => call.path.endsWith('/v1/nvs/jobs')).body)
  assert.equal(body.num_source_views, 8)
  assert.deepEqual(body.scene, scene)
  assert.equal(f.egoCalls.length, 1)
  assert.equal(f.egoCalls[0].path, '/ego/rgb/record/export')
  await assert.rejects(f.call('submit', { scene_path: prepared.scene_path, num_source_views: 1, request_key: 'wrong-count-001' }),
    e => e.code === 'INVALID_SCENE')
  assert.equal(f.state.postCount, 1)
})

test('default sampling uses every frame up to eight and preserves both endpoints above the boundary', async t => {
  const f = await fixture(t); f.state.recordingEnabled = true
  const original = structuredClone(f.evidence.frames)
  for (const [total, expected] of [[2, [0, 1]], [8, [0, 1, 2, 3, 4, 5, 6, 7]], [9, [0, 1, 2, 3, 5, 6, 7, 8]]]) {
    f.evidence.frames = original.slice(0, total)
    const result = await f.call('prepare_recording', { recording_id: f.recordingId, targets })
    assert.equal(result.num_source_views, Math.min(total, 8))
    assert.deepEqual(result.source_frame_indices, expected)
    assert.equal(result.all_frames_used, total <= 8)
    assert.equal(result.source_stamps[0], 0)
    assert.equal(result.source_stamps.at(-1), (total - 1) / 2)
  }
})

test('all short-recording frames are used by default; explicit batches collectively retain every frame', async t => {
  const f = await fixture(t); f.state.recordingEnabled = true
  f.evidence.frames = f.evidence.frames.slice(0, 3)
  const all = await f.call('prepare_recording', { recording_id: f.recordingId, targets })
  assert.equal(all.all_frames_used, true)
  assert.deepEqual(all.source_frame_indices, [0, 1, 2])
  const indices = []
  for (const batch of [[0, 1], [2]]) {
    const prepared = await f.call('prepare_recording', { recording_id: f.recordingId, targets, source_frame_indices: batch })
    const scene = JSON.parse(await readFile(prepared.local_scene_path, 'utf8'))
    indices.push(...prepared.source_frame_indices)
    assert.equal(scene.frames.length, batch.length + 1)
  }
  assert.deepEqual(indices, [0, 1, 2])
})

test('invalid explicit selections and malformed exports never publish a prepared scene', async t => {
  const f = await fixture(t); f.state.recordingEnabled = true
  const args = { recording_id: f.recordingId, targets }
  for (const selection of [{ source_frame_indices: [1, 1] }, { source_frame_indices: [2, 1] },
    { source_frame_indices: [37] }, { num_source_views: 9 }, { num_source_views: 8, source_frame_indices: [0] }]) {
    await assert.rejects(f.call('prepare_recording', { ...args, ...selection }))
  }
  const original = structuredClone(f.evidence)
  for (const change of [
    value => { value.recording_id = 'rgb_' + 'd'.repeat(32) },
    value => { value.camera_id = 'head_camera_left' },
    value => { value.frames[0].camera_to_reference[0][3] = 1 },
    value => { value.frames[2].stamp += 0.1 },
    value => { value.frames[2].image = 'data:image/jpeg;base64,YWJj' },
    value => { value.frames[2].K[0][0] += 1 },
  ]) {
    Object.assign(f.evidence, structuredClone(original)); change(f.evidence)
    await assert.rejects(f.call('prepare_recording', { ...args, num_source_views: 8 }))
  }
  await assert.rejects(readdir(join(f.root, 'inputs/rgbd_video_gen/prepared')), { code: 'ENOENT' })
  assert.equal(f.state.postCount, 0)
})

test('recording preparation cannot write through an input directory link or publish a cancelled scene', async t => {
  const f = await fixture(t); f.state.recordingEnabled = true
  await assert.rejects(f.call('prepare_recording', { recording_id: f.recordingId, targets, num_source_views: 8 }, AbortSignal.abort()))
  const outside = await fixture(t)
  await mkdir(join(f.root, 'inputs'), { recursive: true })
  await symlink(outside.root, join(f.root, 'inputs/rgbd_video_gen'), process.platform === 'win32' ? 'junction' : 'dir')
  const prior = (await readdir(outside.root)).sort()
  await assert.rejects(f.call('prepare_recording', { recording_id: f.recordingId, targets, num_source_views: 8 }), e => e.code === 'INVALID_INPUT_PATH')
  assert.deepEqual((await readdir(outside.root)).sort(), prior)
  assert.equal(f.state.postCount, 0)
})
