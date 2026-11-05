import assert from 'node:assert/strict'
import { readFile, readdir, realpath, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { test } from 'node:test'
import { cameraFixture } from './ego-camera-fixture.mjs'
import { fixture } from './rgbd-video-gen-fixture.mjs'

const gripperTarget = { frame: 'gripper', position: [0, 0, 0], look_at: [1, 0, 0], up: [0, 0, 1] }

test('default preparation reads only middle RGB even when side images are unavailable', async t => {
  const camera = await cameraFixture(t), saved = await camera.capture()
  const remote = await fixture(t, { inputRoot: camera.root })
  const manifest = JSON.parse(await readFile(saved.manifest_path, 'utf8'))
  manifest.views.head_camera_left.image_path = '../missing-left.jpg'
  manifest.views.head_camera_right.image_path = '../missing-right.jpg'
  await writeFile(saved.manifest_path, JSON.stringify(manifest))
  const prepared = await remote.call('prepare_scene', { capture_manifest_path: saved.manifest_path, targets: [gripperTarget] })
  assert.equal(prepared.num_source_views, 1)
  assert.equal(prepared.frame_count, 2)
  const scene = JSON.parse(await readFile(prepared.local_scene_path, 'utf8'))
  assert.deepEqual(scene.metadata.source_cameras, ['head_camera'])
  assert.equal(scene.frames.length, 2)
  assert.deepEqual(scene.frames[0].views.head_camera.camera_to_world, manifest.views.head_camera.camera_to_reference)
  assert.equal(scene.frames[1].views.head_camera.image, null)
  await remote.call('submit', { scene_path: prepared.scene_path, request_key: 'middle-only-001',
    num_source_views: prepared.num_source_views, max_target_views: prepared.max_target_views })
  const body = JSON.parse(remote.calls.find(call => call.method === 'POST').body)
  assert.equal(body.num_source_views, 1)
  assert.equal(body.scene.frames.length, 2)
})

test('general reference targets accept look-at and exact matrices without a gripper pose', async t => {
  const camera = await cameraFixture(t), saved = await camera.capture()
  const remote = await fixture(t, { inputRoot: camera.root })
  const manifest = JSON.parse(await readFile(saved.manifest_path, 'utf8'))
  delete manifest.gripper_pose
  delete manifest.views.head_camera.image_path
  await writeFile(saved.manifest_path, JSON.stringify(manifest))
  const matrix = [[0, -1, 0, 0.5], [1, 0, 0, 0.2], [0, 0, 1, -0.3], [0, 0, 0, 1]]
  const prepared = await remote.call('prepare_scene', { capture_manifest_path: saved.manifest_path,
    source_cameras: ['head_camera_left'], targets: [
      { frame: 'reference', position: [0.2, -0.3, 0.4], look_at: [0.2, -0.3, 1.4], up: [0, -1, 0], label: 'level_view' },
      { frame: 'reference', camera_to_frame: matrix },
    ] })
  const scene = JSON.parse(await readFile(prepared.local_scene_path, 'utf8'))
  assert.equal(prepared.max_target_views, 2)
  assert.equal(prepared.frame_count, 3)
  assert.deepEqual(scene.frames[0].views.head_camera.camera_to_world, manifest.views.head_camera_left.camera_to_reference)
  assert.deepEqual(scene.frames[1].views.head_camera.camera_to_world,
    [[1, 0, 0, 0.2], [0, 1, 0, -0.3], [0, 0, 1, 0.4], [0, 0, 0, 1]])
  assert.deepEqual(scene.frames[2].views.head_camera.camera_to_world, matrix)
  assert.equal(scene.metadata.targets[0].label, 'level_view')
})

test('invalid evidence selection and target descriptions never publish a scene', async t => {
  const camera = await cameraFixture(t), saved = await camera.capture()
  const remote = await fixture(t, { inputRoot: camera.root })
  for (const change of [
    { targets: [] }, { source_cameras: [] }, { source_cameras: ['head_camera', 'head_camera'] },
    { source_cameras: ['unknown'] },
    { targets: [{ ...gripperTarget, look_at: [0, 0, 0] }] },
    { targets: [{ ...gripperTarget, up: [1, 0, 0] }] },
    { targets: [{ frame: 'reference', camera_to_frame: [[1, 0], [0, 1]] }] },
    { targets: [{ ...gripperTarget, camera_to_frame: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] }] },
  ]) await assert.rejects(remote.call('prepare_scene', { capture_manifest_path: saved.manifest_path,
    targets: [gripperTarget], ...change }))
  await assert.rejects(readdir(join(camera.root, 'inputs', 'rgbd_video_gen', 'prepared')), { code: 'ENOENT' })
})

test('saved camera captures become four inline frames with a gripper-forward OpenCV target', async t => {
  const camera = await cameraFixture(t)
  const saved = await camera.capture()
  const remote = await fixture(t, { inputRoot: camera.root })
  const args = { capture_manifest_path: saved.manifest_path, source_cameras: [...camera.ids], targets: [gripperTarget] }
  const [one, two] = await Promise.all([remote.call('prepare_scene', args), remote.call('prepare_scene', args)])
  assert.equal(one.scene_path, two.scene_path)
  assert.equal(one.num_source_views, 3)
  assert.equal(one.max_target_views, 1)
  assert.equal(one.capture_stamp, saved.stamp)
  assert.equal(remote.state.postCount, 0)
  const scene = JSON.parse(await readFile(one.local_scene_path, 'utf8'))
  assert.equal(one.local_scene_path, join(await realpath(camera.root), one.scene_path))
  assert.equal(scene.frames.length, 4)
  for (const [i, id] of camera.ids.entries()) {
    const view = scene.frames[i].views.head_camera
    assert.deepEqual(Buffer.from(view.image.split(',')[1], 'base64'), camera.bytes)
    assert.deepEqual(view.camera_to_world, saved.views[id].camera_to_reference)
  }
  const target = scene.frames[3].views.head_camera
  assert.equal(target.image, null)
  assert.deepEqual(target.camera_to_world, [[0, 0, 1, 0], [-1, 0, 0, 0.5], [0, -1, 0, 0.7], [0, 0, 0, 1]])
  assert.deepEqual(scene.cameras.head_camera.K, saved.views.head_camera.K)
  await remote.call('submit', { scene_path: one.scene_path, request_key: 'prepared-gripper-001',
    num_source_views: one.num_source_views, max_target_views: one.max_target_views })
  const body = JSON.parse(remote.calls.find(c => c.method === 'POST').body)
  assert.deepEqual(body.scene, scene)
  assert.equal(body.num_source_views, 3)
  assert.equal(body.max_target_views, 1)
})

test('invalid or unsynchronized capture geometry never publishes a prepared scene', async t => {
  const camera = await cameraFixture(t), saved = await camera.capture()
  const remote = await fixture(t, { inputRoot: camera.root })
  const original = JSON.parse(await readFile(saved.manifest_path, 'utf8'))
  for (const change of [
    data => { data.gripper_pose.stamp += 1 },
    data => { data.gripper_pose.gripper_to_camera.head_camera_left[0][3] += 1 },
    data => { data.views.head_camera_left.K[0][0] += 1 },
    data => { data.views.head_camera.camera_to_reference[0][3] = 1 },
    data => { data.gripper_pose.gripper_to_camera.head_camera[0][0] = -1 },
    data => { delete data.gripper_pose },
  ]) {
    const data = structuredClone(original); change(data)
    await writeFile(saved.manifest_path, JSON.stringify(data))
    await assert.rejects(remote.call('prepare_scene', { capture_manifest_path: saved.manifest_path, source_cameras: [...camera.ids], targets: [gripperTarget] }), /INVALID_CAPTURE/)
  }
  await assert.rejects(readdir(join(camera.root, 'inputs', 'rgbd_video_gen', 'prepared')), { code: 'ENOENT' })
  assert.equal(remote.state.postCount, 0)
})

test('preparation rejects image paths outside inputRoot, oversized evidence and cancelled operations', async t => {
  const camera = await cameraFixture(t), saved = await camera.capture()
  const remote = await fixture(t, { inputRoot: camera.root })
  const original = JSON.parse(await readFile(saved.manifest_path, 'utf8'))
  const outside = join(remote.root, 'outside.jpg'); await writeFile(outside, camera.bytes)
  const data = structuredClone(original); data.views.head_camera_left.image_path = outside
  await writeFile(saved.manifest_path, JSON.stringify(data))
  await assert.rejects(remote.call('prepare_scene', { capture_manifest_path: saved.manifest_path, source_cameras: [...camera.ids], targets: [gripperTarget] }), /INVALID_INPUT_PATH/)
  await writeFile(saved.manifest_path, JSON.stringify(original))
  const small = await fixture(t, { inputRoot: camera.root, maxEvidenceImageBytes: 4 })
  await assert.rejects(small.call('prepare_scene', { capture_manifest_path: saved.manifest_path, source_cameras: [...camera.ids], targets: [gripperTarget] }), /INVALID_INPUT/)
  const limited = await fixture(t, { inputRoot: camera.root, maxSceneBytes: 100 })
  await assert.rejects(limited.call('prepare_scene', { capture_manifest_path: saved.manifest_path, source_cameras: [...camera.ids], targets: [gripperTarget] }), /REQUEST_TOO_LARGE/)
  await assert.rejects(remote.call('prepare_scene', { capture_manifest_path: saved.manifest_path, source_cameras: [...camera.ids], targets: [gripperTarget] },
    AbortSignal.abort(new Error('prepare cancelled'))), /cancelled/)
  await assert.rejects(readdir(join(camera.root, 'inputs', 'rgbd_video_gen', 'prepared')), { code: 'ENOENT' })
})
