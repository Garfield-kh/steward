// A shipped headless profile replays a recorded model against the real provider and HTTP fixture.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createRequire } from 'node:module'
import { copyFile, mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { egoTools } from '@steward/dsh-tool-ego'
import { fixture } from './rgbd-video-gen-fixture.mjs'
import { cameraFixture } from './ego-camera-fixture.mjs'
import { portableSession } from './portable-session.mjs'
import { launchProfile as launch } from './profile-test.mjs'

const workspace = resolve(import.meta.dirname, '..')
const cliPackage = createRequire(import.meta.url).resolve('@deepseek-ai/dsh/package.json')
const robotPrompt = 'This is an integration test with synthetic capture data and tiny placeholder artifacts; verifying transfer is the expected outcome. You are robot-1 in the Ego Harness. Use the rgbd-video-gen skill to generate one gripper-view prediction from the already supplied capture.json in your workspace. Its exact capture_manifest_path is capture.json, relative to inputRoot; use that path directly without guessing another directory. It includes three saved camera images and the synchronized gripper pose. Load the skill, check capabilities, prepare the scene with the general tool, using its default middle RGB evidence and targets=[{frame:gripper,position:[0,0,0],look_at:[1,0,0],up:[0,0,1]}], submit the returned scene_path with request_key robot-rgbd-snapshot, num_source_views=1, max_target_views=1 and 30 inference steps, query status, read the successful result and download ALL returned artifacts. All robot evidence is already supplied. Do not call any ego_* tool, body_stop or report_problem; do not capture again, move the robot, use World or filesystem tools, delegate subagents or substitute a demo. Briefly report verified local paths and distinguish predictions from observations. Do not print absolute paths in the final answer.'

const temporalPrompt = 'This is a synthetic integration test, not real hardware. You are robot-1 in Ego Harness. The complete motion was already recorded at 2 FPS: recording_id=rgb_cccccccccccccccccccccccccccccccc, 37 middle head_camera RGB frames covering 0 to 18 seconds. Use the rgbd-video-gen skill to predict the CURRENT gripper origin view, forward +X, image-up +Z. Explicitly select 8 uniformly spaced frames covering the whole route; I understand these are sampled from the complete recording. Load the skill, check capabilities, prepare the RECORDING with those targets, submit its returned scene_path and exact returned source/target counts with request_key robot-rgbd-snapshot and 30 inference steps, then query status, result and download ALL returned artifacts. Never replace recorded evidence with a single snapshot, a capture manifest or a demo. Do not navigate, record again, call ego_capture_cameras, use shell/files, World, subagents or alerts. Report total recorded frames versus actually selected source frames and prediction status, omitting absolute paths.'

for (const temporal of [false, true]) test(temporal ? 'robot uses temporal evidence spanning a completed motion recording instead of a final snapshot'
  : 'robot Ego profile discovers the skill and submits novel-view synthesis without a key', { timeout: 240000 }, async t => {
  const f = await fixture(t); f.state.job = 'succeeded'; f.state.recordingEnabled = temporal
  const snapshot = join(import.meta.dirname, temporal ? 'snapshots/rgbd-video-gen-recording' : 'snapshots/rgbd-video-gen')
  const recording = process.env.RGBD_VIDEO_GEN_RECORD === '1'
  const profile = 'ego'
  const cwd = join(f.root, 'robot-workspace')
  await mkdir(cwd)
  const camera = await cameraFixture(t, { cameraCaptureDir: join(cwd, 'captures') })
  const captured = await camera.capture()
  await copyFile(captured.manifest_path, join(cwd, 'capture.json'))
  const env = { ...process.env, DSH_HOME: join(f.root, 'home'), DSH_SNAPSHOT_FILE: join(snapshot, 'session.v4.jsonl') }
  if (!recording) { delete env.DEEPSEEK_API_KEY; delete env.DEEPSEEK_BASE_URL }
  await launch(['--profile', profile, '--from-default-profile', 'headless', '--dump-default-config'], cwd, env)
  await launch(['plugin', '--profile', profile, 'add', join(workspace, 'packages/dsh-bundle-ego')], cwd, env)
  await launch(['plugin', '--profile', profile, 'add', join(workspace, 'packages/dsh-bundle-rgbd-video-gen')], cwd, env)
  const patch = [
    { id: 'tools', config: { mode: 'native' } }, // Replay the fixture's native calls without changing the shipped PTC profile.
    { id: 'session-title-llm', disabled: true },
    { id: 'steward-ego-runtime', config: { endpoint: f.egoEndpoint, timeoutMs: 5000 } },
    { id: 'agent-default-model', config: { provider: 'deepseek-official', model: 'deepseek-flash' } },
    { id: 'steward-rgbd-video-gen-api', config: { endpoint: f.config.endpoint, inputRoot: '.', stateDir: 'state' } },
    { id: 'session-persistence-jsonl', config: { root: './sessions', compression: 'none' } },
    { id: 'session-log-deepseek', config: { enabled: false } },
  ]
  if (!recording) patch.push({ id: 'llm-deepseek', disabled: true }, { id: 'llm-deepseek-account', disabled: true },
    { insert: [{ id: 'llm-replay', name: pathToFileURL(createRequire(cliPackage).resolve('@deepseek-ai/dsh-llm-replay')).href,
      config: { providers: [{ id: 'deepseek-official', models: [{ id: 'deepseek-flash', contextWindow: 128000 }] }] } }] })
  const patchPath = join(f.root, 'snapshot.patch.json')
  await writeFile(patchPath, JSON.stringify(patch))
  const composed = await launch(['--profile', profile, '--patch', patchPath, '--dump-config'], cwd, env)
  const egoRow = composed.replaceAll('\r\n', '\n').split('- id: steward-ego-runtime\n')[1]?.split('\n- ')[0]
  assert.ok(egoRow?.includes(`endpoint: ${f.egoEndpoint}\n`), 'Ego API must use the owned HTTP fixture before starting the model')
  const report = egoTools(f.ctx.ego).find(tool => tool.name === 'report_problem')
  const args = { goal: 'Test isolation', step: 'Report a fixture problem', diagnosis: 'Synthetic regression probe', message: 'Fixture alert' }
  await assert.rejects(report.execute(args, { name: report.name, arguments: args, signal: AbortSignal.timeout(5000) }),
    /Ego API disabled in RGBD synthesis fixture/)
  assert.equal(f.egoCalls.length, 1)
  assert.equal(f.egoCalls[0].path, '/ego/alert')
  assert.equal(JSON.parse(f.egoCalls[0].body).message, args.message)
  const output = await launch(['--profile', profile, '--patch', patchPath, '--json', temporal ? temporalPrompt : robotPrompt], cwd, env)
  assert.deepEqual(f.egoCalls.map(c => c.path), temporal ? ['/ego/alert', '/ego/rgb/record/export'] : ['/ego/alert'])
  const events = output.trim().split('\n').filter(line => line.startsWith('{')).map(line => JSON.parse(line))
  assert.ok(events.some(e => e.type === 'final'))
  const results = events.filter(e => e.type === 'tool_result')
  assert.equal(results.length, 7, output)
  assert.ok(results.every(e => e.status === 'completed'), output)
  const calls = events.filter(e => e.type === 'tool_call')
  const names = new Map(calls.map(e => [e.callId, e.tool]))
  const values = results.map(e => names.get(e.callId) === 'skill' ? { skill: e.result } : JSON.parse(e.result))
  const tail = values.slice(-3)
  assert.equal(tail[0].next_action, 'rgbd_video_gen_result')
  assert.equal(tail[1].completed, true)
  assert.equal(tail[1].meta.has_target_gt, false)
  assert.equal(tail[2].files.length, 4)
  for (const file of tail[2].files) assert.deepEqual(await readFile(file.path), f.bytes)
  assert.equal(f.state.postCount, 1)
  const logs = await readdir(join(cwd, 'sessions'), { recursive: true })
  const log = logs.find(p => p.endsWith('session.v4.jsonl'))
  assert.ok(log)
  const rawSession = await readFile(join(cwd, 'sessions', log), 'utf8')
  const sessionRoot = dirname(JSON.parse(rawSession.split('\n')[0]).cwd)
  {
    assert.deepEqual(calls.map(e => e.tool), ['skill', 'rgbd_video_gen_capabilities', temporal ? 'rgbd_video_gen_prepare_recording' : 'rgbd_video_gen_prepare_scene', 'rgbd_video_gen_submit',
      'rgbd_video_gen_status', 'rgbd_video_gen_result', 'rgbd_video_gen_download'])
    assert.match(values[0].skill, /rgbd-video-gen/)
    const rows = rawSession.trim().split('\n').map(line => JSON.parse(line))
    const header = rows.find(e => e.type === 'request/header').data.header
    const toolNames = header.tools.map(tool => tool.name)
    assert.ok(toolNames.includes('ego_predict') && toolNames.includes('skill_run'))
    assert.ok(toolNames.includes('rgbd_video_gen_submit'))
    assert.ok(!toolNames.some(name => name.startsWith('slarm_')))
    assert.ok(!toolNames.some(name => /^(world_|bash$|pwsh$|fs$|fs_search$)/.test(name)))
    const catalogs = rows.filter(e => e.type === 'user/message' && e.data.source.kind === 'skill-catalog')
    assert.ok(catalogs.some(e => e.data.source.entries.some(skill => skill.name === 'rgbd-video-gen')))
    const submission = JSON.parse(f.calls.find(c => c.path.endsWith('/v1/nvs/jobs')).body)
    assert.equal(submission.scene.format, 'rgbd_videogen_scene_v1')
    assert.equal(submission.num_source_views, temporal ? 8 : 1)
    assert.equal(submission.max_target_views, 1)
    assert.equal(submission.scene.frames.length, temporal ? 9 : 2)
    if (temporal) {
      assert.equal(submission.scene.metadata.recorded_frames, 37)
      assert.equal(submission.scene.metadata.all_frames_used, false)
      assert.deepEqual(submission.scene.metadata.source_frame_indices, [0, 5, 10, 15, 21, 26, 31, 36])
      const preparation = values[2]
      const recorded = JSON.parse(await readFile(preparation.recording_manifest_path, 'utf8'))
      assert.equal(recorded.frames.length, 37)
      for (const [i, index] of submission.scene.metadata.source_frame_indices.entries()) {
        assert.equal(submission.scene.frames[i].views.head_camera.image, f.evidence.frames[index].image)
        assert.deepEqual(submission.scene.frames[i].views.head_camera.camera_to_world, f.evidence.frames[index].camera_to_reference)
      }
    } else assert.deepEqual(submission.scene.metadata.source_cameras, ['head_camera'])
    assert.equal(submission.scene.metadata.targets[0].frame, 'gripper')
    if (!temporal) assert.equal(submission.scene.metadata.capture_stamp, captured.stamp)
  }
  function portable(value) {
    if (Array.isArray(value)) return value.map(portable)
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, portable(item)]))
    if (typeof value !== 'string') return value
    return value.replaceAll(f.config.endpoint, '{{endpoint}}').replaceAll(f.root, '{{cwd}}').replaceAll(sessionRoot, '{{cwd}}').replaceAll('\\', '/')
      .replaceAll(workspace.replaceAll('\\', '/'), '{{dshWorkspace}}')
      .replace(/(state\/)[a-f0-9]{16}(\/)/g, '$1{{service}}$2')
  }
  const transcript = JSON.stringify(values.map(portable), null, 2) + '\n'
  if (recording) {
    await mkdir(snapshot, { recursive: true })
    await writeFile(join(snapshot, 'session.v4.jsonl'), portableSession(rawSession, f.root, f.config.endpoint)
      .replaceAll(sessionRoot.replaceAll('\\', '//'), '{{cwd}}')
      .replaceAll(sessionRoot.replaceAll('\\', '/'), '{{cwd}}')
      .replaceAll(workspace.replaceAll('\\', '/'), '{{dshWorkspace}}'))
    await writeFile(join(snapshot, 'tools.expected.json'), transcript)
  } else if (process.env.UPDATE_TOOL_SNAPSHOTS === '1') {
    await writeFile(join(snapshot, 'tools.expected.json'), transcript)
  } else assert.equal(transcript, await readFile(join(snapshot, 'tools.expected.json'), 'utf8'))
})
