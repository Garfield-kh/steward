// Real robot profile, recorded model calls and the built HTTP provider; replay needs no key.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createRequire } from 'node:module'
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { fixture } from './novelanything-fixture.mjs'
import { portableSession } from './portable-session.mjs'
import { launchProfile } from './profile-test.mjs'

const workspace = resolve(import.meta.dirname, '..')
const cliPackage = createRequire(import.meta.url).resolve('@deepseek-ai/dsh/package.json')
const prompt = 'This integration test uses synthetic RGB-only scene.zip and placeholder artifacts. The scene contains two different timestamped poses of the same middle RGB head_camera without any depth input, followed by one target pose. You are robot-1 in Ego Harness. Use the novelanything skill: load it, check health, upload the supplied workspace scene.zip with dataset scannetpp and request_key novel-snapshot-upload, submit its upload_id with context [0,1], target [2], metrics false and request_key novel-snapshot-submit, query status, read the successful result and download ALL returned artifacts. Use only skill and novelanything_* tools. No ego calls, alerts, movement, World tools, shell/files, subagents or demos. Report download completion, predictions rather than observations, and that no server checksum means only a local digest. Keep the final answer brief and omit absolute paths.'
const sideCameraPrompt = 'You are robot-1 in Ego Harness. Load the novelanything skill before deciding whether my proposed inputs are supported: I want to use head_camera_left and head_camera_right RGB from one timestamp as the two context frames without any depth input. Do not acquire or create other evidence, use demos, or call ego, shell, file, World or alert tools. If the skill does not support these inputs, explain which RGB camera is supported and stop before any upload or inference.'
const depthInputPrompt = 'You are robot-1 in Ego Harness. Load the novelanything skill before deciding whether my proposed inputs are supported. I have two posed RGB images from the middle head_camera and measured PNG depth for each; I want you to include depth_path in those context frames. Do not acquire, create, strip or replace evidence, use demos, or call ego, shell, file, World or alert tools. If this skill only supports RGB input, explain the supported camera and input modality and stop before upload or inference.'
const recordingPrompt = 'This is a synthetic robot integration fixture, not real hardware. The table-south destination is already supplied: x=0.5, y=-1.75, yaw=1.5708. Load the novelanything skill. I explicitly require NovelAnything, not rgbd_video_gen. Check NovelAnything health, then use ego_rgb_recording_start at 2 FPS, max_frames 3, request_key robot-record-snapshot. Predict navigation to the supplied destination with ego_predict, execute the receipt with skill_run wait=false, query skill_status, and stop the recording with ego_rgb_recording_stop. Prepare its succeeded recording with novelanything_prepare_gripper_view num_source_views=2: current gripper link origin, forward +X, image-up +Z, middle RGB intrinsics. Upload the returned archive_path with scannetpp and request_key novel-snapshot-upload. Submit its upload_id and the returned context/target with metrics=false and request_key novel-snapshot-submit. Query status, result and download ALL artifacts. Do not use any other tools, shell/files, World tools, subagents, demos or alerts. Briefly report gripper_view_prediction and distinguish model predictions from measurements; omit absolute paths.'

for (const mode of ['archive', 'side', 'depth', 'recording']) test(mode === 'side'
  ? 'robot NovelAnything skill rejects side-camera contexts before upload or inference'
  : mode === 'depth' ? 'robot NovelAnything skill rejects depth inputs even from the supported middle camera'
  : mode === 'recording' ? 'robot agent composes independent recording, navigation and NovelAnything tools without model substitution'
  : 'robot profile loads NovelAnything skill and replays RGB-only upload, inference and download', { timeout: 240000 }, async t => {
  const sideCameras = mode === 'side', depthInputs = mode === 'depth', unsupported = sideCameras || depthInputs, recorded = mode === 'recording'
  const f = await fixture(t); f.state.status = 'succeeded'; f.state.checksum = false
  const snapshot = join(import.meta.dirname, sideCameras ? 'snapshots/novelanything-center-only'
    : depthInputs ? 'snapshots/novelanything-rgb-only' : recorded ? 'snapshots/novelanything-recording' : 'snapshots/novelanything')
  const recording = process.env.NOVELANYTHING_RECORD === '1'
  const env = { ...process.env, DSH_HOME: join(f.root, 'home'), DSH_SNAPSHOT_FILE: join(snapshot, 'session.v4.jsonl') }
  if (!recording) {
    delete env.DEEPSEEK_API_KEY; delete env.DEEPSEEK_BASE_URL
    // The model may echo absolute archive paths returned by preparation; bind their recorded cwd to this fixture.
    env.DSH_SNAPSHOT_FILE = join(f.root, 'replay.session.v4.jsonl')
    await writeFile(env.DSH_SNAPSHOT_FILE, (await readFile(join(snapshot, 'session.v4.jsonl'), 'utf8'))
      .replaceAll('{{cwd}}', f.root.replaceAll('\\', '/'))
      .replaceAll('{{endpoint}}', f.config.endpoint).replaceAll('{{dshWorkspace}}', workspace.replaceAll('\\', '/')))
  }
  const launch = args => launchProfile(args, f.root, env)
  await launch(['--profile', 'ego', '--from-default-profile', 'headless', '--dump-default-config'])
  await launch(['plugin', '--profile', 'ego', 'add', join(workspace, 'packages/dsh-bundle-ego')])
  await launch(['plugin', '--profile', 'ego', 'add', join(workspace, 'packages/dsh-bundle-novelanything')])
  if (recorded) await launch(['plugin', '--profile', 'ego', 'add', join(workspace, 'packages/dsh-bundle-rgbd-video-gen')])
  const patch = [
    { id: 'tools', config: { mode: 'native' } }, // These recorded sessions exercise native calls; the shipped Ego profile keeps PTC.
    { id: 'session-title-llm', disabled: true },
    { id: 'steward-ego-runtime', config: { endpoint: f.config.endpoint, timeoutMs: 1000 } },
    { id: 'agent-default-model', config: { provider: 'deepseek-official', model: 'deepseek-flash' } },
    { id: 'steward-novelanything-api', config: { endpoint: f.config.endpoint, inputRoot: '.', stateDir: 'state' } },
    { id: 'session-persistence-jsonl', config: { root: './sessions', compression: 'none' } },
    { id: 'session-log-deepseek', config: { enabled: false } },
  ]
  if (!recording) patch.push({ id: 'llm-deepseek', disabled: true }, { id: 'llm-deepseek-account', disabled: true },
    { insert: [{ id: 'llm-replay', name: pathToFileURL(createRequire(cliPackage).resolve('@deepseek-ai/dsh-llm-replay')).href,
      config: { providers: [{ id: 'deepseek-official', models: [{ id: 'deepseek-flash', contextWindow: 128000 }] }] } }] })
  const patchPath = join(f.root, 'snapshot.patch.json'); await writeFile(patchPath, JSON.stringify(patch))
  const output = await launch(['--profile', 'ego', '--patch', patchPath, '--json', sideCameras ? sideCameraPrompt : depthInputs ? depthInputPrompt : recorded ? recordingPrompt : prompt])
  const events = output.trim().split('\n').filter(line => line.startsWith('{')).map(line => JSON.parse(line))
  const calls = events.filter(e => e.type === 'tool_call')
  assert.deepEqual(calls.map(e => e.tool), unsupported ? ['skill'] : recorded
    ? ['skill', 'novelanything_health', 'ego_rgb_recording_start', 'ego_predict', 'skill_run', 'skill_status',
      'ego_rgb_recording_stop', 'novelanything_prepare_gripper_view', 'novelanything_upload', 'novelanything_submit',
      'novelanything_status', 'novelanything_result', 'novelanything_download']
    : ['skill', 'novelanything_health', 'novelanything_upload', 'novelanything_submit',
    'novelanything_status', 'novelanything_result', 'novelanything_download'])
  assert.ok(events.some(e => e.type === 'final'))
  const results = events.filter(e => e.type === 'tool_result')
  assert.equal(results.length, calls.length)
  assert.ok(results.every(e => e.status === 'completed'), JSON.stringify(results.filter(e => e.status !== 'completed')))
  const names = new Map(calls.map(e => [e.callId, e.tool]))
  const values = results.map(e => names.get(e.callId) === 'skill' ? { skill: e.result } : JSON.parse(e.result))
  if (unsupported) {
    assert.equal(f.calls.length, 0, 'Unsupported camera sources must cause no remote request')
    const final = events.find(e => e.type === 'final')
    assert.match(final.text, /head_camera/)
    values.push({ final: final.text })
  } else {
    const downloaded = values.at(-1)
    assert.equal(downloaded.files.length, 3)
    for (const file of downloaded.files) {
      assert.deepEqual(await readFile(file.path), f.bytes)
      assert.equal(file.server_checksum_verified, false)
    }
    const manifest = JSON.parse(await readFile(downloaded.local_manifest_path, 'utf8'))
    assert.ok(manifest.input.local_input.archive_path)
    assert.equal(f.calls.filter(c => c.method === 'POST' && c.path.startsWith('/novelanything/api/')).length, 2)
    assert.ok(f.calls.every(c => c.path.startsWith('/novelanything/api/') || recorded && (c.path.startsWith('/novelanything/ego/') || c.path.startsWith('/novelanything/skill/'))))
    if (recorded) {
      assert.equal(manifest.input.request.context.length, 2)
      assert.ok(f.calls.some(c => c.path.endsWith('/prepare-novelanything')))
      assert.ok(!calls.some(c => c.tool.startsWith('rgbd_video_gen_')))
      const prepared = values[7]
      assert.deepEqual(prepared.metadata.input_modalities, ['rgb'])
      assert.equal(prepared.metadata.fps, 2)
      assert.equal(prepared.label, 'gripper_view_prediction')
      assert.deepEqual(await readFile(prepared.archive_path), Buffer.from(f.prepared.archive_base64, 'base64'))
    }
  }
  const logs = await readdir(join(f.root, 'sessions'), { recursive: true })
  const raw = await readFile(join(f.root, 'sessions', logs.find(p => p.endsWith('session.v4.jsonl'))), 'utf8')
  const rows = raw.trim().split('\n').map(line => JSON.parse(line))
  const header = rows.find(e => e.type === 'request/header').data.header
  const toolNames = header.tools.map(tool => tool.name)
  assert.ok(toolNames.includes('ego_predict') && toolNames.includes('skill_run'))
  assert.ok(toolNames.includes('novelanything_submit'))
  if (recorded) assert.ok(toolNames.includes('rgbd_video_gen_prepare_scene'), 'Alternative model is present while the agent selects the requested model')
  assert.ok(!toolNames.some(name => /^(world_|bash$|pwsh$|fs$|fs_search$)/.test(name)))
  assert.ok(rows.some(e => e.type === 'user/message' && e.data.source.kind === 'skill-catalog'
    && e.data.source.entries.some(skill => skill.name === 'novelanything')))
  function portable(value) {
    if (Array.isArray(value)) return value.map(portable)
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, portable(item)]))
    if (typeof value !== 'string') return value
    return value.replaceAll(f.config.endpoint, '{{endpoint}}').replaceAll(f.root, '{{cwd}}').replaceAll('\\', '/')
      .replaceAll(workspace.replaceAll('\\', '/'), '{{dshWorkspace}}').replace(/(state\/)[a-f0-9]{16}(\/)/g, '$1{{service}}$2')
  }
  const transcript = JSON.stringify(values.map(portable), null, 2) + '\n'
  if (recording) {
    await mkdir(snapshot, { recursive: true })
    await writeFile(join(snapshot, 'session.v4.jsonl'), portableSession(raw, f.root, f.config.endpoint)
      .replaceAll(workspace.replaceAll('\\', '/'), '{{dshWorkspace}}'))
    await writeFile(join(snapshot, 'tools.expected.json'), transcript)
  } else assert.equal(transcript, await readFile(join(snapshot, 'tools.expected.json'), 'utf8'))

  // Installation policy must also leave the two World profiles without this capability.
  for (const profile of ['world', 'world-task']) {
    await launch(['--profile', profile, '--from-default-profile', profile === 'world' ? 'web' : 'headless', '--dump-default-config'])
    await launch(['plugin', '--profile', profile, 'add', join(workspace, 'packages/dsh-bundle-world')])
    const composed = await launch(['--profile', profile, '--dump-config'])
    assert.doesNotMatch(composed, /steward-novelanything/)
  }
})
