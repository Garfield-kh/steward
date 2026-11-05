/** rgbd_video_gen HTTP provider with durable POST identities, bounded metadata and verified downloads. */
import type { Context } from '@deepseek-ai/cordis'
import schema from '@deepseek-ai/schemastery'
import { createHash, randomUUID } from 'node:crypto'
import { openAsBlob, realpathSync } from 'node:fs'
import { mkdir, open, realpath, rename, stat, unlink } from 'node:fs/promises'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { z } from 'zod'
import { RgbdVideoGenService, RgbdVideoGenError, jobId, type JobId, type GenerationOptions, type ScenePreparation,
  type Json, type JsonObject, type RecordingPreparation } from '@steward/dsh-rgbd-video-gen'
import type { RecordingId } from '@steward/dsh-ego'
import * as protocol from './protocol.ts'
import { digest, fileDigest, hasCode, inputDirectory, load, readBounded, reserve, save, saveImage, snapshotFile } from './storage.ts'
import { buildScene } from './scene.ts'
import { buildRecordingScene } from './recording.ts'

/** Keep server artifact identities separate from locally named source reconstructions and target predictions. */
function localArtifactName(filename: string, data: z.infer<typeof protocol.result>): string {
  const match = /^pred_(\d+)\.jpg$/.exec(filename)
  if (!match) return filename
  const sources = protocol.parse(z.number().int().min(1).max(8), data.params.num_source_views)
  const targets = protocol.parse(z.array(protocol.object), data.meta.target_cameras).length
  const frame = Number(match[1])
  if (!Number.isSafeInteger(frame) || frame >= sources + targets) {
    throw new RgbdVideoGenError('Prediction frame has no source or target camera.', 'INVALID_METADATA')
  }
  return frame < sources ? `source_reconstruction_${String(frame).padStart(2, '0')}.jpg`
    : `target_prediction_${String(frame - sources).padStart(2, '0')}.jpg`
}

/** Deployment settings; input/output directories resolve beneath inputRoot, other directories from the invocation directory. */
export interface Config {
  endpoint: string
  inputRoot: string
  stateDir: string
  inputsDir: string
  outputsDir: string
  requestTimeoutMs: number
  uploadTimeoutMs: number
  downloadTimeoutMs: number
  pollIntervalMs: number
  maxWaitSeconds: number
  retryCount: number
  retryDelayMs: number
  maxRetryDelayMs: number
  maxMetadataBytes: number
  maxUploadBytes: number
  maxSceneBytes: number
  maxArtifactBytes: number
  preparedSceneDir: string
  maxEvidenceImageBytes: number
}

const terminal = new Set(['succeeded', 'failed', 'cancelled'])
const jobFields = ['job_id', 'state', 'progress', 'error', 'params', 'scene', 'created_at', 'updated_at']

/** HTTP-backed rgbd_video_gen service. Plugin disposal stops local I/O; accepted remote jobs remain recoverable. */
export class RgbdVideoGenClient extends RgbdVideoGenService {
  static inject = ['ego']
  static Config = schema.object({
    endpoint: schema.string().required(), inputRoot: schema.string().required(), stateDir: schema.string().required(),
    inputsDir: schema.string().default('inputs/rgbd_video_gen'),
    outputsDir: schema.string().default('outputs/rgbd_video_gen'),
    requestTimeoutMs: schema.number().min(100).default(15_000),
    uploadTimeoutMs: schema.number().min(100).default(1_800_000),
    downloadTimeoutMs: schema.number().min(100).default(300_000),
    pollIntervalMs: schema.number().min(100).default(2_000),
    maxWaitSeconds: schema.number().min(0).max(60).default(20),
    retryCount: schema.number().min(0).max(5).default(2),
    retryDelayMs: schema.number().min(1).default(1_000),
    maxRetryDelayMs: schema.number().min(1).default(30_000),
    maxMetadataBytes: schema.number().min(1024).default(2_097_152),
    maxUploadBytes: schema.number().min(1).max(536_870_912).default(536_870_912),
    maxSceneBytes: schema.number().min(1).max(67_108_864).default(67_108_864),
    maxArtifactBytes: schema.number().min(1).default(2_147_483_648),
    preparedSceneDir: schema.string().default('prepared'),
    maxEvidenceImageBytes: schema.number().min(1).max(16_777_216).default(16_777_216),
  })

  private readonly base: string
  private readonly inputRoot: string
  private readonly inputs: string
  private readonly outputs: string
  private readonly root: string
  private readonly lifetime = new AbortController()
  private readonly pending = new Set<Promise<unknown>>()

  constructor(ctx: Context, readonly config: Config) {
    super(ctx)
    const url = new URL(config.endpoint)
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
      throw new RgbdVideoGenError('rgbd_video_gen endpoint must be HTTP(S), without credentials, query or fragment.', 'INVALID_CONFIG')
    }
    this.base = url.href.replace(/\/+$/, '')
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(config.preparedSceneDir)) {
      throw new RgbdVideoGenError('preparedSceneDir must be a single directory name beneath inputRoot.', 'INVALID_CONFIG')
    }
    this.inputRoot = realpathSync.native(resolve(config.inputRoot))
    this.inputs = resolve(this.inputRoot, config.inputsDir)
    this.outputs = resolve(this.inputRoot, config.outputsDir)
    this.root = resolve(config.stateDir, digest(this.base).slice(0, 16))
    ctx.effect(() => async () => {
      this.lifetime.abort()
      await Promise.allSettled([...this.pending])
    })
  }

  private async run<T>(signal: AbortSignal, body: (signal: AbortSignal) => Promise<T>): Promise<T> {
    const combined = AbortSignal.any([signal, this.lifetime.signal])
    combined.throwIfAborted()
    const task = body(combined)
    this.pending.add(task)
    try { return await task } finally { this.pending.delete(task) }
  }

  private async responseObject(response: Response): Promise<JsonObject> {
    const reader = response.body?.getReader()
    if (!reader) throw new RgbdVideoGenError('rgbd_video_gen returned an empty response body.', 'INVALID_METADATA')
    const chunks: Uint8Array[] = []
    let length = 0
    try {
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        length += value.byteLength
        if (length > this.config.maxMetadataBytes) throw new RgbdVideoGenError('rgbd_video_gen metadata exceeds maxMetadataBytes.', 'METADATA_TOO_LARGE')
        chunks.push(value)
      }
    } finally {
      try { await reader.cancel() } finally { reader.releaseLock() }
    }
    let value: unknown
    try { value = JSON.parse(Buffer.concat(chunks).toString('utf8')) } catch (error) {
      throw new RgbdVideoGenError(`rgbd_video_gen returned invalid JSON (HTTP ${response.status}): ${String(error)}`, 'INVALID_METADATA', false, response.status)
    }
    return protocol.parse(protocol.object, value)
  }

  private async json(path: string, request: () => Promise<RequestInit>, signal: AbortSignal,
    timeoutMs = this.config.requestTimeoutMs): Promise<JsonObject> {
    for (let attempt = 0; ; attempt++) {
      signal.throwIfAborted()
      let waitMs = this.config.retryDelayMs * 2 ** attempt
      const init = await request()
      try {
        const response = await fetch(this.base + path, { ...init, redirect: 'error',
          signal: AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]) })
        const body = await this.responseObject(response)
        if (response.ok) return body
        const remote = protocol.object.safeParse(body.error)
        const error = remote.success ? remote.data : {}
        const retryable = response.status === 429 || error.retryable === true
          && [409, 503, 507].includes(response.status)
        const retryAfter = response.headers.get('retry-after')
        if (retryAfter !== null) {
          const seconds = Number(retryAfter)
          const parsed = Number.isFinite(seconds) ? seconds * 1000 : Date.parse(retryAfter) - Date.now()
          if (Number.isFinite(parsed)) waitMs = Math.max(waitMs, parsed)
        }
        throw new RgbdVideoGenError(typeof error.message === 'string' ? error.message : `rgbd_video_gen HTTP ${response.status}`,
          typeof error.code === 'string' ? error.code : 'HTTP_ERROR', retryable, response.status,
          { request_id: body.request_id ?? null, details: error.details ?? null, retry_after_ms: waitMs })
      } catch (error) {
        signal.throwIfAborted()
        const failure = error instanceof RgbdVideoGenError ? error
          : new RgbdVideoGenError(`rgbd_video_gen request failed; retry the same request_key if submission is uncertain: ${String(error)}`, 'TRANSPORT', true)
        if (!failure.retryable || attempt >= this.config.retryCount || waitMs > this.config.maxRetryDelayMs) throw failure
      }
      await delay(waitMs, undefined, { signal })
    }
  }

  private get(path: string, signal: AbortSignal): Promise<JsonObject> {
    return this.json(path, async () => ({ method: 'GET' }), signal)
  }

  private validateWait(waitSeconds: number): void {
    if (!Number.isFinite(waitSeconds) || waitSeconds < 0 || waitSeconds > this.config.maxWaitSeconds) {
      throw new RgbdVideoGenError(`wait_seconds must be between 0 and ${this.config.maxWaitSeconds}.`, 'INVALID_ARGUMENT')
    }
  }

  private async poll(path: string, waitSeconds: number, isTerminal: (state: JsonObject) => boolean,
    signal: AbortSignal): Promise<JsonObject> {
    this.validateWait(waitSeconds)
    const deadline = Date.now() + waitSeconds * 1000
    for (;;) {
      const state = await this.get(path, signal)
      if (isTerminal(state) || Date.now() >= deadline) return state
      await delay(Math.min(this.config.pollIntervalMs, deadline - Date.now()), undefined, { signal })
    }
  }

  capabilities(signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => ({ ready: await this.get('/health/ready', signal),
      capabilities: await this.get('/v1/capabilities', signal) }))
  }

  resolveOptions(options: Partial<GenerationOptions>, signal: AbortSignal): Promise<GenerationOptions> {
    return this.run(signal, async signal => {
      const capabilities = await this.get('/v1/capabilities', signal)
      const defaults = protocol.parse(protocol.options, capabilities.defaults)
      return protocol.parse(protocol.options, { ...defaults,
        ...Object.fromEntries(Object.entries(options).filter(([, value]) => value !== undefined)) })
    })
  }

  private async input(path: string, extension: string, limit: number): Promise<string> {
    const actual = await realpath(resolve(this.inputRoot, path))
    const within = relative(this.inputRoot, actual)
    if (isAbsolute(within) || within === '..' || within.startsWith('..' + sep) || !actual.toLowerCase().endsWith(extension)) {
      throw new RgbdVideoGenError(`Input must be a ${extension} file beneath inputRoot.`, 'INVALID_INPUT_PATH')
    }
    const info = await stat(actual)
    if (!info.isFile() || info.size === 0 || info.size > limit) {
      throw new RgbdVideoGenError('Input is empty, not a file, or exceeds the configured size limit.', 'INVALID_INPUT')
    }
    return actual
  }

  prepareScene(captureManifest: string, options: ScenePreparation, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      const actual = await this.input(captureManifest, '.json', this.config.maxMetadataBytes)
      const bytes = await readBounded(actual, this.config.maxMetadataBytes, signal)
      let value: unknown
      try { value = JSON.parse(bytes.toString('utf8')) } catch (error) {
        throw new RgbdVideoGenError(`Capture manifest is not JSON: ${String(error)}`, 'INVALID_CAPTURE')
      }
      const { scene, stamp, sources, targets } = await buildScene(value, options, async path => {
        const image = await this.input(resolve(dirname(actual), path), '.jpg', this.config.maxEvidenceImageBytes)
        return readBounded(image, this.config.maxEvidenceImageBytes, signal)
      })
      const text = JSON.stringify(scene, null, 2) + '\n'
      if (Buffer.byteLength(text) > this.config.maxSceneBytes) {
        throw new RgbdVideoGenError('Prepared scene exceeds maxSceneBytes.', 'REQUEST_TOO_LARGE')
      }
      signal.throwIfAborted()
      const directory = join(this.inputs, this.config.preparedSceneDir)
      await mkdir(directory, { recursive: true })
      const actualDirectory = await realpath(directory)
      const within = relative(this.inputRoot, actualDirectory)
      if (isAbsolute(within) || within === '..' || within.startsWith('..' + sep)) {
        throw new RgbdVideoGenError('Prepared scene directory leaves inputRoot.', 'INVALID_INPUT_PATH')
      }
      const path = join(actualDirectory, `scene-${digest(text)}.json`)
      await save(path, scene)
      return { scene_path: relative(this.inputRoot, path).split(sep).join('/'), local_scene_path: path,
        capture_manifest_path: actual, capture_stamp: stamp,
        frame_count: sources + targets, num_source_views: sources, max_target_views: targets,
        prediction_label: 'novel_view_prediction',
        next_action: 'rgbd_video_gen_submit' }
    })
  }

  prepareRecording(id: RecordingId, options: RecordingPreparation, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      const raw = await this.ctx.ego.rgbRecordingEvidence(id, signal)
      if (Buffer.byteLength(JSON.stringify(raw)) > this.config.maxSceneBytes) {
        throw new RgbdVideoGenError('Recorded evidence exceeds maxSceneBytes.', 'REQUEST_TOO_LARGE')
      }
      const { data, decoded, indices, metadata, scene } = buildRecordingScene(raw, options)
      if (data.recording_id !== id) throw new RgbdVideoGenError('Recording identity differs from the requested handle.', 'INVALID_RECORDING')
      const text = JSON.stringify(scene, null, 2) + '\n'
      if (Buffer.byteLength(text) > this.config.maxSceneBytes || decoded.some(bytes => bytes.length > this.config.maxEvidenceImageBytes)) {
        throw new RgbdVideoGenError('Recording scene or image exceeds configured input limits.', 'REQUEST_TOO_LARGE')
      }
      const folder = join(this.inputs, this.config.preparedSceneDir, `${id}-${digest(JSON.stringify(data))}`)
      const actual = await inputDirectory(this.inputRoot, folder)
      const frames = []
      for (const [index, view] of data.frames.entries()) {
        signal.throwIfAborted()
        const imagePath = `${index.toString().padStart(6, '0')}.jpg`
        await saveImage(join(actual, imagePath), decoded[index], signal)
        const { image: _inline, ...sensor } = view
        frames.push({ ...sensor, image_path: imagePath, bytes: decoded[index].length })
      }
      const evidencePath = join(actual, 'recording.json')
      const { frames: _frames, ...descriptor } = data
      await save(evidencePath, { ...descriptor, frames })
      const scenePath = join(actual, `scene-${digest(text)}.json`)
      await save(scenePath, scene)
      signal.throwIfAborted()
      return { scene_path: relative(this.inputRoot, scenePath).split(sep).join('/'), local_scene_path: scenePath,
        recording_manifest_path: evidencePath, ...metadata, frame_count: indices.length + options.targets.length,
        num_source_views: indices.length, max_target_views: options.targets.length,
        prediction_label: 'novel_view_prediction', next_action: 'rgbd_video_gen_submit' }
    })
  }

  private async accept(kind: 'submit' | 'upload' | 'demo', requestKey: string, payload: JsonObject,
    path: string, request: () => Promise<RequestInit>, signal: AbortSignal, input: JsonObject): Promise<JsonObject> {
    protocol.parse(protocol.requestKey, requestKey)
    const record = await reserve(this.root, this.base, kind, requestKey, payload)
    let accepted: JsonObject | undefined
    try { accepted = await load(record.path + '.response.json') } catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
    if (!accepted) {
      accepted = await this.json(path, async () => {
        const init = await request()
        const headers = new Headers(init.headers)
        headers.set('Idempotency-Key', record.key)
        return { ...init, method: 'POST', headers }
      }, signal, this.config.uploadTimeoutMs)
    }
    const job = protocol.parse(protocol.jobState, accepted)
    const id = jobId(job.job_id)
    await save(record.path + '.response.json', accepted)
    await save(join(this.root, 'owned', `${id}.json`), { endpoint: this.base, request_record: record.path, input })
    return { ...protocol.pick(accepted, jobFields), request_key: requestKey, request_record: record.path,
      local_input: input,
      next_action: job.state === 'succeeded' ? 'rgbd_video_gen_result' : terminal.has(job.state) ? 'stop' : 'rgbd_video_gen_status' }
  }

  private async submitJson(kind: 'submit' | 'demo', body: JsonObject, requestKey: string,
    signal: AbortSignal): Promise<JsonObject> {
    const text = JSON.stringify(body)
    if (Buffer.byteLength(text) > this.config.maxSceneBytes) throw new RgbdVideoGenError('JSON request exceeds maxSceneBytes.', 'REQUEST_TOO_LARGE')
    const directory = join(this.inputs, `${kind}-${digest(text)}`)
    const requestPath = join(directory, 'request.json')
    const input: JsonObject = { request_path: requestPath, body_sha256: digest(text) }
    await save(requestPath, body)
    if (kind === 'submit') {
      const scenePath = join(directory, 'scene.json')
      input.scene_path = scenePath
      await save(scenePath, protocol.parse(protocol.object, body.scene))
    }
    return this.accept(kind, requestKey, { body_sha256: digest(text) },
      kind === 'demo' ? '/v1/nvs/demo' : '/v1/nvs/jobs', async () => ({
        headers: { 'Content-Type': 'application/json' }, body: text,
      }), signal, input)
  }

  submit(path: string, options: GenerationOptions, requestKey: string, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      const settings = protocol.parse(protocol.options, options)
      const actual = await this.input(path, '.json', this.config.maxSceneBytes)
      const file = await open(actual, 'r')
      let contents: string
      try {
        const chunks: Buffer[] = []
        let bytes = 0
        for await (const chunk of file.createReadStream({ autoClose: false, signal })) {
          bytes += chunk.length
          if (bytes > this.config.maxSceneBytes) throw new RgbdVideoGenError('Scene exceeds maxSceneBytes.', 'REQUEST_TOO_LARGE')
          chunks.push(chunk)
        }
        contents = Buffer.concat(chunks).toString('utf8')
      } finally { await file.close() }
      let decoded: unknown
      try { decoded = JSON.parse(contents) } catch (error) {
        throw new RgbdVideoGenError(`Scene file is not JSON: ${String(error)}`, 'INVALID_SCENE')
      }
      const scene = protocol.parse(protocol.object, decoded)
      if (scene.metadata && typeof scene.metadata === 'object' && !Array.isArray(scene.metadata)
        && typeof scene.metadata.recording_id === 'string'
        && (!Array.isArray(scene.metadata.source_frame_indices) || scene.metadata.source_frame_indices.length !== settings.num_source_views)) {
        throw new RgbdVideoGenError('Use the exact num_source_views returned by recording preparation.', 'INVALID_SCENE')
      }
      const checked = protocol.parse(z.object({ format: z.literal('rgbd_videogen_scene_v1'),
        reference_camera: z.string().min(1), cameras: protocol.object,
        frames: z.array(z.object({ views: z.record(z.string(), protocol.object) })).min(2).max(64),
      }), scene)
      if (!(checked.reference_camera in checked.cameras) || checked.frames.length <= settings.num_source_views) {
        throw new RgbdVideoGenError('Scene requires its reference camera and at least num_source_views + 1 frames. Set num_source_views explicitly for fewer than 6 evidence views.', 'INVALID_SCENE')
      }
      for (const [index, frame] of checked.frames.entries()) {
        const view = frame.views[checked.reference_camera]
        if (!view || index < settings.num_source_views && !view.image) throw new RgbdVideoGenError('Each source frame requires an image for the reference camera.', 'INVALID_SCENE')
        for (const view of Object.values(frame.views)) {
          if (view.image == null) continue
          if (typeof view.image !== 'string' || !/^(?:data:image\/(?:png|jpeg);base64,)?[A-Za-z0-9+/]+={0,2}$/.test(view.image)) {
            throw new RgbdVideoGenError('JSON images must be inline PNG/JPEG base64. Use rgbd_video_gen_upload for relative image paths.', 'INVALID_IMAGE')
          }
          if (!view.image.startsWith('data:')) {
            const header = Buffer.from(view.image.slice(0, 16), 'base64')
            const mime = header.subarray(0, 3).equals(Buffer.from([255, 216, 255])) ? 'jpeg'
              : header.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) ? 'png' : null
            if (!mime) throw new RgbdVideoGenError('Bare base64 must encode a PNG or JPEG image.', 'INVALID_IMAGE')
            // A bare JPEG starts with '/', which can also denote a server-local path.
            view.image = `data:image/${mime};base64,${view.image}`
          }
        }
      }
      scene.frames = protocol.parse(z.array(protocol.object), scene.frames).map((frame, index) => ({ ...frame, views: checked.frames[index].views }))
      return this.submitJson('submit', { scene, ...settings }, requestKey, signal)
    })
  }

  upload(path: string, requestKey: string, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      const actual = await this.input(path, '.zip', this.config.maxUploadBytes)
      const hash = await fileDigest(actual, signal)
      const payload = { archive_sha256: hash, filename: basename(actual) }
      const directory = join(this.inputs, `upload-${digest(JSON.stringify(payload))}`)
      const archive = join(directory, 'archive.zip'), requestPath = join(directory, 'request.json')
      await snapshotFile(actual, archive, hash, signal)
      await save(requestPath, payload)
      return this.accept('upload', requestKey, payload, '/v1/nvs/jobs/upload', async () => {
        if (await fileDigest(archive, signal) !== hash) throw new RgbdVideoGenError('Saved archive changed before upload.', 'INPUT_CHANGED')
        const form = new FormData()
        form.set('archive', await openAsBlob(archive, { type: 'application/zip' }), basename(actual))
        return { body: form }
      }, signal, { archive_path: archive, request_path: requestPath, archive_sha256: hash })
    })
  }

  demo(scene: string, options: GenerationOptions,
    target: { target_mode: 'trajectory' } | { target_mode: 'bev'; num_bev_views: number; bev_altitude_scale: number },
    requestKey: string, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      const settings = protocol.parse(protocol.options, options)
      protocol.parse(z.string().min(1).max(128), scene)
      protocol.parse(z.discriminatedUnion('target_mode', [z.object({ target_mode: z.literal('trajectory') }),
        z.object({ target_mode: z.literal('bev'), num_bev_views: z.number().int().min(1).max(24), bev_altitude_scale: z.number().positive() })]), target)
      return this.submitJson('demo', { scene: { __demo: scene }, ...settings, ...target }, requestKey, signal)
    })
  }

  status(id: JobId, waitSeconds: number, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      const state = await this.poll(`/v1/nvs/jobs/${id}`, waitSeconds, value => {
        const checked = protocol.parse(protocol.jobState, value)
        if (checked.job_id !== id) throw new RgbdVideoGenError('Status belongs to a different job.', 'INVALID_METADATA')
        return terminal.has(checked.state)
      }, signal)
      return { ...protocol.pick(state, jobFields),
        next_action: state.state === 'succeeded' ? 'rgbd_video_gen_result' : terminal.has(String(state.state)) ? 'stop' : 'rgbd_video_gen_status' }
    })
  }

  private async localManifest(id: JobId, value: JsonObject, signal: AbortSignal): Promise<void> {
    const directory = join(this.outputs, id), manifestPath = join(directory, 'manifest.json')
    let previous: JsonObject | undefined
    try { previous = await load(manifestPath) } catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
    if (previous && previous.endpoint !== this.base) {
      throw new RgbdVideoGenError('Output directory belongs to a different endpoint; configure a separate outputsDir.', 'LOCAL_OUTPUT_CONFLICT')
    }
    const data = protocol.parse(protocol.result, value), files: JsonObject[] = []
    for (const item of data.artifacts) {
      const localFilename = localArtifactName(item.filename, data)
      const path = join(directory, localFilename)
      try {
        if ((await stat(path)).size === item.bytes && await fileDigest(path, signal) === item.sha256) {
          files.push({ ...item, local_filename: localFilename, path, verified: true })
        }
      } catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
    }
    let input: Json = null
    try { input = (await load(join(this.root, 'owned', `${id}.json`))).input ?? null }
    catch (error) { if (!hasCode(error, 'ENOENT')) throw error }

    const params = protocol.parse(protocol.object, value.params)
    const sources = protocol.parse(z.number().int().min(1).max(8), params.num_source_views)
    const cameras = protocol.parse(z.array(protocol.object), data.meta.target_cameras)
    const targets = cameras.map((camera, index) => {
      const frame = sources + index, suffix = String(frame).padStart(2, '0')
      return { frame_index: frame, camera, rgb_path: files.find(file => file.filename === `pred_${suffix}.jpg`)?.path ?? null,
        depth_path: files.find(file => file.filename === 'depth.npz')?.path ?? null, depth_key: `v${suffix}` }
    })
    await save(join(directory, 'result.json'), value)
    await save(manifestPath, { version: 1, model: 'rgbd_video_gen', endpoint: this.base, job_id: id,
      input, local_result_path: join(directory, 'result.json'), files, targets })
  }

  private async manifest(id: JobId, signal: AbortSignal): Promise<z.infer<typeof protocol.result>> {
    const state = await this.status(id, 0, signal)
    if (state.state !== 'succeeded') throw new RgbdVideoGenError(`Job is ${state.state}; no successful result is available.`,
      terminal.has(String(state.state)) ? 'RESULT_UNAVAILABLE' : 'RESULT_NOT_READY')
    const value = await this.get(`/v1/nvs/jobs/${id}/result`, signal)
    const data = protocol.parse(protocol.result, value)
    if (data.job_id !== id || new Set(data.artifacts.map(a => a.filename.toLowerCase())).size !== data.artifacts.length) {
      throw new RgbdVideoGenError('Result has a different job id or colliding filenames.', 'INVALID_METADATA')
    }
    const localNames = data.artifacts.map(item => localArtifactName(item.filename, data).toLowerCase())
    if (new Set(localNames).size !== localNames.length) {
      throw new RgbdVideoGenError('Artifacts collide after local frame naming.', 'INVALID_METADATA')
    }
    for (const item of data.artifacts) {
      if (['manifest.json', 'result.json'].includes(item.filename.toLowerCase())) {
        throw new RgbdVideoGenError('Artifact filename collides with local result metadata.', 'INVALID_METADATA')
      }
      const expected = `/v1/nvs/jobs/${id}/artifacts/${item.filename}`
      // Reverse proxies may return either API-relative or already prefixed paths.
      const url = data.artifact_urls[item.filename]
      if (url !== expected && url !== new URL(this.base).pathname.replace(/\/$/, '') + expected) {
        throw new RgbdVideoGenError('Artifact URL does not belong to this job.', 'INVALID_METADATA')
      }
    }
    await this.localManifest(id, value, signal)
    return data
  }

  result(id: JobId, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      const data = await this.manifest(id, signal)
      return { ...data, artifacts: data.artifacts.map(item => ({ ...item, local_filename: localArtifactName(item.filename, data) })),
        completed: true, local_result_path: join(this.outputs, id, 'result.json'),
        local_manifest_path: join(this.outputs, id, 'manifest.json'),
        interpretation: 'RGB and depth are generated predictions, not observed geometry. Target PSNR is available only with target ground truth. Local source_reconstruction_*.jpg files reconstruct inputs; target_prediction_*.jpg files predict requested views, numbered from zero. Use artifacts[].filename for downloads and local_filename/path for local consumers. ZIP contents and depth keys retain server names.' }
    })
  }

  download(id: JobId, filenames: string[], signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      if (!filenames.length || new Set(filenames).size !== filenames.length) throw new RgbdVideoGenError('Select unique filenames from rgbd_video_gen_result.', 'INVALID_ARGUMENT')
      const data = await this.manifest(id, signal)
      const selected = filenames.map(name => {
        const item = data.artifacts.find(item => item.filename === name)
        if (!item) throw new RgbdVideoGenError(`Unknown result filename: ${name}`, 'UNKNOWN_ARTIFACT')
        if (item.bytes > this.config.maxArtifactBytes) throw new RgbdVideoGenError('Artifact exceeds maxArtifactBytes.', 'ARTIFACT_TOO_LARGE')
        return item
      })
      const files: Json[] = []
      const directory = join(this.outputs, id)
      await mkdir(directory, { recursive: true })
      for (const item of selected) {
        const localFilename = localArtifactName(item.filename, data)
        const target = join(directory, localFilename)
        let reused = false
        try { reused = (await stat(target)).size === item.bytes && await fileDigest(target, signal) === item.sha256 }
        catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
        if (!reused && localFilename !== item.filename) {
          const legacy = join(directory, item.filename)
          try {
            if ((await stat(legacy)).size === item.bytes && await fileDigest(legacy, signal) === item.sha256) {
              signal.throwIfAborted()
              await rename(legacy, target)
              reused = true
            }
          } catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
        }
        if (!reused) {
          const temporary = `${target}.${randomUUID()}.part`
          const response = await fetch(`${this.base}/v1/nvs/jobs/${id}/artifacts/${item.filename}`, { redirect: 'error',
            signal: AbortSignal.any([signal, AbortSignal.timeout(this.config.downloadTimeoutMs)]) })
          if (!response.ok || !response.body) {
            await response.body?.cancel()
            throw new RgbdVideoGenError(`Artifact download returned HTTP ${response.status}.`, 'DOWNLOAD_FAILED', false, response.status)
          }
          try {
            const reader = response.body.getReader()
            try {
              const file = await open(temporary, 'wx')
              try {
                const hash = createHash('sha256')
                let bytes = 0
                for (;;) {
                  const chunk = await reader.read()
                  if (chunk.done) break
                  signal.throwIfAborted()
                  bytes += chunk.value.byteLength
                  if (bytes > item.bytes) throw new RgbdVideoGenError('Artifact exceeds declared size.', 'INTEGRITY_FAILURE')
                  hash.update(chunk.value)
                  await file.writeFile(chunk.value)
                }
                if (bytes !== item.bytes || hash.digest('hex') !== item.sha256) throw new RgbdVideoGenError('Artifact size or SHA256 mismatch.', 'INTEGRITY_FAILURE')
                await file.sync()
              } finally { await file.close() }
            } finally { try { await reader.cancel() } finally { reader.releaseLock() } }
            await rename(temporary, target)
          } finally {
            try { await unlink(temporary) } catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
          }
        }
        files.push({ filename: item.filename, local_filename: localFilename, kind: item.kind, path: target, bytes: item.bytes,
          sha256: item.sha256, verified: true, reused })
      }
      await this.localManifest(id, data, signal)
      return { job_id: id, files, local_manifest_path: join(this.outputs, id, 'manifest.json') }
    })
  }

  cancel(id: JobId, waitSeconds: number, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      this.validateWait(waitSeconds)
      try { await load(join(this.root, 'owned', `${id}.json`)) } catch (error) {
        if (!hasCode(error, 'ENOENT')) throw error
        throw new RgbdVideoGenError('Only jobs submitted or recovered in this rgbd_video_gen store can be cancelled.', 'UNKNOWN_LOCAL_TASK')
      }
      const record = await reserve(this.root, this.base, 'cancel', `cancel:${id}`, { job_id: id })
      await this.json(`/v1/nvs/jobs/${id}/cancel`, async () => ({ method: 'POST',
        headers: { 'Idempotency-Key': record.key } }), signal)
      const state = await this.status(id, waitSeconds, signal)
      return { ...state, cancelled: state.state === 'cancelled' }
    })
  }
}

export default RgbdVideoGenClient
