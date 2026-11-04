/** NovelAnything HTTP provider with local POST recovery and confined artifact downloads. */
import type { Context } from '@deepseek-ai/cordis'
import schema from '@deepseek-ai/schemastery'
import { createHash, randomUUID } from 'node:crypto'
import { createReadStream, realpathSync } from 'node:fs'
import { mkdir, readFile, realpath, stat, open, rename, unlink } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { NovelAnythingService, NovelAnythingError, remoteId, type RemoteId, type JsonObject,
  type Submission } from '@steward/dsh-novelanything'
import * as protocol from './protocol.ts'
import { body, contained, digest, directory, hasCode, publish, save } from './storage.ts'
import type { RecordingId } from '@steward/dsh-ego'

/** Deployment settings; all local directories resolve beneath the robot inputRoot. */
export interface Config {
  endpoint: string
  tokenEnv: string
  inputRoot: string
  stateDir: string
  inputsDir: string
  outputsDir: string
  requestTimeoutMs: number
  transferTimeoutMs: number
  pollIntervalMs: number
  maxWaitSeconds: number
  maxMetadataBytes: number
  maxUploadBytes: number
  maxArtifactBytes: number
}

const terminal = new Set(['succeeded', 'failed', 'cancelled', 'interrupted'])

/** Accepted remote work survives local cancellation, timeout and provider disposal. */
export class NovelAnythingClient extends NovelAnythingService {
  static inject = ['ego']
  static Config = schema.object({
    endpoint: schema.string().required(), inputRoot: schema.string().required(),
    tokenEnv: schema.string().default('NOVELANYTHING_API_KEY'),
    stateDir: schema.string().default('var/novelanything'),
    inputsDir: schema.string().default('inputs/novelanything'),
    outputsDir: schema.string().default('outputs/novelanything'),
    requestTimeoutMs: schema.number().min(100).default(15_000),
    transferTimeoutMs: schema.number().min(100).default(300_000),
    pollIntervalMs: schema.number().min(100).default(2_000),
    maxWaitSeconds: schema.number().min(0).max(60).default(20),
    maxMetadataBytes: schema.number().min(1024).default(2_097_152),
    maxUploadBytes: schema.number().min(1).max(268_435_456).default(268_435_456),
    maxArtifactBytes: schema.number().min(1).default(2_147_483_648),
  })
  private readonly base: string
  private readonly root: string
  private readonly state: string
  private readonly inputs: string
  private readonly outputs: string
  private readonly token: string
  private readonly lifetime = new AbortController()
  private readonly pending = new Set<Promise<unknown>>()

  constructor(ctx: Context, readonly config: Config) {
    super(ctx)
    const url = new URL(config.endpoint)
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
      throw new NovelAnythingError('Endpoint must use HTTP(S) without credentials, query or fragment.', 'INVALID_CONFIG')
    }
    this.base = url.href.replace(/\/+$/, '') + '/'
    this.token = process.env[config.tokenEnv] ?? ''
    this.root = realpathSync.native(resolve(config.inputRoot))
    this.state = resolve(this.root, config.stateDir, digest(this.base).slice(0, 16))
    this.inputs = resolve(this.root, config.inputsDir)
    this.outputs = resolve(this.root, config.outputsDir)
    for (const path of [this.state, this.inputs, this.outputs]) contained(this.root, path)
    if (new Set([this.state, this.inputs, this.outputs]).size !== 3) {
      throw new NovelAnythingError('State, input and output directories must differ.', 'INVALID_CONFIG')
    }
    ctx.effect(() => async () => {
      this.lifetime.abort(new NovelAnythingError('NovelAnything provider disposed; remote jobs continue.', 'DISPOSED'))
      await Promise.allSettled([...this.pending])
    })
  }

  private async run<T>(signal: AbortSignal, work: (signal: AbortSignal) => Promise<T>): Promise<T> {
    const combined = AbortSignal.any([signal, this.lifetime.signal])
    combined.throwIfAborted()
    const task = work(combined)
    this.pending.add(task)
    try { return await task } finally { this.pending.delete(task) }
  }

  private async request<T>(path: string, init: RequestInit, signal: AbortSignal,
    consume: (response: Response) => Promise<T>, timeout = this.config.requestTimeoutMs): Promise<T> {
    const headers = new Headers(init.headers)
    headers.set('Accept', 'application/json')
    if (this.token) headers.set('Authorization', `Bearer ${this.token}`)
    try {
      const response = await fetch(new URL(path, this.base), { ...init, headers, redirect: 'error',
        signal: AbortSignal.any([signal, AbortSignal.timeout(timeout)]) })
      if (!response.ok) {
        const detail = (await body(response, this.config.maxMetadataBytes)).toString('utf8')
        throw new NovelAnythingError(`NovelAnything HTTP ${response.status}: ${detail}`, 'HTTP_ERROR', response.status)
      }
      return await consume(response)
    } catch (error) {
      if (error instanceof NovelAnythingError) throw error
      if (signal.aborted) throw new NovelAnythingError('Local request stopped; remote jobs may continue.', 'ABORTED', null, { cause: error })
      throw new NovelAnythingError('NovelAnything transfer failed; inspect the retained job before continuing.', 'TRANSPORT', null, { cause: error })
    }
  }

  private json(path: string, signal: AbortSignal, init: RequestInit = {}): Promise<JsonObject> {
    return this.request(path, init, signal, async response => {
      const bytes = await body(response, this.config.maxMetadataBytes)
      let value: unknown
      try { value = JSON.parse(bytes.toString('utf8')) } catch (error) {
        throw new NovelAnythingError('NovelAnything returned malformed JSON.', 'INVALID_DATA', response.status, { cause: error })
      }
      return protocol.parse(protocol.object, value)
    }, init.method === 'POST' ? this.config.transferTimeoutMs : this.config.requestTimeoutMs)
  }

  private async load(path: string): Promise<JsonObject> {
    if ((await stat(path)).size > this.config.maxMetadataBytes) {
      throw new NovelAnythingError('Saved NovelAnything metadata exceeds maxMetadataBytes.', 'INVALID_LOCAL_STATE')
    }
    const text = await readFile(path, 'utf8')
    let value: unknown
    try { value = JSON.parse(text) } catch (error) {
      throw new NovelAnythingError('Saved NovelAnything metadata is malformed.', 'INVALID_LOCAL_STATE', null, { cause: error })
    }
    return protocol.parse(protocol.object, value)
  }

  private async post(kind: 'upload' | 'submit', key: string, payload: JsonObject, init: RequestInit,
    signal: AbortSignal, validate: (response: JsonObject) => void): Promise<JsonObject> {
    if (!/^[A-Za-z0-9._:-]{8,128}$/.test(key)) throw new NovelAnythingError('Use an 8–128 character request_key.', 'INVALID_KEY')
    await directory(this.root, this.state)
    const path = join(this.state, `${kind}-${digest(key)}.json`)
    const fingerprint = digest(JSON.stringify(payload))
    try { await save(path, { version: 1, fingerprint, payload }, true) } catch (error) {
      if (!hasCode(error, 'EEXIST')) throw error
      const prior = protocol.parse(protocol.receipt, await this.load(path))
      if (prior.fingerprint !== fingerprint) throw new NovelAnythingError('This request_key belongs to different inputs/options.', 'KEY_CONFLICT')
      if (prior.response) { validate(prior.response); return prior.response }
      throw new NovelAnythingError('Previous POST has an unknown outcome. Reconcile server jobs/uploads with the operator; do not submit with a new key.', 'UNKNOWN_OUTCOME')
    }
    try {
      const response = await this.json(kind === 'upload' ? 'api/uploads' : 'api/jobs', signal, init)
      validate(response)
      await save(path, { version: 1, fingerprint, payload, response })
      return response
    } catch (error) {
      throw new NovelAnythingError('POST was attempted but no accepted reply was saved. Do not resubmit; reconcile with the operator. '
        + (error instanceof NovelAnythingError ? error.message : 'Transport or local persistence failed.'),
      'UNKNOWN_OUTCOME', error instanceof NovelAnythingError ? error.status : null, { cause: error })
    }
  }

  health(signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, signal => this.json('api/health', signal))
  }
  samples(id: RemoteId | null, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, signal => this.json('api/samples' + (id === null ? '' : `/${id}`), signal))
  }
  prepareGripperView(id: RecordingId, sourceCount: number | undefined, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      if (!this.ctx.ego) throw new NovelAnythingError('Robot Ego provider is required for recording preparation.', 'MISSING_ROBOT_CAPABILITY')
      const result = protocol.parse(protocol.prepared, await this.ctx.ego.prepareNovelAnything(id, sourceCount, signal))
      if (typeof result.archive_base64 !== 'string' || typeof result.bytes !== 'number'
        || result.bytes <= 0 || result.bytes > this.config.maxUploadBytes
        || result.archive_base64.length > 4 * Math.ceil(this.config.maxUploadBytes / 3)
        || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(result.archive_base64)) {
        throw new NovelAnythingError('Invalid prepared RGB archive or configured upload limit exceeded.', 'INVALID_RECORDING')
      }
      const bytes = Buffer.from(result.archive_base64, 'base64')
      if (bytes.length !== result.bytes || bytes[0] !== 0x50 || bytes[1] !== 0x4b) {
        throw new NovelAnythingError('Prepared archive bytes do not match its metadata.', 'INVALID_RECORDING')
      }
      const folder = await directory(this.root, join(this.inputs, 'prepared'))
      const sha256 = digest(bytes), archive = join(folder, `${id}-${sha256}.zip`)
      signal.throwIfAborted()
      await publish(archive, bytes)
      const metadata = protocol.parse(protocol.object, result.metadata)
      const manifest = join(folder, `${id}-${sha256}.json`)
      await save(manifest, { ...metadata, archive_path: archive, sha256, bytes: bytes.length,
        context: result.context!, target: result.target! })
      return { archive_path: archive, local_manifest_path: manifest, sha256, bytes: bytes.length,
        context: result.context!, target: result.target!, num_source_views: result.num_source_views!,
        label: 'gripper_view_prediction', metadata }
    })
  }
  upload(path: string, dataset: string, requestKey: string, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      protocol.parse(protocol.dataset, dataset)
      const source = await realpath(resolve(this.root, path))
      contained(this.root, source)
      if (!source.toLowerCase().endsWith('.zip') || !(await stat(source)).isFile()) {
        throw new NovelAnythingError('Upload requires a scene ZIP beneath inputRoot.', 'INVALID_INPUT')
      }
      const chunks: Buffer[] = []
      let size = 0
      for await (const chunk of createReadStream(source, { signal })) {
        size += chunk.length
        if (size > this.config.maxUploadBytes) throw new NovelAnythingError('Scene ZIP exceeds maxUploadBytes.', 'INPUT_TOO_LARGE')
        chunks.push(chunk)
      }
      const bytes = Buffer.concat(chunks)
      if (bytes.length < 4 || bytes[0] !== 0x50 || bytes[1] !== 0x4b) throw new NovelAnythingError('Input is not a ZIP.', 'INVALID_INPUT')
      const sha256 = digest(bytes)
      await directory(this.root, this.inputs)
      const archive = join(this.inputs, `${sha256}.zip`)
      await publish(archive, bytes)
      const localInput = { archive_path: archive, sha256, bytes: bytes.length, dataset }
      const form = new FormData()
      form.append('dataset', dataset)
      form.append('file', new Blob([new Uint8Array(bytes)], { type: 'application/zip' }), 'scene.zip')
      const response = await this.post('upload', requestKey, localInput, { method: 'POST', body: form }, signal,
        value => { protocol.parse(protocol.identity, value.upload_id) })
      const id = remoteId(String(response.upload_id))
      await save(join(this.state, `upload-${id}.json`), localInput)
      return { ...response, local_input: localInput }
    })
  }
  submit(request: Submission, requestKey: string, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      const payload = protocol.parse(protocol.submission, request)
      const health = await this.json('api/health', signal)
      if (health.status !== 'ready' && health.ready !== true) {
        throw new NovelAnythingError('Model is not ready; inspect novelanything_health before submitting.', 'NOT_READY')
      }
      const response = await this.post('submit', requestKey, payload, { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }, signal,
      value => { protocol.parse(protocol.job, value) })
      let input: JsonObject | null = null
      if (payload.upload_id) {
        try { input = await this.load(join(this.state, `upload-${payload.upload_id}.json`)) } catch (error) {
          if (!hasCode(error, 'ENOENT')) throw error
        }
      }
      await save(join(this.state, `job-${remoteId(String(response.id))}.json`), { request: payload, local_input: input })
      return { ...response, local_input: input, next_action: 'novelanything_status' }
    })
  }

  private async getJob(id: RemoteId, signal: AbortSignal): Promise<JsonObject> {
    const value = await this.json(`api/jobs/${id}`, signal)
    const parsed = protocol.parse(protocol.job, value)
    if (parsed.id !== id) throw new NovelAnythingError('Returned job identity differs from request.', 'INVALID_DATA')
    return value
  }
  status(id: RemoteId, waitSeconds: number, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      if (!Number.isFinite(waitSeconds) || waitSeconds < 0 || waitSeconds > this.config.maxWaitSeconds) {
        throw new NovelAnythingError('wait_seconds exceeds the configured polling bound.', 'INVALID_WAIT')
      }
      const deadline = Date.now() + waitSeconds * 1000
      for (;;) {
        const job = await this.getJob(id, signal)
        const state = String(job.status)
        if (terminal.has(state) || Date.now() >= deadline) return { ...job,
          next_action: state === 'succeeded' ? 'novelanything_result' : terminal.has(state) ? 'novelanything_log' : 'novelanything_status' }
        await delay(Math.min(this.config.pollIntervalMs, deadline - Date.now()), undefined, { signal })
      }
    })
  }

  private artifacts(job: JsonObject) {
    const result = protocol.parse(protocol.object, job.result)
    if (!Array.isArray(result.artifacts)) throw new NovelAnythingError('Result has no artifact list.', 'INVALID_DATA')
    const seen = new Set<string>()
    const artifacts = result.artifacts.map(value => {
      const artifact = protocol.parse(protocol.artifact, value)
      const path = protocol.artifactPath(artifact.path)
      if (seen.has(path.toLowerCase())) throw new NovelAnythingError('Duplicate artifact path.', 'INVALID_ARTIFACT')
      seen.add(path.toLowerCase())
      const expected = new URL(`api/jobs/${job.id}/artifacts/${path.split('/').map(encodeURIComponent).join('/')}`, this.base)
      const url = new URL(artifact.url, this.base)
      if (url.href !== expected.href) throw new NovelAnythingError('Artifact URL differs from the job artifact route.', 'INVALID_ARTIFACT')
      return artifact
    })
    for (const path of seen) {
      if ([...seen].some(other => other !== path && other.startsWith(path + '/'))) {
        throw new NovelAnythingError('Artifact file paths overlap directories.', 'INVALID_ARTIFACT')
      }
    }
    return artifacts
  }
  private async successful(id: RemoteId, signal: AbortSignal): Promise<JsonObject> {
    const job = await this.getJob(id, signal)
    if (job.status !== 'succeeded') throw new NovelAnythingError(`Job is ${job.status}; result is unavailable.`, 'RESULT_UNAVAILABLE')
    this.artifacts(job)
    return job
  }
  result(id: RemoteId, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => ({ ...await this.successful(id, signal), next_action: 'novelanything_download' }))
  }
  log(id: RemoteId, signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => ({ id, log: await this.request(`api/jobs/${id}/log`, {}, signal,
      async response => (await body(response, this.config.maxMetadataBytes)).toString('utf8')) }))
  }
  download(id: RemoteId, paths: string[], signal: AbortSignal): Promise<JsonObject> {
    return this.run(signal, async signal => {
      const job = await this.successful(id, signal)
      const artifacts = this.artifacts(job)
      if (!paths.length || new Set(paths).size !== paths.length || paths.some(path => !artifacts.some(a => a.path === path))) {
        throw new NovelAnythingError('Choose unique artifact paths returned by novelanything_result.', 'INVALID_SELECTION')
      }
      const folder = await directory(this.root, join(this.outputs, id))
      const manifestPath = join(folder, 'manifest.json')
      let previous: JsonObject = { files: [] }
      try { previous = await this.load(manifestPath) } catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
      if ('endpoint' in previous && (previous.endpoint !== this.base || previous.id !== id)) {
        throw new NovelAnythingError('Output directory belongs to another endpoint/job.', 'OUTPUT_CONFLICT')
      }
      const files = protocol.parse(protocol.object.array(), previous.files)
      let input: JsonObject | null = null
      try { input = await this.load(join(this.state, `job-${id}.json`)) } catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
      await save(join(folder, 'job.json'), job)
      const selected: JsonObject[] = []
      for (const path of paths) {
        const artifact = artifacts.find(a => a.path === path)!
        const destination = join(await directory(folder, dirname(join(folder, path))), path.split('/').at(-1)!)
        const temporary = `${destination}.${randomUUID()}.part`
        try {
          const downloaded = await this.request(`api/jobs/${id}/artifacts/${path.split('/').map(encodeURIComponent).join('/')}`, {}, signal,
            async response => {
              const handle = await open(temporary, 'wx', 0o600)
              const hash = createHash('sha256')
              let size = 0
              try {
                if (!response.body) throw new NovelAnythingError('Empty artifact body.', 'INVALID_DATA')
                for await (const chunk of response.body) {
                  size += chunk.length
                  if (size > this.config.maxArtifactBytes) throw new NovelAnythingError('Artifact exceeds maxArtifactBytes.', 'ARTIFACT_TOO_LARGE')
                  hash.update(chunk)
                  await handle.writeFile(chunk)
                }
                await handle.sync()
              } finally { await handle.close() }
              const sha256 = hash.digest('hex')
              if (artifact.bytes !== undefined && size !== artifact.bytes
                || artifact.sha256 !== undefined && sha256 !== artifact.sha256.toLowerCase()) {
                throw new NovelAnythingError('Artifact differs from server size/hash.', 'INTEGRITY_FAILURE')
              }
              return { artifact_path: path, path: destination, bytes: size, sha256,
                server_checksum_verified: artifact.sha256 !== undefined }
            }, this.config.transferTimeoutMs)
          signal.throwIfAborted()
          await rename(temporary, destination)
          const index = files.findIndex(file => file.artifact_path === path)
          if (index < 0) files.push(downloaded)
          else files[index] = downloaded
          await save(manifestPath, { version: 1, endpoint: this.base, id, input, result: job.result!, files })
          selected.push(downloaded)
        } finally {
          try { await unlink(temporary) } catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
        }
      }
      return { id, remote_succeeded: true, files: selected, local_manifest_path: manifestPath }
    })
  }
}

export default NovelAnythingClient
