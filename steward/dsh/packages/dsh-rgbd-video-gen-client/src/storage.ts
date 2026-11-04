/** Atomic local metadata and streaming file digests. */
import { createHash, randomUUID } from 'node:crypto'
import { createReadStream, createWriteStream } from 'node:fs'
import { pipeline } from 'node:stream/promises'
import { link, mkdir, open, readFile, realpath, rename, unlink, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { RgbdVideoGenError, type JsonObject } from '@steward/dsh-rgbd-video-gen'
import { object, parse, savedRequest } from './protocol.ts'

/** Check a filesystem failure without changing other errors.
 * @param error - caught failure.
 * @param code - expected Node filesystem code.
 * @returns whether it is that failure.
 */
export function hasCode(error: unknown, code: string): boolean {
  return error instanceof Error && 'code' in error && error.code === code
}

/** Hash a string for operation names and service namespaces.
 * @param value - text to hash.
 * @returns lowercase SHA256.
 */
export function digest(value: string): string { return createHash('sha256').update(value).digest('hex') }

/** Create an input directory without following a parent outside the configured robot workspace.
 * @param root - canonical workspace root.
 * @param path - requested prepared-input directory.
 * @returns canonical directory after checking each parent.
 */
export async function inputDirectory(root: string, path: string): Promise<string> {
  const within = relative(root, resolve(path))
  if (isAbsolute(within) || within === '..' || within.startsWith('..' + sep)) {
    throw new RgbdVideoGenError('Prepared input directory leaves inputRoot.', 'INVALID_INPUT_PATH')
  }
  let current = root
  for (const part of within.split(sep).filter(Boolean)) {
    current = join(current, part)
    try { await mkdir(current) } catch (error) { if (!hasCode(error, 'EEXIST')) throw error }
    current = await realpath(current)
    const actual = relative(root, current)
    if (isAbsolute(actual) || actual === '..' || actual.startsWith('..' + sep)) {
      throw new RgbdVideoGenError('Prepared input parent leaves inputRoot.', 'INVALID_INPUT_PATH')
    }
  }
  return current
}

/** Publish a complete sensor image without exposing partial bytes or replacing prior data on cancellation.
 * @param path - admitted local image destination.
 * @param bytes - decoded sensor JPEG.
 * @param signal - local cancellation.
 * @returns completion after atomic publication and temporary-file cleanup.
 */
export async function saveImage(path: string, bytes: Buffer, signal: AbortSignal): Promise<void> {
  const temporary = `${path}.${randomUUID()}.part`
  try {
    await writeFile(temporary, bytes, { signal, flag: 'wx' })
    signal.throwIfAborted()
    await rename(temporary, path)
  } finally {
    try { await unlink(temporary) } catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
  }
}

/** Hash a file without buffering its contents.
 * @param path - local file.
 * @param signal - cancellation.
 * @returns lowercase SHA256.
 */
export async function fileDigest(path: string, signal: AbortSignal): Promise<string> {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path, { signal })) hash.update(chunk)
  return hash.digest('hex')
}

/** Archive exact upload bytes before submitting, publishing only a verified complete copy.
 * @param source - admitted upload file.
 * @param target - content-addressed archive path.
 * @param sha256 - expected source digest.
 * @param signal - cancellation.
 * @returns completion after publication and cleanup.
 */
export async function snapshotFile(source: string, target: string, sha256: string, signal: AbortSignal): Promise<void> {
  await mkdir(dirname(target), { recursive: true })
  const temporary = `${target}.${randomUUID()}.part`
  try {
    await pipeline(createReadStream(source), createWriteStream(temporary, { flags: 'wx' }), { signal })
    if (await fileDigest(temporary, signal) !== sha256) {
      throw new RgbdVideoGenError('Input changed while archiving; no upload was sent.', 'INPUT_CHANGED')
    }
    await rename(temporary, target)
  } finally {
    try { await unlink(temporary) } catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
  }
}

/** Read a local input with a byte bound that also applies if it grows while being read.
 * @param path - admitted file path.
 * @param limit - maximum file bytes.
 * @param signal - local cancellation.
 * @returns complete file bytes.
 */
export async function readBounded(path: string, limit: number, signal: AbortSignal): Promise<Buffer> {
  const chunks: Buffer[] = []
  let length = 0
  for await (const chunk of createReadStream(path, { signal })) {
    length += chunk.length
    if (length > limit) throw new RgbdVideoGenError('Input exceeds its configured byte limit.', 'REQUEST_TOO_LARGE')
    chunks.push(chunk)
  }
  return Buffer.concat(chunks)
}

/** Publish complete JSON atomically, optionally refusing replacement.
 * @param path - destination.
 * @param value - JSON data.
 * @param exclusive - true reserves an immutable request identity; an existing path raises EEXIST.
 * @returns completion after publication and temporary-file cleanup.
 */
export async function save(path: string, value: JsonObject, exclusive = false): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  const temporary = `${path}.${randomUUID()}.tmp`
  try {
    const file = await open(temporary, 'wx')
    try {
      await file.writeFile(JSON.stringify(value, null, 2) + '\n', 'utf8')
      await file.sync()
    } finally { await file.close() }
    if (exclusive) await link(temporary, path)
    else await rename(temporary, path)
  } finally {
    try { await unlink(temporary) } catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
  }
}

/** Load JSON with explicit failure on a malformed or unsupported saved record.
 * @param path - local record.
 * @returns decoded object.
 */
export async function load(path: string): Promise<JsonObject> {
  const contents = await readFile(path, 'utf8')
  let value: unknown
  try { value = JSON.parse(contents) } catch (error) {
    throw new RgbdVideoGenError(`Cannot read saved rgbd_video_gen record ${path}: ${String(error)}`, 'INVALID_LOCAL_STATE')
  }
  return parse(object, value)
}

/** Reserve a stable server key before the first POST, rejecting changed content on replay.
 * @param root - service-specific operation store.
 * @param endpoint - configured API base.
 * @param kind - operation type.
 * @param requestKey - caller's stable retry identity.
 * @param payload - canonical request fields, including input digest for uploads.
 * @returns durable record path and idempotency key.
 */
export async function reserve(root: string, endpoint: string, kind: 'submit' | 'upload' | 'demo' | 'cancel', requestKey: string,
  payload: JsonObject): Promise<{ path: string; key: string }> {
  const path = join(root, 'requests', `${kind}-${digest(requestKey)}.json`)
  const candidate = { version: 1, endpoint, kind, request_key: requestKey, idempotency_key: randomUUID(), payload }
  try { await save(path, candidate, true) } catch (error) { if (!hasCode(error, 'EEXIST')) throw error }
  const stored = parse(savedRequest, await load(path))
  if (stored.endpoint !== endpoint || stored.kind !== kind || stored.request_key !== requestKey
    || JSON.stringify(stored.payload) !== JSON.stringify(payload)) {
    throw new RgbdVideoGenError('This request_key already belongs to different input or options; use a new key only for a new operation.',
      'LOCAL_IDEMPOTENCY_CONFLICT')
  }
  return { path, key: stored.idempotency_key }
}
