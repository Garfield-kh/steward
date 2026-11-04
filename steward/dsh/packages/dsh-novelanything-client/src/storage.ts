/** Atomic private metadata, bounded stream reads and path confinement. */
import { randomUUID, createHash } from 'node:crypto'
import { mkdir, open, rename, link, unlink, realpath } from 'node:fs/promises'
import { dirname, relative, isAbsolute, sep } from 'node:path'
import { NovelAnythingError, type JsonObject } from '@steward/dsh-novelanything'

/** Hash input bytes or canonical request text.
 * @param value - content to hash.
 * @returns lowercase SHA256.
 */
export function digest(value: string | Uint8Array): string { return createHash('sha256').update(value).digest('hex') }
/** Identify a Node filesystem error.
 * @param error - caught error.
 * @param code - expected error code.
 * @returns whether codes match.
 */
export function hasCode(error: unknown, code: string): boolean {
  return error instanceof Error && 'code' in error && error.code === code
}
/** Require a resolved path to stay under its owning directory.
 * @param root - resolved directory.
 * @param path - resolved candidate.
 */
export function contained(root: string, path: string): void {
  const part = relative(root, path)
  if (isAbsolute(part) || part === '..' || part.startsWith('..' + sep)) {
    throw new NovelAnythingError('Path escapes its configured root.', 'INVALID_PATH')
  }
}
/** Create a directory and reject links resolving outside the configured root.
 * @param root - resolved owner directory.
 * @param path - destination directory.
 * @returns resolved directory.
 */
export async function directory(root: string, path: string): Promise<string> {
  contained(root, path)
  if (path !== root) await directory(root, dirname(path))
  await mkdir(path, { recursive: true })
  const actual = await realpath(path)
  contained(root, actual)
  return actual
}
/** Publish bytes through an exclusive temporary file; clean up on failure.
 * @param path - destination beneath an already admitted directory.
 * @param bytes - complete bytes.
 * @param exclusive - reserve without replacement when true.
 */
export async function publish(path: string, bytes: Uint8Array, exclusive = false): Promise<void> {
  const temporary = `${path}.${randomUUID()}.part`
  try {
    const handle = await open(temporary, 'wx', 0o600)
    try { await handle.writeFile(bytes); await handle.sync() } finally { await handle.close() }
    if (exclusive) await link(temporary, path)
    else await rename(temporary, path)
  } finally {
    try { await unlink(temporary) } catch (error) { if (!hasCode(error, 'ENOENT')) throw error }
  }
}
/** Atomically save complete JSON metadata.
 * @param path - destination file.
 * @param value - JSON object.
 * @param exclusive - reserve without replacement when true.
 */
export async function save(path: string, value: JsonObject, exclusive = false): Promise<void> {
  await publish(path, Buffer.from(JSON.stringify(value, null, 2) + '\n'), exclusive)
}
/** Read bounded bytes from a web response, cancelling the reader on any outcome.
 * @param response - HTTP response with an attached cancellation signal.
 * @param limit - maximum body bytes.
 * @returns complete response bytes.
 */
export async function body(response: Response, limit: number): Promise<Buffer> {
  const reader = response.body?.getReader()
  if (!reader) throw new NovelAnythingError('Empty HTTP response.', 'INVALID_DATA')
  const chunks: Uint8Array[] = []
  let size = 0
  try {
    for (;;) {
      const chunk = await reader.read()
      if (chunk.done) break
      size += chunk.value.length
      if (size > limit) throw new NovelAnythingError('Response exceeds configured byte limit.', 'RESPONSE_TOO_LARGE')
      chunks.push(chunk.value)
    }
  } finally { try { await reader.cancel() } finally { reader.releaseLock() } }
  return Buffer.concat(chunks)
}
