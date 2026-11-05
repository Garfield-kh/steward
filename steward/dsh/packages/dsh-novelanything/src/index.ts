/** Hosted NovelAnything RGB-only input synthesis; predictions never authorize robot motion. */
import { Service, type Context } from '@deepseek-ai/cordis'
import { brandString, type Branded } from '@deepseek-ai/dsh-brand'
import type { RecordingId } from '@steward/dsh-ego'

/** JSON data retained in tool output and local manifests. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }
/** Remote or saved JSON object. */
export type JsonObject = { [key: string]: Json }
/** Remote job, upload or sample identity. */
export type RemoteId = Branded<'NovelAnythingRemoteId'>

/** Admit a remote identity as a single URL/path component.
 * @param value - identity from tools or remote metadata.
 * @returns validated identity.
 */
export function remoteId(value: string): RemoteId {
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(value)) {
    throw new NovelAnythingError('Invalid NovelAnything identity.', 'INVALID_ID')
  }
  return brandString<RemoteId>(value)
}

/** Structured operation failure. UNKNOWN_OUTCOME forbids automatic resubmission. */
export class NovelAnythingError extends Error {
  constructor(message: string, readonly code: string, readonly status: number | null = null, options?: ErrorOptions) {
    super(`[${code}] ${message}`, options)
    this.name = 'NovelAnythingError'
  }
}

/** Explicit submission settings. Absent frame lists use the scene's server defaults. */
export interface Submission {
  sample_id?: string
  upload_id?: string
  context?: number[]
  target?: number[]
  metrics: boolean
}

declare module '@deepseek-ai/cordis' {
  interface Context { novelAnything: NovelAnythingService }
}

/** Remote generation API. Cancellation/disposal stops local I/O, never remote jobs. */
export abstract class NovelAnythingService extends Service {
  constructor(ctx: Context) { super(ctx, 'novelAnything') }
  /** Inspect model readiness without generating.
   * @param signal - local cancellation.
   * @returns server health fields.
   */
  abstract health(signal: AbortSignal): Promise<JsonObject>
  /** List samples or inspect an explicit sample; demos do not describe robot surroundings.
   * @param id - sample identity, or null to list samples.
   * @param signal - local cancellation.
   * @returns server sample metadata.
   */
  abstract samples(id: RemoteId | null, signal: AbortSignal): Promise<JsonObject>
  /** Prepare a local NovelAnything ZIP from a finished middle RGB recording and the current gripper pose.
   * @param id - successful robot recording identity.
   * @param sourceCount - explicit 2–16 frame count; undefined selects up to 16 evenly spaced frames.
   * @param signal - local cancellation.
   * @returns archive_path, context/target indices, measured-pose metadata and gripper_view_prediction label; no motion or inference.
   */
  abstract prepareGripperView(id: RecordingId, sourceCount: number | undefined, signal: AbortSignal): Promise<JsonObject>
  /** Archive and upload a scene ZIP without inference. Stable keys recover accepted replies locally.
   * @param path - ZIP beneath inputRoot.
   * @param dataset - preprocessing convention.
   * @param requestKey - caller identity; unknown outcomes require operator reconciliation.
   * @param signal - local cancellation.
   * @returns upload identity and archived input paths.
   */
  abstract upload(path: string, dataset: string, requestKey: string, signal: AbortSignal): Promise<JsonObject>
  /** Submit one source and optional disjoint frame lists. POST is never retried automatically.
   * @param request - explicit metrics flag and source/view selection.
   * @param requestKey - persistent local request identity.
   * @param signal - local cancellation.
   * @returns accepted job, not completion.
   */
  abstract submit(request: Submission, requestKey: string, signal: AbortSignal): Promise<JsonObject>
  /** Inspect or poll a job; progress alone never means success.
   * @param id - returned remote job identity.
   * @param waitSeconds - zero for one query, otherwise bounded local polling.
   * @param signal - stops local waiting only.
   * @returns latest job and next action.
   */
  abstract status(id: RemoteId, waitSeconds: number, signal: AbortSignal): Promise<JsonObject>
  /** Read result descriptors only after succeeded.
   * @param id - remote job identity.
   * @param signal - local cancellation.
   * @returns job result including artifact paths.
   */
  abstract result(id: RemoteId, signal: AbortSignal): Promise<JsonObject>
  /** Read a bounded server log for diagnosis.
   * @param id - remote job identity.
   * @param signal - local cancellation.
   * @returns log text.
   */
  abstract log(id: RemoteId, signal: AbortSignal): Promise<JsonObject>
  /** Download selected result paths, preserving directories and publishing complete files atomically.
   * @param id - successful job identity.
   * @param paths - exact nonempty artifact paths from result().
   * @param signal - aborts transfers and removes partial files.
   * @returns absolute paths, local manifest and local SHA256; server checksum verification only if supplied.
   */
  abstract download(id: RemoteId, paths: string[], signal: AbortSignal): Promise<JsonObject>
}
