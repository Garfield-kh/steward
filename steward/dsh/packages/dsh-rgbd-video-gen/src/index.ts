/** rgbd_video_gen service API. Accepted remote jobs survive tool cancellation and plugin disposal. */
import { Service, type Context } from '@deepseek-ai/cordis'
import { brandString, type Branded } from '@deepseek-ai/dsh-brand'
import type { RecordingId } from '@steward/dsh-ego'

/** JSON values preserved in remote metadata and local receipts. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }
/** Object returned by an rgbd_video_gen operation. */
export type JsonObject = { [key: string]: Json }
/** Remote generation identity. */
export type JobId = Branded<'RgbdVideoGenJobId'>

/** Validate a job identity before constructing a URL or local path.
 * @param value - id supplied by a tool, response or saved record.
 * @returns admitted job identity.
 */
export function jobId(value: string): JobId {
  if (!/^j_[A-Za-z0-9_-]{1,96}$/.test(value)) throw new RgbdVideoGenError('Invalid rgbd_video_gen job_id.', 'INVALID_JOB_ID')
  return brandString<JobId>(value)
}

/** Explicit generation settings; callers resolve defaults before submission. */
export interface GenerationOptions {
  num_source_views: number
  max_target_views: number
  num_inference_steps: number
  seed: number
  output_fps: number
  flow_dtype: 'bf16' | 'fp32'
}

/** Virtual camera expressed in the captured reference or synchronized gripper frame.
 * Supply either camera_to_frame (OpenCV camera-to-frame matrix) or position/look_at/up.
 */
export interface SceneTarget {
  frame: 'reference' | 'gripper'
  camera_to_frame?: number[][]
  position?: number[]
  look_at?: number[]
  up?: number[]
  label?: string
}

/** Scene preparation input; evidence defaults to the middle RGB camera only. */
export interface ScenePreparation {
  source_cameras?: string[]
  targets: SceneTarget[]
}

/** Temporal middle-camera evidence and explicit targets; choose count or chronological indices. */
export interface RecordingPreparation {
  num_source_views?: number
  source_frame_indices?: number[]
  targets: SceneTarget[]
}

/** Failure with a stable code; TRANSPORT can mean an unknown submission outcome. */
export class RgbdVideoGenError extends Error {
  constructor(message: string, readonly code: string, readonly retryable = false,
    readonly status: number | null = null, readonly details: Json = null) {
    super(`[${code}] ${message}`)
    this.name = 'RgbdVideoGenError'
  }
}

declare module '@deepseek-ai/cordis' {
  interface Context { rgbdVideoGen: RgbdVideoGenService }
}

/** Remote rgbd_video_gen operations. Local I/O cancellation never cancels remote generation. */
export abstract class RgbdVideoGenService extends Service {
  constructor(ctx: Context) { super(ctx, 'rgbdVideoGen') }
  /** Query readiness and capabilities without generating images.
   * @param signal - local cancellation.
   * @returns live service metadata.
   */
  abstract capabilities(signal: AbortSignal): Promise<JsonObject>
  /** Resolve omitted generation settings from the server's current defaults.
   * @param options - explicit overrides; absent fields use capabilities.defaults.
   * @param signal - local cancellation.
   * @returns complete validated generation settings.
   */
  abstract resolveOptions(options: Partial<GenerationOptions>, signal: AbortSignal): Promise<GenerationOptions>
  /** Prepare selected RGB evidence and explicit target views from a saved robot capture.
   * @param captureManifest - capture.json beneath inputRoot.
   * @param options - source cameras (default middle only) and target matrices or look-at descriptions.
   * @param signal - local cancellation.
   * @returns inputRoot-relative scene_path, absolute local_scene_path and source/target counts; no remote job is submitted.
   */
  abstract prepareScene(captureManifest: string, options: ScenePreparation, signal: AbortSignal): Promise<JsonObject>
  /** Prepare temporal evidence from a completed recording, retaining all RGB samples locally.
   * @param id - completed middle-camera recording.
   * @param options - targets and up to eight source frames, default all frames up to eight, otherwise eight uniformly spaced frames including both endpoints.
   * @param signal - local cancellation.
   * @returns scene path, full evidence manifest and exact selected frame indices; no remote submission.
   */
  abstract prepareRecording(id: RecordingId, options: RecordingPreparation, signal: AbortSignal): Promise<JsonObject>
  /** Submit a scene JSON file with inline images beneath inputRoot.
   * @param path - local rgbd_videogen_scene_v1 JSON path.
   * @param options - resolved generation settings.
   * @param requestKey - stable identity to reuse after an uncertain outcome.
   * @param signal - local cancellation.
   * @returns accepted job identity, archived local_input paths and current state, not completion.
   */
  abstract submit(path: string, options: GenerationOptions, requestKey: string, signal: AbortSignal): Promise<JsonObject>
  /** Upload a ZIP containing scene.json and relative image files, using server generation defaults.
   * @param path - local ZIP beneath inputRoot.
   * @param requestKey - stable upload identity.
   * @param signal - local cancellation.
   * @returns accepted job identity, archived local_input paths and state.
   */
  abstract upload(path: string, requestKey: string, signal: AbortSignal): Promise<JsonObject>
  /** Submit a named server demo; names must come from the operator or service documentation.
   * @param scene - installed demo name.
   * @param options - resolved generation settings.
   * @param target - trajectory or BEV settings; BEV poses are generated by the server.
   * @param requestKey - stable submission identity.
   * @param signal - local cancellation.
   * @returns accepted job identity, archived local_input paths and state.
   */
  abstract demo(scene: string, options: GenerationOptions,
    target: { target_mode: 'trajectory' } | { target_mode: 'bev'; num_bev_views: number; bev_altitude_scale: number },
    requestKey: string, signal: AbortSignal): Promise<JsonObject>
  /** Query a job, optionally polling up to the configured wait bound.
   * @param id - actual remote job id.
   * @param waitSeconds - zero for one query; otherwise bounded polling.
   * @param signal - stops local waiting only.
   * @returns latest state, progress and next action.
   */
  abstract status(id: JobId, waitSeconds: number, signal: AbortSignal): Promise<JsonObject>
  /** Read a successful result and persist its complete manifest.
   * @param id - job identity.
   * @param signal - local cancellation.
   * @returns metrics, artifact descriptors and manifest path; rejects pending or failed jobs.
   */
  abstract result(id: JobId, signal: AbortSignal): Promise<JsonObject>
  /** Download exact remote filenames; local prediction names distinguish source reconstruction from target prediction. Only size/hash-verified files are published.
   * @param id - successful job identity.
   * @param filenames - unique nonempty selection from result().
   * @param signal - aborts transfers and removes incomplete files.
   * @returns verified local paths, remote filename/local_filename mappings and local_manifest_path; completed downloads survive a later transfer failure.
   */
  abstract download(id: JobId, filenames: string[], signal: AbortSignal): Promise<JsonObject>
  /** Explicitly request cancellation of a job submitted or recovered in this store.
   * @param id - locally owned job identity.
   * @param waitSeconds - bounded polling after submitting cancellation.
   * @param signal - local cancellation; remote cancellation can be retried.
   * @returns latest state; only cancelled=true confirms cancellation.
   */
  abstract cancel(id: JobId, waitSeconds: number, signal: AbortSignal): Promise<JsonObject>
}
