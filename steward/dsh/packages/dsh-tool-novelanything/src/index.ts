/** Robot NovelAnything tools rendered as ordinary logged JSON/text results. */
import type { Context } from '@deepseek-ai/cordis'
import { defineTool, type ToolDefinition } from '@deepseek-ai/dsh-tools'
import { remoteId, type NovelAnythingService, type Json } from '@steward/dsh-novelanything'
import { recordingId } from '@steward/dsh-ego'

/** Loader identity. */
export const name = 'steward-tool-novelanything'
/** Required service registrations. */
export const inject = ['tools', 'novelAnything']
const output = { schema: { type: 'json' } as const,
  render: (_args: unknown, value: Json) => [{ type: 'text' as const, text: JSON.stringify(value, null, 2) }] }
const key = { type: 'string' as const, required: true as const,
  description: 'Stable local request identity, 8–128 letters/digits or ._:-. Reuse after accepted replies. UNKNOWN_OUTCOME requires operator reconciliation; never retry with a new key.' }
const job = { type: 'string' as const, required: true as const, description: 'Exact job id returned by NovelAnything; never invent identities.' }

/** Create model-facing tools over the hosted model service.
 * @param service - injected NovelAnything provider.
 * @returns definitions in registration order.
 */
export function novelAnythingTools(service: NovelAnythingService): ToolDefinition[] {
  return [
    defineTool({ name: 'novelanything_health', description: 'Check hosted NovelAnything model readiness without inference.', parameters: {}, output,
      execute: (_args, exec) => service.health(exec.signal) }),
    defineTool({ name: 'novelanything_samples', description: 'List server demo samples or inspect one before selecting frames. Samples are not observations of robot surroundings.',
      parameters: { sample_id: { type: 'string', description: 'Omit to list samples; otherwise use an actual listed sample id.' } }, output,
      execute: (args, exec) => service.samples(args.sample_id === undefined ? null : remoteId(args.sample_id), exec.signal) }),
    defineTool({ name: 'novelanything_upload', description: 'Archive and upload a robot-workspace ZIP with root scene.json and posed RGB files, without inference. '
      + 'Robot RGB inputs must come only from the middle head_camera; depth inputs are unsupported by this skill; exclude head_camera_left and head_camera_right. '
      + 'Context needs RGB, OpenCV world-to-camera w2c and fxfycxcy; omit depth_path. Targets need w2c; no depth input or GT is used. A single capture or SLARM/rgbd_video_gen ZIP is not this format.',
      parameters: { archive_path: { type: 'string', required: true, description: 'Existing compliant scene ZIP beneath robot inputRoot.' },
        dataset: { type: 'string', enum: ['re10k', 'scannetpp', 'dl3dv', 'mp3d'], description: 'Preprocessing convention; default re10k.' }, request_key: key }, output,
      execute: (args, exec) => service.upload(args.archive_path, args.dataset ?? 're10k', args.request_key, exec.signal) }),
    defineTool({ name: 'novelanything_prepare_gripper_view', description: 'Prepare a local NovelAnything ZIP from a succeeded ego_rgb_recording. '
      + 'Uses only recorded middle head_camera RGB, with no depth input, '
      + 'and derives shared-frame w2c from synchronized robot odometry, encoders and the robot model. '
      + 'The target is the CURRENT gripper link origin looking along gripper +X with image-up +Z, using middle-camera intrinsics. '
      + 'Returns archive_path and context/target indices; retains all sampled evidence in the ZIP. No navigation, general shell/files or inference. '
      + 'Upload the returned archive with novelanything_upload, then submit with returned indices and label the result gripper_view_prediction. '
      + 'If preparation is unavailable or fails, report that failure; never substitute rgbd_video_gen for an explicitly requested NovelAnything job.',
      parameters: { recording_id: { type: 'string', required: true, description: 'Actual succeeded rgb_ recording identity.' },
        num_source_views: { type: 'number', description: 'Optional 2–16 contexts, at most recorded frames; omitted selects up to 16 across the recording.' } }, output,
      execute: (args, exec) => service.prepareGripperView(recordingId(args.recording_id), args.num_source_views, exec.signal) }),
    defineTool({ name: 'novelanything_submit', description: 'Queue NovelAnything RGB/depth prediction from exactly one sample_id or upload_id. '
      + 'When the user explicitly requests NovelAnything, do not substitute rgbd_video_gen if preparation or submission fails. '
      + 'Set both context and target or neither (scene defaults). Context: 2–16 distinct indices with RGB and calibration; target: 1–6 distinct disjoint indices. '
      + 'For robot evidence, all context frames must be from the same middle head_camera at different timestamps/poses; never combine side-camera RGB or supply depth. '
      + 'Targets may omit GT. Returns acceptance, not completion. No automatic POST retry; UNKNOWN_OUTCOME must be reconciled by the operator.',
      parameters: { sample_id: { type: 'string', description: 'Explicit demo sample id; mutually exclusive with upload_id.' },
        upload_id: { type: 'string', description: 'Id returned by novelanything_upload; mutually exclusive with sample_id.' },
        context: { type: 'array', items: { type: 'number' }, description: 'Zero-based context frame indices; omit with target for scene defaults.' },
        target: { type: 'array', items: { type: 'number' }, description: 'Zero-based target frame indices, disjoint from context.' },
        metrics: { type: 'boolean', description: 'Default true: score only targets with GT of that modality. False disables scoring, not prediction.' }, request_key: key }, output,
      execute: (args, exec) => service.submit({
        ...(args.sample_id === undefined ? {} : { sample_id: args.sample_id }),
        ...(args.upload_id === undefined ? {} : { upload_id: args.upload_id }),
        ...(args.context === undefined ? {} : { context: args.context }),
        ...(args.target === undefined ? {} : { target: args.target }), metrics: args.metrics ?? true,
      }, args.request_key, exec.signal) }),
    defineTool({ name: 'novelanything_status', description: 'Inspect or briefly poll a NovelAnything job. queued/running are pending; failed/cancelled/interrupted stop synthesis. Local timeout does not cancel remote work.',
      parameters: { job_id: job, wait_seconds: { type: 'number', description: 'Default 0: one query. Otherwise bounded by provider maxWaitSeconds (default 20).' } }, output,
      execute: (args, exec) => service.status(remoteId(args.job_id), args.wait_seconds ?? 0, exec.signal) }),
    defineTool({ name: 'novelanything_result', description: 'Read successful result metadata and exact artifact paths. Rejects every state except succeeded. Generated RGB and depth are predictions, not observations or motion clearance.',
      parameters: { job_id: job }, output, execute: (args, exec) => service.result(remoteId(args.job_id), exec.signal) }),
    defineTool({ name: 'novelanything_log', description: 'Read bounded NovelAnything server job log for failure diagnosis.',
      parameters: { job_id: job }, output, execute: (args, exec) => service.log(remoteId(args.job_id), exec.signal) }),
    defineTool({ name: 'novelanything_download', description: 'Download exact artifact paths from a successful result into outputs/novelanything/<job_id>, preserving subdirectories. '
      + 'Returns absolute files and local_manifest_path. Local SHA256 is recorded; server checksum verification occurs only when the server supplies a checksum.',
      parameters: { job_id: job, artifact_paths: { type: 'array', items: { type: 'string' }, required: true,
        description: 'Unique nonempty selection of exact result.artifacts[].path values.' } }, output,
      execute: (args, exec) => service.download(remoteId(args.job_id), args.artifact_paths, exec.signal) }),
  ]
}

/** Register all tool contributions with effect-owned disposal.
 * @param ctx - injected Cordis context.
 */
export function apply(ctx: Context): void {
  for (const tool of novelAnythingTools(ctx.novelAnything)) ctx.effect(() => ctx.tools.register(tool))
}
