/** rgbd_video_gen tools; JSON results use the ordinary logged DSH tool pipeline. */
import { recordingId } from '@steward/dsh-ego'
import type { Context } from '@deepseek-ai/cordis'
import { defineTool, type ToolDefinition } from '@deepseek-ai/dsh-tools'
import { jobId, RgbdVideoGenError, type Json, type GenerationOptions, type RgbdVideoGenService } from '@steward/dsh-rgbd-video-gen'

/** Loader identity. */
export const name = 'steward-tool-rgbd-video-gen'
/** Services required from the bundle and DSH base. */
export const inject = ['tools', 'rgbdVideoGen']

const output = { schema: { type: 'json' } as const,
  render: (_args: unknown, value: Json) => [{ type: 'text' as const, text: JSON.stringify(value, null, 2) }] }
const key = { type: 'string' as const, required: true as const,
  description: 'Stable operation key: 8-128 letters/digits or ._:-. Reuse exactly after disconnects or unknown outcomes; new keys create new work.' }
const job = { type: 'string' as const, required: true as const, description: 'Actual j_ job_id returned by rgbd_video_gen. Never invent ids.' }
const wait = { type: 'number' as const, description: 'Bounded wait in seconds, default 0 (one query); provider limit defaults to 20. Aborting polling does not cancel generation.' }
const generation = {
  num_source_views: { type: 'number' as const, description: 'First 1-8 frames are evidence images; omitted uses current server default (normally 6). Explicitly set for fewer than 6 evidence images, e.g. 3 for a three-camera snapshot plus target poses. The model was trained on 6 evidence views; other counts may reduce quality.' },
  max_target_views: { type: 'number' as const, description: 'Target view limit, 1-24; omitted uses server default.' },
  num_inference_steps: { type: 'number' as const, description: 'Euler sampling steps, 1-100; omitted uses server default.' },
  seed: { type: 'number' as const, description: 'Nonnegative integer random seed; omitted uses server default.' },
  output_fps: { type: 'number' as const, description: 'Output video fps, 1-60; omitted uses server default.' },
  flow_dtype: { type: 'string' as const, enum: ['bf16', 'fp32'] as const, description: 'Precision; omitted uses server default.' },
}

const targetParameter = { type: 'array', required: true, description: '1-24 virtual cameras. For each supply a matrix OR all three look-at vectors, never both.',
          items: { type: 'object', additionalProperties: false, properties: {
            frame: { type: 'string', enum: ['reference', 'gripper'], required: true },
            camera_to_frame: { type: 'array', items: { type: 'array', items: { type: 'number' } }, description: 'Rigid 4x4 OpenCV camera-to-frame matrix.' },
            position: { type: 'array', items: { type: 'number' }, description: 'Camera centre [x,y,z] in the selected frame, metres.' },
            look_at: { type: 'array', items: { type: 'number' }, description: 'Point [x,y,z] to look toward, in the selected frame, metres.' },
            up: { type: 'array', items: { type: 'number' }, description: 'Image-up direction [x,y,z] in the selected frame; nonparallel to viewing direction.' },
            label: { type: 'string', description: 'Optional target label retained in scene metadata.' },
          } } } as const

function overrides(args: Partial<GenerationOptions>): Partial<GenerationOptions> {
  return { num_source_views: args.num_source_views, max_target_views: args.max_target_views,
    num_inference_steps: args.num_inference_steps, seed: args.seed, output_fps: args.output_fps, flow_dtype: args.flow_dtype }
}

/** Build tool definitions over the injected service.
 * @param service - remote provider or test implementation.
 * @returns definitions in registration order.
 */
export function rgbdVideoGenTools(service: RgbdVideoGenService): ToolDefinition[] {
  return [
    defineTool({ name: 'rgbd_video_gen_capabilities',
      description: 'Read rgbd_video_gen readiness, current defaults and limits. Does not run generation.', parameters: {}, output,
      execute: (_args, exec) => service.capabilities(exec.signal) }),
    defineTool({ name: 'rgbd_video_gen_prepare_scene',
      description: 'Build and save a SINGLE-TIMESTAMP snapshot scene from ego_capture_cameras capture.json. For whole-motion evidence use prepare_recording instead. '
        + 'Defaults to ONLY the middle head_camera RGB; explicitly select other source_cameras when requested. '
        + 'Targets are virtual OpenCV cameras described by a rigid camera_to_frame matrix OR position/look_at/up vectors. '
        + 'frame=reference uses the captured middle camera axes (right/down/forward); frame=gripper uses the synchronized '
        + 'gripper axes (forward/left/up). All positions are metres. Gripper-forward example: frame=gripper, '
        + 'position=[0,0,0], look_at=[1,0,0], up=[0,0,1]. No target is chosen implicitly. '
        + 'Reuses middle-camera intrinsics. Submit the returned scene_path with its exact num_source_views and max_target_views. '
        + 'No shell, file-authoring tools, subagents or remote generation are needed for preparation.',
      parameters: {
        capture_manifest_path: { type: 'string', required: true, description: 'Exact capture manifest_path beneath inputRoot.' },
        source_cameras: { type: 'array', items: { type: 'string' },
          description: 'Unique evidence camera IDs in order. Default [head_camera]; use side cameras only when explicitly requested.' },
        targets: targetParameter,
      }, output,
      execute: (args, exec) => service.prepareScene(args.capture_manifest_path,
        { ...(args.source_cameras === undefined ? {} : { source_cameras: args.source_cameras }), targets: args.targets }, exec.signal) }),
    defineTool({ name: 'rgbd_video_gen_prepare_recording',
      description: 'Prepare a scene from a succeeded ego_rgb_recording, retaining ALL recorded middle head_camera RGB images and measured poses locally. '
        + 'Use for evidence captured throughout motion; never replace it with the final ego_capture_cameras snapshot. '
        + 'Selected source images are timestamped views from the same camera in one measured frame. Targets use the same explicit matrix/look-at descriptions as prepare_scene; '
        + 'frame=reference is the FIRST recorded middle OpenCV camera, frame=gripper is the CURRENT gripper. '
        + 'The remote API declares at most 8 source views per job. Omit selection to use all if at most 8, otherwise uniformly sample 8 across the full recording including both endpoints. Optional '
        + 'num_source_views overrides the uniform sample count; source_frame_indices selects chronological indices for subsets/batches. '
        + 'Never claim all images conditioned the model when only a subset was used. Returns selected indices, all_frames_used, '
        + 'full recording_manifest_path and exact submission counts; no navigation or remote inference.',
      parameters: { recording_id: { type: 'string', required: true, description: 'Actual succeeded rgb_ recording identity.' },
        num_source_views: { type: 'number', description: 'Optional 1-8 uniformly spaced temporal views; default min(8, recorded_frames). Includes route endpoints when count is at least 2. Exclusive with source_frame_indices.' },
        source_frame_indices: { type: 'array', items: { type: 'number' }, description: 'Optional 1-8 distinct increasing recorded frame indices, for explicit subsets or batches. Exclusive with count.' },
        targets: targetParameter }, output,
      execute: (args, exec) => service.prepareRecording(recordingId(args.recording_id),
        { num_source_views: args.num_source_views, source_frame_indices: args.source_frame_indices, targets: args.targets }, exec.signal) }),
    defineTool({ name: 'rgbd_video_gen_submit',
      description: 'Upload a local rgbd_videogen_scene_v1 JSON scene with inline PNG/JPEG base64 images. Requires 2-64 ordered frames and at least num_source_views + 1 frames, known intrinsics and OpenCV camera-to-world poses in metres. '
        + 'First num_source_views frames provide evidence; remaining frames are target poses with optional images for GT comparison. Returns an accepted job, not completion. '
        + 'This sends images to the configured remote server. It does not estimate poses or accept SLARM capture ZIPs.',
      parameters: { scene_path: { type: 'string', required: true, description: 'JSON file beneath inputRoot; relative paths resolve from that root.' }, request_key: key, ...generation }, output,
      execute: async (args, exec) => service.submit(args.scene_path, await service.resolveOptions(overrides(args), exec.signal), args.request_key, exec.signal) }),
    defineTool({ name: 'rgbd_video_gen_upload',
      description: 'Upload a local ZIP containing rgbd_videogen_scene_v1 scene.json and its relative image files. Uses server generation defaults. '
        + 'Not compatible with world_capture slarm_scene_v1 ZIPs. Returns job_id; use status then result and download.',
      parameters: { archive_path: { type: 'string', required: true, description: 'ZIP beneath inputRoot, at most the configured upload limit.' }, request_key: key }, output,
      execute: (args, exec) => service.upload(args.archive_path, args.request_key, exec.signal) }),
    defineTool({ name: 'rgbd_video_gen_demo',
      description: 'Generate from a named server-installed demo, e.g. waymo080 or re10k_172c99489b18e0e6. No local images are needed. '
        + 'trajectory uses its target camera path; bev generates bird-eye poses and has no target ground truth. With bev, explicitly provide num_bev_views and bev_altitude_scale.',
      parameters: { scene: { type: 'string', required: true }, request_key: key, ...generation,
        target_mode: { type: 'string', enum: ['trajectory', 'bev'], required: true },
        num_bev_views: { type: 'number', description: 'Required for bev: integer 1-24.' },
        bev_altitude_scale: { type: 'number', description: 'Required for bev: positive height multiplier of the source camera horizontal radius.' } }, output,
      execute: async (args, exec) => {
        if (args.target_mode === 'bev' && (args.num_bev_views === undefined || args.bev_altitude_scale === undefined)) {
          throw new RgbdVideoGenError('bev requires num_bev_views and bev_altitude_scale.', 'INVALID_ARGUMENT')
        }
        if (args.target_mode === 'trajectory' && (args.num_bev_views !== undefined || args.bev_altitude_scale !== undefined)) {
          throw new RgbdVideoGenError('BEV settings require target_mode=bev.', 'INVALID_ARGUMENT')
        }
        const target = args.target_mode === 'bev'
          ? { target_mode: 'bev' as const, num_bev_views: args.num_bev_views!, bev_altitude_scale: args.bev_altitude_scale! }
          : { target_mode: 'trajectory' as const }
        return service.demo(args.scene, await service.resolveOptions(overrides(args), exec.signal), target, args.request_key, exec.signal)
      } }),
    defineTool({ name: 'rgbd_video_gen_status',
      description: 'Query generation. queued/running are pending; succeeded requires result; failed/cancelled means stop and report error. 100% progress alone is not success.',
      parameters: { job_id: job, wait_seconds: wait }, output,
      execute: (args, exec) => service.status(jobId(args.job_id), args.wait_seconds ?? 0, exec.signal) }),
    defineTool({ name: 'rgbd_video_gen_result',
      description: 'Read a successful result and save its manifest. RGB, metric depth and videos are predictions, not observed geometry. '
        + 'Target PSNR exists only with target GT. target_cameras[i] maps to pred_{num_source_views+i}.jpg, not always pred_{6+i}.jpg. Select remote filenames for download; local_filename distinguishes source_reconstruction_*.jpg from target_prediction_*.jpg.',
      parameters: { job_id: job }, output, execute: (args, exec) => service.result(jobId(args.job_id), exec.signal) }),
    defineTool({ name: 'rgbd_video_gen_download',
      description: 'Save under outputsDir and return local_manifest_path for downstream consumers. Download exact artifact filenames from a successful result (videos, frames.zip, depth.npz, comparison strip, meta.json). '
        + 'Saves loose predictions as source_reconstruction_*.jpg or zero-based target_prediction_*.jpg; ZIP entries retain remote names. Verifies byte counts and SHA256 before publishing local files; retries reuse verified files. No arbitrary URL or destination path is accepted.',
      parameters: { job_id: job, filenames: { type: 'array', items: { type: 'string' }, required: true } }, output,
      execute: (args, exec) => service.download(jobId(args.job_id), args.filenames, exec.signal) }),
    defineTool({ name: 'rgbd_video_gen_cancel',
      description: 'Explicitly cancel a job submitted or recovered in this local store. Stopping a chat does not cancel accepted remote work. Only cancelled=true confirms cancellation.',
      parameters: { job_id: job, wait_seconds: wait }, output,
      execute: (args, exec) => service.cancel(jobId(args.job_id), args.wait_seconds ?? 0, exec.signal) }),
  ]
}

/** Register tools with disposal owned by this plugin.
 * @param ctx - context providing tools and rgbdVideoGen.
 * @returns no value.
 */
export function apply(ctx: Context): void {
  for (const tool of rgbdVideoGenTools(ctx.rgbdVideoGen)) ctx.effect(() => ctx.tools.register(tool))
}
