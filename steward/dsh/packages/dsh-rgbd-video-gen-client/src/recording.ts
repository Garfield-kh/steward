/** Validate complete RGB recordings and select temporal evidence for bounded model inputs. */
import { z } from 'zod'
import { RgbdVideoGenError, type RecordingPreparation } from '@steward/dsh-rgbd-video-gen'
import { matrix, preparation, resolveTargets } from './scene.ts'
import { object, parse } from './protocol.ts'

const frame = z.object({ stamp: z.number().finite().nonnegative(), width: z.number().int().min(32).max(8192),
  height: z.number().int().min(32).max(8192), K: z.array(z.array(z.number().finite()).length(3)).length(3),
  camera_to_reference: matrix, image: z.string(), odometry: object, joint_encoders: object })
const evidence = z.object({ recording_id: z.string().regex(/^rgb_[a-f0-9]{32}$/), status: z.literal('succeeded'),
  camera_id: z.literal('head_camera'), input_modalities: z.tuple([z.literal('rgb')]), fps: z.number().positive().finite(),
  reference_frame: z.literal('first_recorded_head_camera_opencv'), target_stamp: z.number().finite().nonnegative(),
  pose_source: z.literal('robot_odometry_and_joint_encoders'), gripper_to_reference: matrix,
  frames: z.array(frame).min(2) })
const selection = z.object({ num_source_views: z.number().int().min(1).max(8).optional(),
  source_frame_indices: z.array(z.number().int().nonnegative()).min(1).max(8).optional() })
  .refine(v => v.num_source_views === undefined || v.source_frame_indices === undefined, 'Choose count or indices, never both')

/** Build model input from measured frames across a recording and preserve every source sample.
 * @param value - untrusted robot evidence export.
 * @param options - explicit targets and temporal selection.
 * @returns validated full evidence, selected indices and a bounded inline-image scene.
 */
export function buildRecordingScene(value: unknown, options: RecordingPreparation) {
  const data = parse(evidence, value), spec = parse(preparation, { targets: options.targets })
  const choose = parse(selection, { num_source_views: options.num_source_views, source_frame_indices: options.source_frame_indices })
  const first = data.frames[0], count = choose.num_source_views ?? Math.min(8, data.frames.length)
  if (first.camera_to_reference.some((row, i) => row.some((v, j) => Math.abs(v - (i === j ? 1 : 0)) > 1e-6))) {
    throw new RgbdVideoGenError('The first recorded middle camera must define the reference origin.', 'INVALID_RECORDING')
  }
  const indices = choose.source_frame_indices ?? Array.from({ length: count }, (_, i) =>
    count === 1 ? 0 : Math.round(i * (data.frames.length - 1) / (count - 1)))
  if (choose.num_source_views !== undefined && count > data.frames.length || indices.some((v, i) => v >= data.frames.length || i > 0 && v <= indices[i - 1])) {
    throw new RgbdVideoGenError('Evidence indices must be distinct, chronological and within the recording.', 'INVALID_RECORDING')
  }
  const decoded = data.frames.map((view, i) => {
    if (Math.abs(view.stamp - first.stamp - i / data.fps) > 1e-6 || view.stamp > data.target_stamp
      || view.width !== first.width || view.height !== first.height || JSON.stringify(view.K) !== JSON.stringify(first.K)
      || view.width * view.height > 16_000_000 || view.K[0][0] <= 0 || view.K[1][1] <= 0
      || view.K[0][1] !== 0 || view.K[1][0] !== 0 || view.K[0][2] < 0 || view.K[0][2] > view.width
      || view.K[1][2] < 0 || view.K[1][2] > view.height || view.K[2].some((v, j) => v !== (j === 2 ? 1 : 0))
      || !/^data:image\/jpeg;base64,(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(view.image)) {
      throw new RgbdVideoGenError('Recording requires consistent timestamped middle-camera JPEGs and intrinsics.', 'INVALID_RECORDING')
    }
    const bytes = Buffer.from(view.image.slice('data:image/jpeg;base64,'.length), 'base64')
    if (bytes[0] !== 0xff || bytes[1] !== 0xd8 || bytes[2] !== 0xff) throw new RgbdVideoGenError('Recording image is not JPEG.', 'INVALID_RECORDING')
    return bytes
  })
  const targets = resolveTargets(spec.targets, data.gripper_to_reference)
  const metadata = { recording_id: data.recording_id, fps: data.fps, camera_id: data.camera_id,
    input_modalities: data.input_modalities, recorded_frames: data.frames.length, source_frame_indices: indices,
    source_stamps: indices.map(index => data.frames[index].stamp), first_stamp: first.stamp,
    last_stamp: data.frames.at(-1)!.stamp, all_frames_used: indices.length === data.frames.length,
    coordinate_frame: data.reference_frame, pose_source: data.pose_source, target_stamp: data.target_stamp,
    targets: spec.targets.map(target => ({ frame: target.frame, label: target.label ?? null })), predicted: true }
  return { data, decoded, indices, metadata, scene: { format: 'rgbd_videogen_scene_v1', reference_camera: 'head_camera',
    cameras: { head_camera: { width: first.width, height: first.height, K: first.K } },
    frames: [...indices.map(index => ({ views: { head_camera: { image: data.frames[index].image,
      camera_to_world: data.frames[index].camera_to_reference } } })),
    ...targets.map(transform => ({ views: { head_camera: { image: null, camera_to_world: transform } } }))], metadata } }
}
