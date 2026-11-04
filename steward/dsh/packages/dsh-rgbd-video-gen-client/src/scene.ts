/** Prepare selected RGB evidence and explicit virtual-camera targets in one local reference frame. */
import { z } from 'zod'
import { RgbdVideoGenError, type ScenePreparation, type JsonObject } from '@steward/dsh-rgbd-video-gen'

/** Rigid OpenCV camera transform accepted from files and model JSON. */
export const matrix = z.array(z.array(z.number().finite()).length(4)).length(4).refine(value => {
  if (value[3].some((v, i) => Math.abs(v - (i === 3 ? 1 : 0)) > 1e-8)) return false
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    const dot = value.slice(0, 3).reduce((sum, row) => sum + row[i] * row[j], 0)
    if (Math.abs(dot - (i === j ? 1 : 0)) > 1e-6) return false
  }
  const [a, b, c] = value
  const determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
    + a[2] * (b[0] * c[1] - b[1] * c[0])
  return Math.abs(determinant - 1) <= 1e-6 && value.slice(0, 3).every(row => Math.abs(row[3]) < 1e6)
}, 'Expected a right-handed rigid transform in metres')
const camera = z.object({ width: z.number().int().min(32).max(8192), height: z.number().int().min(32).max(8192),
  K: z.array(z.array(z.number().finite()).length(3)).length(3), camera_to_reference: matrix })
const view = camera.extend({ image_path: z.string().min(1) })
const capture = z.object({ stamp: z.number().finite().nonnegative(), reference_camera: z.literal('head_camera'),
  camera_axes: z.literal('opencv_x_right_y_down_z_forward'), calibration_source: z.literal('rig_calibration'),
  views: z.record(z.string(), z.unknown()), gripper_pose: z.unknown().optional() })
const gripper = z.object({ stamp: z.number().finite().nonnegative(),
  pose_source: z.literal('joint_encoders_and_robot_model'), gripper_frame: z.literal('gripper'),
  gripper_axes: z.literal('x_forward_y_left_z_up'), camera_axes: z.literal('opencv_x_right_y_down_z_forward'),
  gripper_to_camera: z.record(z.string(), matrix) })
const vector = z.array(z.number().finite()).length(3)
const targetFields = { frame: z.enum(['reference', 'gripper']), label: z.string().min(1).max(128).optional() }
/** Explicit snapshot evidence and virtual-camera selection. */
export const preparation = z.object({ source_cameras: z.array(z.string().min(1)).min(1).max(8)
  .refine(ids => new Set(ids).size === ids.length, 'Evidence cameras must be unique').default(['head_camera']),
  targets: z.array(z.union([
    z.object({ ...targetFields, camera_to_frame: matrix }).strict(),
    z.object({ ...targetFields, position: vector, look_at: vector, up: vector }).strict(),
  ])).min(1).max(24) }).strict()

function checked<T>(schema: z.ZodType<T>, value: unknown, code = 'INVALID_CAPTURE'): T {
  const parsed = schema.safeParse(value)
  if (!parsed.success) throw new RgbdVideoGenError(parsed.error.message, code)
  return parsed.data
}

function multiply(a: number[][], b: number[][]): number[][] {
  return a.map(row => b[0].map((_, column) => row.reduce((sum, value, k) => sum + value * b[k][column], 0)))
}

function equal(a: number[][], b: number[][]): boolean {
  return a.every((row, i) => row.every((v, j) => Math.abs(v - b[i][j]) < 1e-6))
}

function cross(a: number[], b: number[]): number[] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

function normalized(v: number[]): number[] {
  const length = Math.hypot(...v)
  if (length < 1e-10) throw new RgbdVideoGenError('Target direction and up must be nonzero and nonparallel.', 'INVALID_TARGET')
  return v.map(component => component / length)
}

/** Resolve explicit target descriptions into OpenCV camera-to-reference matrices.
 * @param targets - validated target descriptions.
 * @param gripToReference - current gripper transform, required for gripper targets.
 * @returns one camera-to-reference matrix per target.
 */
export function resolveTargets(targets: z.infer<typeof preparation>['targets'], gripToReference: number[][] | null): number[][][] {
  return targets.map(target => {
    let transform: number[][]
    if ('camera_to_frame' in target) transform = target.camera_to_frame
    else {
      const forward = normalized(target.look_at.map((v, i) => v - target.position[i]))
      const right = normalized(cross(forward, normalized(target.up))), down = cross(forward, right)
      transform = [...target.position.map((p, i) => [right[i], down[i], forward[i], p]), [0, 0, 0, 1]]
    }
    return checked(matrix, target.frame === 'gripper' && gripToReference ? multiply(gripToReference, transform) : transform, 'INVALID_TARGET')
  })
}

/** Build selected evidence frames and one or more explicitly requested virtual cameras.
 * @param value - untrusted robot capture manifest.
 * @param options - evidence selection and target specifications; omitted selection means middle RGB only.
 * @param image - loads selected JPEGs within the permitted input directory.
 * @returns scene, capture timestamp and exact submission counts in the captured middle-camera reference.
 */
export async function buildScene(value: unknown, options: ScenePreparation, image: (path: string) => Promise<Buffer>):
Promise<{ scene: JsonObject; stamp: number; sources: number; targets: number }> {
  const spec = checked(preparation, options, 'INVALID_ARGUMENT')
  const data = checked(capture, value), middle = checked(camera, data.views.head_camera)
  const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  if (!equal(middle.camera_to_reference, identity)) throw new RgbdVideoGenError('The middle camera must define the reference origin.', 'INVALID_CAPTURE')
  const K = middle.K
  if (K[0][0] <= 0 || K[1][1] <= 0 || K[0][1] !== 0 || K[1][0] !== 0
    || K[2][0] !== 0 || K[2][1] !== 0 || K[2][2] !== 1
    || K[0][2] < 0 || K[0][2] > middle.width || K[1][2] < 0 || K[1][2] > middle.height
    || middle.width * middle.height > 16_000_000) {
    throw new RgbdVideoGenError('Invalid capture pixel intrinsics.', 'INVALID_CAPTURE')
  }
  const grip = spec.targets.some(target => target.frame === 'gripper') ? checked(gripper, data.gripper_pose) : null
  if (grip && grip.stamp !== data.stamp) throw new RgbdVideoGenError('Capture and gripper timestamps differ.', 'INVALID_CAPTURE')
  const gripToReference = grip ? checked(matrix, grip.gripper_to_camera.head_camera) : null
  const targetMatrices = resolveTargets(spec.targets, gripToReference)
  const frames = []
  for (const id of spec.source_cameras) {
    const source = checked(view, data.views[id])
    if (source.width !== middle.width || source.height !== middle.height || !equal(source.K, K)) {
      throw new RgbdVideoGenError('Selected evidence cameras must share middle-camera resolution and intrinsics.', 'INVALID_CAPTURE')
    }
    if (grip && gripToReference && !equal(multiply(source.camera_to_reference,
      checked(matrix, grip.gripper_to_camera[id])), gripToReference)) {
      throw new RgbdVideoGenError('Capture camera and gripper transforms are inconsistent.', 'INVALID_CAPTURE')
    }
    const bytes = await image(source.image_path)
    if (bytes[0] !== 0xff || bytes[1] !== 0xd8 || bytes[2] !== 0xff) {
      throw new RgbdVideoGenError('Capture evidence must contain JPEG images.', 'INVALID_IMAGE')
    }
    frames.push({ views: { head_camera: { image: `data:image/jpeg;base64,${bytes.toString('base64')}`,
      camera_to_world: source.camera_to_reference } } })
  }
  return { stamp: data.stamp, sources: frames.length, targets: targetMatrices.length,
    scene: { format: 'rgbd_videogen_scene_v1', reference_camera: 'head_camera',
      cameras: { head_camera: { width: middle.width, height: middle.height, K } },
      frames: [...frames, ...targetMatrices.map(transform => ({ views: { head_camera: { image: null, camera_to_world: transform } } }))],
      metadata: { capture_stamp: data.stamp, coordinate_frame: 'captured_head_camera_opencv',
        source_cameras: spec.source_cameras, targets: spec.targets.map(target => ({ frame: target.frame, label: target.label ?? null })),
        pose_source: grip?.pose_source ?? 'rig_calibration', predicted: true } } }
}
