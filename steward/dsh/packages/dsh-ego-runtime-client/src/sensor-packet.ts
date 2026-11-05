/** Validation shared by robot camera and encoder-derived pose responses. */
import { EgoError, type JsonObject } from '@steward/dsh-ego'

/** Decode an object from a sensor response.
 * @param value - untrusted response field.
 * @returns object fields.
 */
export function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new EgoError('Invalid camera sensor object')
  return value as Record<string, unknown>
}

/** Decode a finite square matrix.
 * @param value - untrusted response field.
 * @param size - required row and column count.
 * @returns matrix coefficients.
 */
export function matrix(value: unknown, size: number): number[][] {
  if (!Array.isArray(value) || value.length !== size) throw new EgoError('Invalid camera calibration matrix')
  return value.map((row: unknown) => {
    if (!Array.isArray(row) || row.length !== size || !row.every(v => typeof v === 'number' && Number.isFinite(v))) {
      throw new EgoError('Invalid camera calibration matrix')
    }
    return row as number[]
  })
}

function rigidTransform(value: unknown): number[][] {
  const result = matrix(value, 4)
  const rotation = result.slice(0, 3)
  if (result[3]!.some((v, i) => Math.abs(v - (i === 3 ? 1 : 0)) > 1e-9)
    || rotation.some((row, i) => rotation.some((other, j) =>
      Math.abs(row.slice(0, 3).reduce((sum, v, k) => sum + v * other[k]!, 0) - (i === j ? 1 : 0)) > 1e-9))) {
    throw new EgoError('Invalid rigid camera transform')
  }
  const [a, b, c] = rotation
  const determinant = a![0]! * (b![1]! * c![2]! - b![2]! * c![1]!)
    - a![1]! * (b![0]! * c![2]! - b![2]! * c![0]!) + a![2]! * (b![0]! * c![1]! - b![1]! * c![0]!)
  if (Math.abs(determinant - 1) > 1e-9) throw new EgoError('Camera rotation must be right-handed')
  return result
}

function intrinsics(value: unknown) {
  const view = object(value)
  if (typeof view.width !== 'number' || !Number.isSafeInteger(view.width) || view.width <= 0
    || typeof view.height !== 'number' || !Number.isSafeInteger(view.height) || view.height <= 0) {
    throw new EgoError('Invalid camera calibration dimensions')
  }
  const K = matrix(view.K, 3)
  if (K[0]![0]! <= 0 || K[1]![1]! <= 0 || K[2]!.some((v, i) => v !== (i === 2 ? 1 : 0))) {
    throw new EgoError('Invalid camera pixel intrinsics')
  }
  return { width: view.width, height: view.height, K, camera_to_reference: rigidTransform(view.camera_to_reference) }
}

/** Decode fixed robot calibration, stripping absolute poses and other unsupported fields.
 * @param value - untrusted Ego API response.
 * @returns RGB rig intrinsics and fixed relative extrinsics.
 */
export function parseCameraCalibration(value: unknown): JsonObject {
  const packet = object(value), views = object(packet.views)
  const ids = Object.keys(views)
  if (packet.reference_camera !== 'head_camera' || packet.calibration_source !== 'rig_calibration'
    || packet.camera_axes !== 'opencv_x_right_y_down_z_forward'
    || ids.length === 0 || !ids.includes('head_camera') || !ids.every(id => /^[a-z][a-z0-9_]*$/.test(id))) throw new EgoError('Invalid camera rig calibration')
  return { reference_camera: packet.reference_camera, calibration_source: packet.calibration_source,
    camera_axes: packet.camera_axes, views: Object.fromEntries(ids.map(id => [id, intrinsics(views[id])])) }
}

/** Decode robot-owned gripper pose metadata without carrying extra fields into the transcript.
 * @param value - untrusted Ego API result.
 * @returns timestamped joint readings and gripper-to-base/camera transforms.
 */
export function parseGripperPose(value: unknown): JsonObject {
  const pose = object(value), cameras = object(pose.gripper_to_camera), joints = object(pose.joint_positions)
  const ids = Object.keys(cameras)
  if (typeof pose.stamp !== 'number' || !Number.isFinite(pose.stamp) || pose.stamp < 0
    || pose.pose_source !== 'joint_encoders_and_robot_model' || pose.gripper_frame !== 'gripper'
    || pose.gripper_axes !== 'x_forward_y_left_z_up' || pose.camera_axes !== 'opencv_x_right_y_down_z_forward'
    || ids.length === 0 || !ids.includes('head_camera') || !ids.every(id => /^[a-z][a-z0-9_]*$/.test(id))
    || !['lift', 'reach'].every(id => typeof joints[id] === 'number' && Number.isFinite(joints[id]))) {
    throw new EgoError('Invalid gripper pose sensor response')
  }
  const transform = (value: unknown) => {
    const result = matrix(value, 4)
    if (result[3]!.some((v, i) => v !== (i === 3 ? 1 : 0))) throw new EgoError('Invalid gripper pose transform')
    return result
  }
  return { stamp: pose.stamp, pose_source: pose.pose_source, gripper_frame: pose.gripper_frame,
    gripper_axes: pose.gripper_axes, camera_axes: pose.camera_axes,
    joint_positions: { lift: joints.lift, reach: joints.reach },
    gripper_to_base: transform(pose.gripper_to_base),
    gripper_to_camera: Object.fromEntries(ids.map(id => [id, transform(cameras[id])])) }
}
