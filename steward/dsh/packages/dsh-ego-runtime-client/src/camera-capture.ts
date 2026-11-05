/** Validate sensor packets and publish complete captures without inline images in tool results. */
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { EgoError, type JsonObject } from '@steward/dsh-ego'

import { object, matrix, parseGripperPose } from './sensor-packet.ts'


/**
 * Save a validated packet from the robot's existing cameras; incomplete directories are removed on failure.
 * @param value - untrusted Ego API response.
 * @param directory - configured output root.
 * @param maxBytes - maximum total decoded RGB JPEG size.
 * @param signal - caller cancellation.
 * @returns metadata, manifest path and saved image paths without image bytes or absolute camera poses.
 */
export async function saveCameraCapture(value: unknown, directory: string, maxBytes: number,
  signal?: AbortSignal): Promise<JsonObject> {
  const packet = object(value), views = object(packet.views)
  const cameraIds = Object.keys(views)
  if (typeof packet.stamp !== 'number' || !Number.isFinite(packet.stamp) || packet.stamp < 0
    || packet.reference_camera !== 'head_camera' || packet.calibration_source !== 'rig_calibration'
    || packet.camera_axes !== 'opencv_x_right_y_down_z_forward'
    || cameraIds.length === 0 || !cameraIds.includes('head_camera') || !cameraIds.every(id => /^[a-z][a-z0-9_]*$/.test(id))) {
    throw new EgoError('Invalid synchronized camera sensor packet')
  }
  const gripper_pose = parseGripperPose(packet.gripper_pose)
  if (gripper_pose.stamp !== packet.stamp) throw new EgoError('Camera and gripper sensor timestamps differ')
  let totalBytes = 0
  const decoded = cameraIds.map(id => {
    const view = object(views[id])
    if (typeof view.width !== 'number' || !Number.isSafeInteger(view.width) || view.width <= 0
      || typeof view.height !== 'number' || !Number.isSafeInteger(view.height) || view.height <= 0
      || typeof view.image !== 'string' || !/^data:image\/jpeg;base64,(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(view.image)) {
      throw new EgoError('Invalid camera JPEG or dimensions')
    }
    const K = matrix(view.K, 3), camera_to_reference = matrix(view.camera_to_reference, 4)
    if (K[0]![0]! <= 0 || K[1]![1]! <= 0 || K[2]![2] !== 1
      || camera_to_reference[3]!.some((v, i) => v !== (i === 3 ? 1 : 0))) {
      throw new EgoError('Invalid camera calibration')
    }
    const encoded = view.image.slice('data:image/jpeg;base64,'.length)
    if (totalBytes + Math.floor(encoded.length * 3 / 4) - (encoded.endsWith('==') ? 2 : encoded.endsWith('=') ? 1 : 0) > maxBytes) {
      throw new EgoError('Camera capture exceeds cameraMaxBytes')
    }
    const bytes = Buffer.from(encoded, 'base64')
    if (bytes[0] !== 0xff || bytes[1] !== 0xd8 || bytes[2] !== 0xff) throw new EgoError('Camera image is not a JPEG')
    totalBytes += bytes.length
    return { id, bytes, metadata: { width: view.width, height: view.height, K, camera_to_reference } }
  })
  signal?.throwIfAborted()
  const root = resolve(directory)
  await mkdir(root, { recursive: true })
  const captureDir = await mkdtemp(join(root, 'capture-'))
  try {
    const saved: JsonObject = {}
    for (const view of decoded) {
      const filename = `${view.id}.jpg`
      await writeFile(join(captureDir, filename), view.bytes, { signal, flag: 'wx' })
      saved[view.id] = { ...view.metadata, image_path: join(captureDir, filename), bytes: view.bytes.length }
    }
    const result: JsonObject = { stamp: packet.stamp, reference_camera: packet.reference_camera,
      camera_axes: packet.camera_axes, calibration_source: packet.calibration_source, views: saved, gripper_pose }
    const manifest = join(captureDir, 'capture.json')
    await writeFile(manifest, JSON.stringify(result, null, 2) + '\n', { signal, flag: 'wx' })
    signal?.throwIfAborted()
    return { ...result, manifest_path: manifest }
  } catch (error) {
    await rm(captureDir, { recursive: true, force: true })
    throw error
  }
}
