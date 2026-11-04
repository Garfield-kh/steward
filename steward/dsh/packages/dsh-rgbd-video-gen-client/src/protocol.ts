/** Validation of rgbd_video_gen wire metadata and durable request identities. */
import { z } from 'zod'
import { RgbdVideoGenError, type JsonObject } from '@steward/dsh-rgbd-video-gen'

/** Lossless JSON object. */
export const object = z.record(z.string(), z.json())
/** POST identity required by every rgbd_video_gen mutation. */
export const requestKey = z.string().regex(/^[A-Za-z0-9._:-]{8,128}$/)
const id = z.string().regex(/^j_[A-Za-z0-9_-]{1,96}$/)
/** Closed job states supported by polling. */
export const jobState = z.object({ job_id: id, state: z.enum(['queued', 'running', 'succeeded', 'failed', 'cancelled']) })
/** Generation fields and server-documented limits. */
export const options = z.object({
  num_source_views: z.number().int().min(1).max(8), max_target_views: z.number().int().min(1).max(24),
  num_inference_steps: z.number().int().min(1).max(100), seed: z.number().int().nonnegative().safe(),
  output_fps: z.number().int().min(1).max(60), flow_dtype: z.enum(['bf16', 'fp32']),
})
const filename = z.string().max(200).regex(/^[A-Za-z0-9][A-Za-z0-9_.-]*$/)
  .refine(value => !value.endsWith('.') && !/^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)/i.test(value))
/** Required integrity fields; output URLs are checked against the requested job separately. */
export const result = z.object({
  job_id: id, state: z.literal('succeeded'), metrics: object, params: object, scene: object, meta: object,
  artifacts: z.array(z.object({ kind: z.string(), filename, bytes: z.number().int().nonnegative().safe(),
    sha256: z.string().regex(/^[0-9a-f]{64}$/), media_type: z.string() })).min(1),
  artifact_urls: z.record(z.string(), z.string()),
})
/** Immutable operation record; payload contains digests rather than image contents. */
export const savedRequest = z.object({ version: z.literal(1), endpoint: z.string(),
  kind: z.enum(['submit', 'upload', 'demo', 'cancel']), request_key: requestKey,
  idempotency_key: z.string().uuid(), payload: object }).strict()

/** Validate data received from a tool, file or HTTP response.
 * @param schema - fields required by this consumer.
 * @param value - untrusted input.
 * @returns validated value, or a structured metadata failure.
 */
export function parse<T>(schema: z.ZodType<T>, value: unknown): T {
  const checked = schema.safeParse(value)
  if (!checked.success) throw new RgbdVideoGenError(`Invalid rgbd_video_gen data: ${checked.error.message}`, 'INVALID_METADATA')
  return checked.data
}

/** Project fields without replacing explicit nulls.
 * @param value - decoded object.
 * @param fields - fields to retain.
 * @returns selected fields.
 */
export function pick(value: JsonObject, fields: readonly string[]): JsonObject {
  return Object.fromEntries(fields.filter(key => key in value).map(key => [key, value[key]]))
}
