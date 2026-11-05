/** Validation of remote JSON, tool requests and portable artifact paths. */
import { z } from 'zod'
import { NovelAnythingError } from '@steward/dsh-novelanything'

/** Bounded remote identity syntax shared by durable request records. */
export const identity = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/)
/** Open remote metadata preserving fields needed by downstream consumers. */
export const object = z.record(z.string(), z.json())
/** Supported preprocessing conventions. */
export const dataset = z.enum(['re10k', 'scannetpp', 'dl3dv', 'mp3d'])
const indices = z.array(z.number().int().nonnegative()).refine(a => new Set(a).size === a.length)
/** Generation request: exactly one source, both frame overrides or neither. */
export const submission = z.object({ sample_id: identity.optional(), upload_id: identity.optional(),
  context: indices.min(2).max(16).optional(), target: indices.min(1).max(6).optional(), metrics: z.boolean(),
}).strict().refine(v => Number(v.sample_id !== undefined) + Number(v.upload_id !== undefined) === 1,
  'Choose exactly one sample_id or upload_id').refine(v => (v.context === undefined) === (v.target === undefined),
  'Set both context and target, or neither').refine(v => !v.context?.some(i => v.target?.includes(i)),
  'Context and target must be disjoint')
/** Remote job state; unknown states must not poll indefinitely or admit result downloads. */
export const job = z.object({ id: identity, status: z.enum(['queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted']) }).passthrough()
/** Successful result artifact descriptor. Hash and byte count are optional in this API. */
export const artifact = z.object({ path: z.string(), url: z.string(),
  bytes: z.number().int().nonnegative().optional(), sha256: z.string().regex(/^[a-fA-F0-9]{64}$/).optional(),
}).passthrough()
/** Durable operation receipt. A reservation without a reply has an unknown remote outcome. */
export const receipt = z.object({ version: z.literal(1), fingerprint: z.string(), payload: object,
  response: object.optional() }).strict()
/** Local sensor export before saving or uploading; image bytes never enter model output. */
export const prepared = z.object({ archive_base64: z.string(), bytes: z.number().int().positive(),
  num_source_views: z.number().int().min(2).max(16), context: indices.min(2).max(16), target: indices.length(1),
  metadata: object,
}).refine(value => value.context.length === value.num_source_views && value.target[0] === value.num_source_views
  && value.context.every((index, i) => index === i)
  && value.metadata.model === 'NovelAnything' && value.metadata.camera_id === 'head_camera'
  && Array.isArray(value.metadata.input_modalities) && value.metadata.input_modalities.length === 1
  && value.metadata.input_modalities[0] === 'rgb' && value.metadata.label === 'gripper_view_prediction', 'Invalid middle-camera gripper scene metadata')

/** Validate untrusted data and retain a stable failure code.
 * @param schema - expected JSON schema.
 * @param value - remote, saved or tool data.
 * @returns parsed value.
 */
export function parse<T>(schema: z.ZodType<T>, value: unknown): T {
  const result = schema.safeParse(value)
  if (!result.success) throw new NovelAnythingError(result.error.message, 'INVALID_DATA')
  return result.data
}

/** Admit portable relative artifact paths and reject Windows aliases and reserved metadata names.
 * @param value - path from a remote artifact descriptor.
 * @returns unchanged portable path.
 */
export function artifactPath(value: string): string {
  if (!value || value.split('/').some(part => !part || part === '.' || part === '..'
    || /[\\:%?#<>"|\x00-\x1f]/.test(part) || /[. ]$/.test(part)
    || /^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)/i.test(part))
    || ['manifest.json', 'job.json'].includes(value.split('/')[0]!.toLowerCase())) {
    throw new NovelAnythingError('Unsafe artifact path returned by NovelAnything.', 'INVALID_ARTIFACT')
  }
  return value
}
