/**
 * Ego Harness safety policy for robot motion.
 *
 * - `tools/pre-execute`: a `skill_run` call is allowed only when its receipt exists on the
 *   robot, is feasible, unused, unexpired and within the receipt's own `maxRisk` (falling back to
 *   the plugin `maxRisk`); otherwise the call is denied with the reason, so the model re-plans.
 * - `ctx.tools.guard`: a monotonic final denial of `skill_run` without a receipt argument, which
 *   later policy listeners cannot undo.
 * - `/estop`: a human command that stops all motion without a model turn.
 *
 * The Ego Runtime re-checks every receipt when it runs; this policy is the Harness-side layer
 * of the two independent layers described in steward.md §6.6.
 * @module @steward/dsh-robot-safety
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type {} from '@deepseek-ai/dsh-commands'
import type { PreToolDecision, ToolExecution } from '@deepseek-ai/dsh-tools'
import type { EgoService, PredictionReceipt } from '@steward/dsh-ego'

export const name = 'steward-robot-safety'
export const inject = ['tools', 'ego']

/** Validated plugin configuration. */
export interface Config {
  /**
   * Fallback risk ceiling when a receipt omits `maxRisk`. The Ego Runtime stamps each receipt with
   * its own limit (pick may be higher than place); this value does not override that stamp.
   */
  maxRisk: number
}

export const Config: z<Config> = z.object({
  maxRisk: z.number().min(0).max(1).default(0.35),
})

const RUN_TOOL = 'skill_run'

function receiptArgument(args: unknown): string | undefined {
  if (typeof args !== 'object' || args === null) return undefined
  const value = (args as { receipt?: unknown }).receipt
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

/**
 * Decide whether a receipt may be executed.
 * @param receipt - receipt read from the robot.
 * @param maxRisk - configured Harness fallback when the receipt has no `maxRisk`.
 * @returns the denial reason, or undefined when execution may proceed.
 */
export function receiptDenial(receipt: PredictionReceipt, maxRisk: number): string | undefined {
  if (receipt.used) return `receipt ${receipt.id} was already executed; call ego_predict again`
  if (!receipt.feasible) return `receipt ${receipt.id} is not feasible: ${receipt.issues.join('; ')}`
  if (receipt.expired || (receipt.now !== undefined && receipt.now > receipt.expiresAt)) {
    return `receipt ${receipt.id} expired at t=${receipt.expiresAt}; call ego_predict again`
  }
  const limit = typeof receipt.maxRisk === 'number' ? receipt.maxRisk : maxRisk
  if (receipt.risk > limit) {
    return `receipt ${receipt.id} has risk ${receipt.risk} above the limit ${limit}: ${receipt.issues.join('; ')}`
  }
  return undefined
}

/**
 * Pre-execution decision for one tool call.
 * @param ego - robot Ego service.
 * @param exec - the tool execution under review.
 * @param maxRisk - configured Harness fallback risk limit.
 * @returns a deny decision, or undefined to delegate to the next policy.
 */
export async function reviewExecution(ego: EgoService, exec: Readonly<ToolExecution>, maxRisk: number):
  Promise<PreToolDecision | undefined> {
  if (exec.name !== RUN_TOOL) return undefined
  const id = receiptArgument(exec.arguments)
  if (id === undefined) return { kind: 'deny', reason: 'skill_run needs a receipt from ego_predict' }
  let receipt: PredictionReceipt
  try {
    receipt = await ego.receipt(id, exec.signal)
  } catch (error) {
    return { kind: 'deny', reason: `receipt ${id} could not be verified: ${(error as Error).message}` }
  }
  const reason = receiptDenial(receipt, maxRisk)
  return reason === undefined ? undefined : { kind: 'deny', reason }
}

export function apply(ctx: Context, config: Config): void {
  ctx.on('tools/pre-execute', async (exec, next) => (await reviewExecution(ctx.ego, exec, config.maxRisk)) ?? next())
  ctx.tools.guard(exec => exec.name === RUN_TOOL && receiptArgument(exec.arguments) === undefined
    ? 'skill_run needs a receipt from ego_predict'
    : undefined)
  ctx.inject(['commands'], (commandCtx) => {
    commandCtx.commands.register({
      name: 'estop',
      description: 'Stop all robot motion immediately',
      handler: async () => {
        const result = await ctx.ego.stop()
        return { kind: 'success', text: `Robot motion stopped${result.stopped ? ` (interrupted ${String(result.stopped)})` : ''}.` }
      },
    })
  })
}
