/** Bounded text-only continuation through the host's existing in-turn inbox. */
import type { Context } from '@deepseek-ai/cordis'
import type { UserMessage } from '@deepseek-ai/dsh-llm'

const CONTINUE = '任务尚未明确结束，请继续调用工具执行。若已完成，请明确说明“任务已完成”；'
  + '若确认无法继续，请明确说明“确认无法继续”并结束。不要只解释计划或代码修正。'

/** Explicit declarations only: a quoted example or an unfinished task is not an exit. */
function explicitlyEnded(text: string): boolean {
  const prose = text.replace(/```[\s\S]*?```/g, '').replace(/^\s*>.*$/gm, '')
  return prose.split(/[\n。！!]/).some(line => {
    const sentence = line.trim().replace(/^[-*]\s+/, '').replace(/\*\*/g, '')
    return /^(?:任务(?:已经|已)?(?:全部)?完成|(?:我)?(?:已经|已)完成(?:全部|所有)?任务|(?:我)?确认无法继续|(?:我)?无法继续|(?:我)?放弃(?:任务)?|结束任务)(?:[，,:：；;]|$)/.test(sentence)
      || /^(?:(?:the\s+)?task\s+(?:(?:is|has been)\s+)?(?:complete|completed)|done|I\s+(?:(?:have\s+)?completed\s+the\s+task|cannot continue|give up))(?:[.,:;]|$)/i.test(sentence)
  })
}

/** Register on the public stop boundary; never restart a process or reset its budgets.
 * The host still owns admission of every subsequent step and cancellation.
 */
export function installContinuation(ctx: Context, limit: number, makeMessage: (text: string) => UserMessage): void {
  if (limit <= 0) return
  const sessions = new WeakMap<object, {
    turn: number; step: number; consecutive: number; eligible: boolean; promptedStep: number; exhausted: boolean
  }>()
  ctx.on('session/event', (session, event) => {
    if (event.type === 'turn/start') {
      sessions.set(session, { turn: event.data.turn, step: 0, consecutive: 0,
        eligible: false, promptedStep: -1, exhausted: false })
      return
    }
    if (event.type !== 'assistant/message') return
    const state = sessions.get(session)
    if (state === undefined || state.turn !== event.data.turn || event.data.step <= state.step) return
    state.step = event.data.step
    const blocks = event.data.message.content
    const hasTools = blocks.some(block => block.type === 'tool-call')
    const text = blocks.map(block => block.type === 'text' ? block.text : '').join('\n').trim()
    const stoppedByBudget = event.data.stream?.some(record => record.type === 'chunk'
      && record.chunk.type === 'finish' && record.chunk.reason.kind !== 'stop'
      && record.chunk.reason.kind !== 'tool-calls')
    // Keep max-tokens sticky for this turn, just as the outer loop does.
    state.exhausted ||= Boolean(stoppedByBudget || event.data.interrupted)
    state.consecutive = hasTools ? 0 : state.consecutive + 1
    state.eligible = !hasTools && text.length > 0 && !explicitlyEnded(text) && !state.exhausted
  })
  ctx.on('agent/turn-stopping', ({ agent, turn, signal }) => {
    const state = sessions.get(agent.session)
    if (signal.aborted || state === undefined || state.turn !== turn || !state.eligible
      || state.consecutive >= limit || state.promptedStep === state.step) return
    state.promptedStep = state.step
    agent.steer(makeMessage(CONTINUE))
  })
}
