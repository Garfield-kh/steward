/**
 * Ego Harness reflection on failed robot steps.
 *
 * In the physical world a step usually fails because the scene differs from what the model
 * assumed, not because the program was wrong: perception names objects by shape and appearance
 * (the box of an instruction can be in the Ego-State as `picture`), objects lie hidden, turned or
 * out of place, and a retry of the same call fails the same way. Reflection gives the model the
 * facts (the operator's instruction, the failure, earlier failures under the same instruction, the
 * Ego Runtime's cause and naming evidence from `ctx.ego.reflect`) and then asks generic questions
 * (goal conditions, expected versus observed, several physical causes, the cheapest evidence for
 * each, preconditions to create first, a checked plan) so that the model applies its own knowledge
 * of the physical world. The questions name dimensions to examine, never a remedy for a task.
 *
 * - `agent/inbox/claimed`: an operator message (source `user`) becomes the instruction quoted by
 *   later reflections of that agent and resets its reflection history and budget.
 * - `tools/post-execute`: a failure of a configured robot tool, or the failure of a program tool
 *   (`run_code`) none of whose nested failures was reflected on, attaches the reflection as an
 *   `additionalContexts` user message of source `ego-reflection`; the outcome is otherwise the
 *   downstream decision. Inside a program the message reaches the model with the program's result.
 * - tool `reflect`: the model reports a difficulty no tool reported and receives the reflection
 *   as the tool result; it counts towards the ordinal and history but is never refused.
 * @module @steward/dsh-ego-reflection
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { createUserMessage, type ContentBlock } from '@deepseek-ai/dsh-llm'
import { defineTool, type PostToolDecision, type ToolDefinition, type ToolExecution, type ToolExecutionResult } from '@deepseek-ai/dsh-tools'
import type { Candidate, EgoService, FailureReport, JsonObject, Reflection } from '@steward/dsh-ego'

declare module '@deepseek-ai/dsh-llm' {
  interface MessageSourceMap {
    /** Reflection on a failed robot step attached by `@steward/dsh-ego-reflection`. */
    'ego-reflection': { kind: 'ego-reflection' }
  }
}

export const name = 'steward-ego-reflection'
export const inject = ['tools', 'ego']

/** Validated plugin configuration. */
export interface Config {
  /** Tools whose failures are reflected on. */
  tools: string[]
  /** Program tools (`run_code`) whose own failure is reflected on when no nested failure of the same program was. */
  programTools: string[]
  /** Reflections per operator instruction; later failures pass without one. */
  maxPerInstruction: number
  /** Objects listed per reflection (the call's targets first, then candidates). */
  candidateLimit: number
  /** Text after the last occurrence of this marker in an operator message is the instruction; empty: the whole message. */
  instructionMarker: string
  /** Longest instruction quoted, in characters; a longer one is cut at its end. */
  instructionMaxChars: number
  /** Longest failure text quoted, in characters. */
  failureMaxChars: number
  /** Earlier failures under one instruction quoted by a reflection. */
  historyLimit: number
  /** Longest quoted earlier failure, in characters. */
  historyMaxChars: number
  /** Prediction receipts remembered to name the target of a failed `skill_run`. */
  receiptMemory: number
}

export const Config: z<Config> = z.object({
  tools: z.array(z.string()).default(['ego_query', 'ego_predict', 'skill_run', 'skill_status', 'ego_inspect',
    'ego_match_face', 'ego_annotate', 'ego_explore']),
  programTools: z.array(z.string()).default(['run_code']),
  maxPerInstruction: z.natural().min(1).default(4),
  candidateLimit: z.natural().min(1).default(6),
  instructionMarker: z.string().default(''),
  instructionMaxChars: z.natural().min(40).default(800),
  failureMaxChars: z.natural().min(40).default(400),
  historyLimit: z.natural().default(6),
  historyMaxChars: z.natural().min(40).default(160),
  receiptMemory: z.natural().min(1).default(64),
})

/** A failed robot step as reflection reads it. */
export interface Failure {
  tool: string
  arguments: JsonObject
  message: string
}

function record(value: unknown): JsonObject | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as JsonObject : undefined
}

function blockText(content: readonly ContentBlock[]): string {
  return content.map(block => block.type === 'text' ? block.text : '').join('\n').trim()
}

/**
 * The failure a tool result reports, if any: an error result, an infeasible prediction, a
 * failed skill, or a lookup that found nothing.
 * @param tool - tool name.
 * @param args - the call's arguments.
 * @param result - the dispatch outcome.
 * @returns the failure, or undefined for a successful step.
 */
export function failureOf(tool: string, args: unknown, result: Readonly<ToolExecutionResult>): Failure | undefined {
  const argumentsOf = record(args) ?? {}
  if (result.isError) return { tool, arguments: argumentsOf, message: blockText(result.content) || 'the call failed' }
  const value = record(result.value)
  if (value === undefined) return undefined
  if (tool === 'ego_predict' && value.feasible === false) {
    const issues = Array.isArray(value.issues) ? value.issues.join('; ') : ''
    return { tool, arguments: argumentsOf, message: `${String(value.skill ?? argumentsOf.skill)} is not feasible: ${issues}` }
  }
  if ((tool === 'skill_run' || tool === 'skill_status') && value.state === 'failed') {
    return { tool, arguments: argumentsOf, message: `${String(value.skill)} failed: ${String(value.message ?? '')}` }
  }
  if (tool === 'ego_query' && Array.isArray(value.objects) && value.objects.length === 0) {
    const what = typeof argumentsOf.category === 'string' ? argumentsOf.category : 'matching object'
    const near = typeof argumentsOf.near === 'string' ? ` near ${argumentsOf.near}` : ''
    return { tool, arguments: argumentsOf, message: `no ${what}${near} in the ego state` }
  }
  return undefined
}

/**
 * The instruction an operator message carries.
 * @param text - the message text.
 * @param marker - text after its last occurrence is the instruction; empty: the whole text.
 * @param maxChars - longest instruction kept; a longer one keeps its end.
 * @returns the instruction, or undefined for a message without text.
 */
export function instructionOf(text: string, marker: string, maxChars: number): string | undefined {
  const at = marker ? text.lastIndexOf(marker) : -1
  const instruction = (at >= 0 ? text.slice(at + marker.length) : text).trim()
  if (!instruction) return undefined
  return instruction.length > maxChars ? `…${instruction.slice(-maxChars)}` : instruction
}

function clip(text: string, maxChars: number): string {
  return text.length > maxChars ? `${text.slice(0, maxChars)}…` : text
}

function percent(p: number): string {
  return `${Math.round(p * 100)}%`
}

/**
 * One line describing an object and the evidence behind its name.
 * @param c - the object.
 * @returns the line, without a leading bullet.
 */
export function candidateLine(c: Candidate): string {
  const facts: string[] = []
  if (c.label) facts.push(`你标注为 ${c.label}`)
  if (c.perceived && Object.keys(c.perceived).length) {
    facts.push('感知类别 ' + Object.entries(c.perceived).map(([k, p]) => `${k} ${percent(p)}`).join(', '))
  }
  if (c.segmenter && Object.keys(c.segmenter).length) {
    facts.push('分割网络 ' + Object.entries(c.segmenter)
      .map(([k, s]) => `${k} ${s.score.toFixed(2)}${s.prompts.length ? `〔${s.prompts.join('/')}〕` : ''}`).join(', '))
  }
  if (c.extentM) facts.push(`尺寸 ${c.extentM.map(v => v.toFixed(2)).join('×')} m`)
  if (c.color) facts.push(`颜色 ${c.color}`)
  if (c.on) facts.push(`在 ${c.on} 上`)
  facts.push(`位置 (${c.position.slice(0, 3).map(v => v.toFixed(2)).join(', ')})`)
  let line = `${c.id} ${c.role && c.role !== c.category ? `${c.category}（role ${c.role}）` : c.category}：${facts.join('；')}`
  if (c.doubts.length) line += `。存疑：${c.doubts.join('；')}`
  if (c.score !== undefined && c.why?.length) line += `。吻合 ${c.score.toFixed(2)}：${c.why.join('；')}`
  return line
}

/** Where the reflection stands within one operator instruction. */
export interface ReflectionContext {
  /** The operator's instruction, if known. */
  instruction?: string
  /** Earlier failures under the same instruction, oldest first. */
  history: string[]
  /** 1 for the first reflection under the instruction. */
  ordinal: number
}

/** Name of the tool the model calls to reflect on a difficulty no tool reported. */
export const REFLECT_TOOL = 'reflect'

/**
 * Questions that make the model reason about a difficulty from its own knowledge of the physical
 * world. They name the dimensions to examine, never a remedy for a particular task.
 */
export const QUESTIONS = [
  '一、目标：把指令要求的最终状态拆成可以逐条检验的条件。现在哪些已经满足、哪些还没有？你凭什么判断？',
  '二、现象：这一步你预期会发生什么，实际发生了什么？差异具体在哪里？不要只复述报错。',
  '三、原因：设想一个人在真实环境里亲手做这件事，会在哪里遇到困难？用你对物理世界的常识列出至少三个可能的原因，不要只想一个。逐项检查：',
  '   - 感知：物体是否被认错或叫错（感知按外观和形状命名，不知道操作员的叫法），是否只看到了一部分、被别的东西挡住、只看到了某一面；',
  '   - 物体的状态：位置、朝向、哪一面朝上、是否相互叠压或遮挡、数量是否齐全、是否离开了原来的地方；',
  '   - 动作：够不着、抓不稳、会碰撞、放置的空间不够；',
  '   - 任务理解：你对指令的理解和操作员的意思是否一致。',
  '四、证据：每个可能的原因，用什么观察能以最小代价证实或排除？先核实，再行动。',
  '五、前提：要顺利完成最终动作，世界先要处于什么状态（看得清、拿得到、彼此分开、齐全、朝向合适……）？是否应当先创造这些条件，而不是直接做最终动作？',
  '六、计划：选出证据最支持的原因，写出下一步做什么、预期看到什么结果、在程序里怎样检验；不要原样重复失败的做法。',
  '七、核实后确定只有操作员才能解决时（例如东西不在了、无法到达），调用 report_problem 说明。',
]

/** Questions added from the second reflection under one instruction on. */
export const RECURRING_QUESTIONS = [
  '回顾：之前反思时的假设，哪些被证实、哪些被否定？这几次失败有没有共同的模式？'
    + '有没有一直没考虑到的原因，或者需要退回到更早的步骤重新做？',
]

/**
 * The reflection text for one failure or reported difficulty.
 * @param failure - the failed step; `tool` is {@link REFLECT_TOOL} for a difficulty the model reported.
 * @param context - the instruction, the earlier failures under it and the reflection's ordinal.
 * @param evidence - the Ego Runtime's reflection evidence, or the reason it is unavailable.
 * @param config - limits.
 * @returns the message text.
 */
export function reflectionText(failure: Failure, context: ReflectionContext,
  evidence: Reflection | { error: string }, config: Pick<Config, 'candidateLimit' | 'failureMaxChars'>): string {
  const what = clip(failure.message, config.failureMaxChars)
  const lines = [failure.tool === REFLECT_TOOL ? `【反思】你遇到的困难：${what}` : `【反思】${failure.tool} 失败：${what}`]
  lines.push(context.instruction ? `操作员的原始指令：「${context.instruction}」` : '操作员的原始指令：（本轮没有收到操作员消息）')
  if (context.history.length) lines.push('这条指令下之前的失败：', ...context.history.map(h => `- ${h}`))
  if ('error' in evidence) {
    lines.push(`（无法取得 Ego 证据：${evidence.error}）`)
  } else {
    if (evidence.cause.key !== 'unknown') lines.push(`失败归类：${evidence.cause.label}。常见对策：${evidence.cause.advice}。`)
    const targets = evidence.targets.slice(0, config.candidateLimit)
    if (targets.length) lines.push('这一步涉及的物体：', ...targets.map(c => `- ${candidateLine(c)}`))
    if (evidence.missing.length) lines.push(`Ego State 里已没有：${evidence.missing.join(', ')}`)
    const others = evidence.candidates.slice(0, Math.max(0, config.candidateLimit - targets.length))
    const sought = evidence.sought.name || evidence.sought.size
      ? `（对照 ${[evidence.sought.name, evidence.sought.size?.map(v => `${v} m`).join('×')].filter(Boolean).join('，')}）` : ''
    if (others.length) lines.push(`可能就是指令所指、或名字存疑的其他物体${sought}：`, ...others.map(c => `- ${candidateLine(c)}`))
  }
  lines.push(`这是这条指令下的第 ${context.ordinal} 次反思。下一次行动之前，先用文字逐条回答：`)
  if (context.ordinal > 1) lines.push(...RECURRING_QUESTIONS)
  lines.push(...QUESTIONS)
  lines.push('需要时用 ego_candidates（传入指令里的名字和大小）、ego_inspect、ego_observe 或靠近观察核实，用 ego_annotate 改正物体的名字和可供性。')
  return lines.join('\n')
}

interface AgentState {
  instruction?: string
  reflections: number
  signatures: Set<string>
  history: string[]
  /** `rootCallId`s of programs whose nested failures were already reflected on. */
  roots: Set<string>
}

function signature(failure: Failure): string {
  return `${failure.tool}\u0000${failure.message.replace(/\d+(\.\d+)?/g, '#')}`
}

/**
 * Build the `reflect` tool: the model reports a difficulty no tool reported (a result that does
 * not match the goal, no progress) and receives the same evidence and questions.
 * @param ask - produces the reflection text for a difficulty.
 * @returns the tool definition.
 */
export function reflectTool(ask: (difficulty: string, name: string | undefined, size: number[] | undefined,
  exec: ToolExecution) => Promise<string>): ToolDefinition {
  return defineTool({
    name: REFLECT_TOOL,
    description: 'Reflect before trying again when the task is not progressing although no tool failed: a result does not '
      + 'match the goal, something expected is missing, or the same approach keeps falling short. Describe the difficulty; '
      + 'returns the operator\'s instruction, earlier failures under it, the robot\'s evidence on the objects involved, '
      + 'and the questions to answer before the next step. Print the returned text and answer it.',
    parameters: {
      difficulty: { type: 'string', required: true, description: 'What is going wrong, with the observations that show it.' },
      name: { type: 'string', description: 'The object the instruction means, as a category word, when one is involved.' },
      size: { type: 'array', items: { type: 'number' }, description: 'Its approximate dimensions in metres, if known.' },
    },
    output: { schema: { type: 'string' }, render: (_a, v) => [{ type: 'text', text: v }] },
    async execute(args, exec) {
      return ask(args.difficulty, args.name, args.size, exec)
    },
  })
}

export function apply(ctx: Context, config: Config) {
  const ego: EgoService = ctx.ego
  const tools = new Set(config.tools)
  const programs = new Set(config.programTools)
  const states = new WeakMap<Agent, AgentState>()
  const fresh = (): AgentState => ({ reflections: 0, signatures: new Set(), history: [], roots: new Set() })
  const detached = fresh()
  const receipts = new Map<string, JsonObject>()
  const stateOf = (agent: Agent | undefined): AgentState => {
    if (agent === undefined) return detached
    let state = states.get(agent)
    if (state === undefined) states.set(agent, state = fresh())
    return state
  }

  const reflect = async (state: AgentState, failure: Failure, report: FailureReport, signal: AbortSignal): Promise<string> => {
    let evidence: Reflection | { error: string }
    try {
      evidence = await ego.reflect(report, signal)
    } catch (error) {
      evidence = { error: error instanceof Error ? error.message : String(error) }
    }
    const text = reflectionText(failure, { instruction: state.instruction, history: [...state.history], ordinal: state.reflections + 1 },
      evidence, config)
    state.reflections += 1
    state.history.push(`${failure.tool}：${clip(failure.message, config.historyMaxChars)}`)
    while (state.history.length > config.historyLimit) state.history.shift()
    return text
  }

  ctx.on('agent/inbox/claimed', ({ agent, message }) => {
    if (message.source.kind !== 'user') return
    const instruction = instructionOf(blockText(message.content), config.instructionMarker, config.instructionMaxChars)
    if (instruction === undefined) return
    Object.assign(stateOf(agent), { ...fresh(), instruction })
  })

  ctx.effect(() => ctx.tools.register(reflectTool(async (difficulty, sought, size, exec) => {
    const failure: Failure = { tool: REFLECT_TOOL, arguments: {}, message: difficulty }
    return reflect(stateOf(exec.agent), failure, { tool: REFLECT_TOOL, message: difficulty, name: sought, size }, exec.signal)
  })))

  ctx.on('tools/post-execute', async (exec: ToolExecution, result, next): Promise<PostToolDecision> => {
    const decision = await next()
    const program = programs.has(exec.name) && exec.parent === undefined
    if (!tools.has(exec.name) && !program) return decision
    const value = result.isError ? undefined : record(result.value)
    if (exec.name === 'ego_predict' && typeof value?.receipt === 'string') {
      receipts.set(value.receipt, record(exec.arguments) ?? {})
      while (receipts.size > config.receiptMemory) receipts.delete(receipts.keys().next().value!)
    }
    const state = stateOf(exec.agent)
    const failure = program
      ? (result.isError && !state.roots.has(exec.rootCallId) ? failureOf(exec.name, {}, result) : undefined)
      : failureOf(exec.name, exec.arguments, result)
    if (failure === undefined || exec.signal.aborted) return decision
    const key = signature(failure)
    if (state.reflections >= config.maxPerInstruction || state.signatures.has(key)) return decision
    state.signatures.add(key)
    if (!program) state.roots.add(exec.rootCallId)
    const receipt = typeof failure.arguments.receipt === 'string' ? receipts.get(failure.arguments.receipt) : undefined
    const text = await reflect(state, failure, {
      tool: failure.tool,
      arguments: receipt ? { ...failure.arguments, params: receipt } : failure.arguments,
      message: failure.message,
    }, exec.signal)
    const message = createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'ego-reflection' } })
    return { ...decision, additionalContexts: [...decision.additionalContexts ?? [], message] }
  })
}
