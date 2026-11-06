import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  apply, candidateLine, CHECKPOINT_QUESTIONS, checkpointText, Config, DEEP_QUESTIONS, failureOf, IDLE_QUESTIONS, instructionOf,
  RECURRING_QUESTIONS, reflectionText, sceneDigest, SHALLOW_QUESTIONS,
} from '@steward/dsh-ego-reflection'

const ok = (value) => ({ isError: false, value, content: [{ type: 'text', text: JSON.stringify(value) }] })
const failed = (text) => ({ isError: true, content: [{ type: 'text', text }], error: { message: text } })

const LID = {
  id: 'e:5', category: 'picture', role: 'picture', extentM: [0.46, 0.46, 0.02], color: null, on: 'e:2',
  position: [0.6, 0.1, 0.77], lastSeenAgoS: 0.5, perceived: { picture: 0.55, plate: 0.3 },
  segmenter: { picture: { score: 0.8, prompts: ['photo', 'picture'] } },
  doubts: ['the robot has no shape knowledge of a picture: the name comes from appearance alone'],
  score: 1.2, why: ['size fit 1.00 against 0.46x0.46 m'],
}

const REFLECTION = {
  cause: { key: 'perception-missing', label: '感知缺失：需要的物体不在 Ego State 里', advice: '到能看清目标的位置环顾' },
  sought: { name: 'box', size: null },
  targets: [],
  missing: [],
  candidates: [LID],
}

const SUMMARY = {
  self: { x: -0.42, y: 0.59, yawDeg: -90.2, holding: null },
  objects: [
    { id: 'e:4', category: 'dinner table', role: 'dinner table', on: null, existence: 1 },
    { id: 'e:2', category: 'picture', label: 'box', on: 'e:4', existence: 1 },
    ...Array.from({ length: 8 }, (_, i) => ({ id: `e:${10 + i}`, category: 'puzzle_piece', on: i < 7 ? 'e:4' : null,
      existence: i === 7 ? 0.3 : 1 })),
  ],
}

test('failures are read from errors, infeasible receipts, failed skills and empty lookups', () => {
  assert.deepEqual(failureOf('skill_run', { receipt: 'r1' }, failed('receipt r1 expired')),
    { tool: 'skill_run', arguments: { receipt: 'r1' }, message: 'receipt r1 expired' })
  assert.equal(failureOf('ego_predict', { skill: 'pick', target: 'e:5' },
    ok({ receipt: 'r1', skill: 'pick', feasible: false, issues: ['height 0.02 m is outside the lift range'] })).message,
  'pick is not feasible: height 0.02 m is outside the lift range')
  assert.equal(failureOf('skill_run', {}, ok({ skill: 'pick', state: 'failed', message: 'missed' })).message, 'pick failed: missed')
  assert.equal(failureOf('ego_query', { category: 'box' }, ok({ objects: [] })).message, 'no box in the ego state')
  assert.equal(failureOf('ego_predict', { skill: 'pick' }, ok({ receipt: 'r1', feasible: true, issues: [] })), undefined)
  assert.equal(failureOf('ego_query', { category: 'cup' }, ok({ objects: [{ id: 'e:1' }] })), undefined)
})

test('the instruction is the text after the last marker, keeping the end of a long one', () => {
  assert.equal(instructionOf('说明……\n\n操作员任务：把拼图倒在桌上', '操作员任务：', 100), '把拼图倒在桌上')
  assert.equal(instructionOf('把盒子拿来', '操作员任务：', 100), '把盒子拿来')
  assert.equal(instructionOf('abcdefghij'.repeat(5), '', 40), `…${'abcdefghij'.repeat(4)}`)
  assert.equal(instructionOf('操作员任务：  ', '操作员任务：', 100), undefined)
})

test('a candidate line shows its naming evidence, doubts and fit', () => {
  assert.equal(candidateLine(LID), 'e:5 picture：感知类别 picture 55%, plate 30%；分割网络 picture 0.80〔photo/picture〕；'
    + '尺寸 0.46×0.46×0.02 m；在 e:2 上；位置 (0.60, 0.10, 0.77)。存疑：the robot has no shape knowledge of a picture: '
    + 'the name comes from appearance alone。吻合 1.20：size fit 1.00 against 0.46x0.46 m')
})

test('the scene digest groups objects by the name the robot uses, largest group first', () => {
  assert.deepEqual(sceneDigest(SUMMARY, 2), [
    '机器人在 (-0.42, 0.59)，朝向 -90°，手里是空的。',
    '- puzzle_piece ×8（e:10, e:11, e:12, e:13, e:14, e:15 …）：7 个在 e:4 上，1 个不在已知支撑面上；其中 1 个存在概率低于 50%',
    '- box ×1（e:2）：在 e:4 上',
    '- 另有 1 类物体未列出',
  ])
})

const LIMITS = { candidateLimit: 6, failureMaxChars: 400 }
const SHALLOW = { instructions: ['把拼图盒里的拼图倒在桌上'], history: [], ordinal: 1, depth: 'shallow', steps: 3 }

test('a shallow reflection lists the facts, then asks a few questions', () => {
  const text = reflectionText({ tool: 'ego_query', arguments: { category: 'box' }, message: 'no box in the ego state' },
    SHALLOW, REFLECTION, LIMITS)
  const lines = text.split('\n')
  assert.deepEqual(lines.slice(0, 4), ['【反思】ego_query 失败：no box in the ego state', '操作员的原始指令：「把拼图盒里的拼图倒在桌上」',
    '失败归类：感知缺失：需要的物体不在 Ego State 里。常见对策：到能看清目标的位置环顾。', '可能就是指令所指、或名字存疑的其他物体（对照 box）：'])
  assert.ok(lines[4].startsWith('- e:5 picture'))
  assert.equal(lines[5], '这是这条指令下的第 1 次反思。下一次行动之前，用几句话逐条回答：')
  assert.deepEqual(lines.slice(6, 6 + SHALLOW_QUESTIONS.length), SHALLOW_QUESTIONS)
  assert.ok(!text.includes('【宏观】') && !text.includes('当前场景'))
})

test('a deep reflection says why, digests the scene and asks macro, micro and strategy questions', () => {
  const text = reflectionText({ tool: 'reflect', arguments: {}, message: '只匹配上 9 块' }, {
    instructions: ['拼好拼图', '继续'], history: ['skill_run：pick failed: missed'], ordinal: 3, depth: 'deep',
    trigger: '同样的失败又出现了一次', steps: 14, scene: ['- puzzle_piece ×16'],
  }, { ...REFLECTION, cause: { key: 'unknown', label: '', advice: '' }, candidates: [] }, LIMITS)
  const lines = text.split('\n')
  assert.deepEqual(lines.slice(0, 10), ['【深度反思】你遇到的困难：只匹配上 9 块', '为什么深入反思：同样的失败又出现了一次',
    '操作员的消息（从早到晚）：', '- 「拼好拼图」', '- 「继续」', '这条指令下之前的失败：', '- skill_run：pick failed: missed',
    '当前场景（Ego State 摘要，这条指令下已运行 14 个程序）：', '- puzzle_piece ×16',
    '这是这条指令下的第 3 次反思，是一次深度反思。先在思考中充分推敲，再分层逐条作答，每条几句话：'])
  assert.deepEqual(lines.slice(10, 10 + RECURRING_QUESTIONS.length + DEEP_QUESTIONS.length), [...RECURRING_QUESTIONS, ...DEEP_QUESTIONS])
  assert.ok(!text.includes('失败归类'))
})

test('without Ego evidence the reflection still quotes the failure and the instruction', () => {
  const text = reflectionText({ tool: 'skill_run', arguments: {}, message: 'x'.repeat(50) }, { ...SHALLOW, instructions: [] },
    { error: 'connect ECONNREFUSED' }, { candidateLimit: 6, failureMaxChars: 40 })
  assert.deepEqual(text.split('\n').slice(0, 3), [`【反思】skill_run 失败：${'x'.repeat(40)}…`,
    '操作员的原始指令：（本轮没有收到操作员消息）', '（无法取得 Ego 证据：connect ECONNREFUSED）'])
})

test('a checkpoint reviews progress with the scene', () => {
  const lines = checkpointText({ instructions: ['拼好拼图'], steps: 9, since: 8, scene: ['- puzzle_piece ×16'] }).split('\n')
  assert.deepEqual(lines, ['【阶段回顾】这条指令下已运行 9 个程序，自上次反思以来的 8 个程序没有工具报错。', '操作员的原始指令：「拼好拼图」',
    '当前场景（Ego State 摘要）：', '- puzzle_piece ×16', '用几句话逐条回答，再继续：', ...CHECKPOINT_QUESTIONS])
})

const EFFORT = { act: 'low', instruction: 'high', recover: 'high', shallow: 'high', deep: 'max', checkpoint: 'high' }

function harness(config = {}) {
  const handlers = {}
  const reports = []
  const registered = []
  const ctx = {
    ego: { reflect: async (report) => { reports.push(report); return REFLECTION }, summary: async () => SUMMARY },
    on: (name, handler) => { handlers[name] = handler },
    effect: (fn) => fn(),
    tools: { register: (tool) => { registered.push(tool); return () => {} } },
  }
  apply(ctx, Config({ instructionMarker: '操作员任务：', effort: EFFORT, checkpointEvery: 0, idleEvery: 0, ...config }))
  const agent = {}
  const signal = new AbortController().signal
  const claim = (text, kind = 'user') => handlers['agent/inbox/claimed']({ agent, message: { source: { kind }, content: [{ type: 'text', text }] }, turn: 1 })
  const run = (name, args, result, extra = {}) => handlers['tools/post-execute'](
    { name, arguments: args, agent, signal, rootCallId: 'call-1', parent: {}, ...extra }, result,
    async () => ({ kind: 'accept' }))
  const program = (result = ok(null), rootCallId = 'call-1') => run('run_code', { code: '...' }, result, { parent: undefined, rootCallId })
  const ask = (args) => registered[0].execute(args, { agent, signal })
  const effort = async () => (await handlers['agent/request']({ agent, turn: 1, step: 1, signal },
    async () => ({ provider: 'deepseek', model: 'flash', reasoningEffort: 'high' }))).reasoningEffort
  const text = (decision) => decision.additionalContexts.map(m => m.content[0].text).join('\n')
  return { reports, claim, run, program, ask, effort, text, registered }
}

test('a failed robot step gets one reflection message carrying the operator instruction', async () => {
  const h = harness()
  h.claim('说明\n\n操作员任务：把拼图倒在桌上')
  h.claim('后台提醒', 'ego-reflection')
  const decision = await h.run('ego_query', { category: 'box' }, ok({ objects: [] }))
  assert.equal(decision.kind, 'accept')
  const [message] = decision.additionalContexts
  assert.equal(message.role, 'user')
  assert.deepEqual(message.source, { kind: 'ego-reflection' })
  assert.match(message.content[0].text, /^【反思】[^\n]*\n操作员的原始指令：「把拼图倒在桌上」/)
  assert.deepEqual(h.reports, [{ tool: 'ego_query', arguments: { category: 'box' }, message: 'no box in the ego state' }])
})

test('a failed skill_run is reported with the parameters of the receipt it ran', async () => {
  const h = harness()
  await h.run('ego_predict', { skill: 'pick', target: 'e:5' }, ok({ receipt: 'r7', feasible: true, issues: [] }))
  await h.run('skill_run', { receipt: 'r7' }, ok({ skill: 'pick', state: 'failed', message: 'missed the grasp' }))
  assert.deepEqual(h.reports, [{ tool: 'skill_run', arguments: { receipt: 'r7', params: { skill: 'pick', target: 'e:5' } },
    message: 'pick failed: missed the grasp' }])
})

test('a recurring failure and a failure after several shallow reflections are reflected on deeply', async () => {
  const h = harness({ deepAfter: 2 })
  h.claim('操作员任务：拼好拼图')
  const fail = (category) => h.run('ego_query', { category }, ok({ objects: [] }))
  assert.match(h.text(await fail('box')), /^【反思】/)
  const again = h.text(await fail('box'))
  assert.match(again, /^【深度反思】[^\n]*\n为什么深入反思：同样的失败又出现了一次/)
  assert.match(again, /当前场景（Ego State 摘要，这条指令下已运行 0 个程序）：\n机器人在 \(-0\.42, 0\.59\)/)
  assert.equal((await fail('box')).additionalContexts, undefined, 'one deep reflection per recurring failure')
  assert.match(h.text(await fail('lid')), /^【反思】/)
  assert.match(h.text(await fail('cup')), /^【反思】/)
  assert.match(h.text(await fail('bowl')), /^【深度反思】[^\n]*\n为什么深入反思：前 2 次反思之后任务仍然受阻/)
})

test('reflections beyond the budgets pass, and a new operator message resets them but keeps the earlier one', async () => {
  const h = harness({ maxPerInstruction: 2, maxDeepPerInstruction: 0 })
  h.claim('操作员任务：拼好拼图')
  const fail = (category) => h.run('ego_query', { category }, ok({ objects: [] }))
  assert.match(h.text(await fail('box')), /^【反思】/)
  assert.equal((await fail('box')).additionalContexts, undefined, 'no deep budget')
  assert.match(h.text(await fail('lid')), /^【反思】/)
  assert.equal((await fail('cup')).additionalContexts, undefined, 'no budget left')
  h.claim('操作员任务：继续')
  const text = h.text(await fail('box'))
  assert.match(text, /操作员的消息（从早到晚）：\n- 「拼好拼图」\n- 「继续」/)
  assert.match(text, /第 1 次反思/)
})

test('tools outside the configured set and successful steps are left alone', async () => {
  const h = harness()
  assert.deepEqual(await h.run('ego_capture_cameras', {}, failed('boom')), { kind: 'accept' })
  assert.deepEqual(await h.run('ego_state', {}, ok({ objects: [] })), { kind: 'accept' })
  assert.deepEqual(h.reports, [])
})

test('a failed program is reflected on only when none of its nested failures was', async () => {
  const h = harness()
  await h.run('ego_query', { category: 'box' }, ok({ objects: [] }), { rootCallId: 'call-1' })
  assert.equal((await h.program(failed('Error: no box'), 'call-1')).additionalContexts, undefined)
  const decision = await h.program(failed('Error: only 9 pieces matched'), 'call-2')
  assert.match(h.text(decision), /^【反思】run_code 失败：Error: only 9 pieces matched/)
  assert.deepEqual(h.reports.map(r => r.tool), ['ego_query', 'run_code'])
})

test('the reflect tool answers a reported difficulty deeply unless a shallow check is asked for', async () => {
  const h = harness()
  assert.equal(h.registered[0].name, 'reflect')
  h.claim('操作员任务：拼好拼图')
  await h.run('skill_run', { receipt: 'r1' }, ok({ skill: 'pick', state: 'failed', message: 'missed' }))
  const text = await h.ask({ difficulty: '桌上只找到 14 块，盒子上写着 16 片', name: 'puzzle_piece' })
  assert.match(text, /^【深度反思】你遇到的困难：桌上只找到 14 块，盒子上写着 16 片\n为什么深入反思：你报告了一个没有工具报错的困难。/)
  assert.match(text, /- skill_run：pick failed: missed/)
  assert.match(text, /第 2 次反思/)
  assert.ok(text.includes(RECURRING_QUESTIONS[0]))
  assert.deepEqual(h.reports.at(-1), { tool: 'reflect', message: '桌上只找到 14 块，盒子上写着 16 片', name: 'puzzle_piece', size: undefined })
  assert.match(await h.ask({ difficulty: '放下后位置偏了', depth: 'shallow' }), /^【反思】你遇到的困难：放下后位置偏了/)
})

test('a run of programs without reflection ends with a checkpoint; a failing program defers it', async () => {
  const h = harness({ checkpointEvery: 3 })
  h.claim('操作员任务：拼好拼图')
  assert.equal((await h.program()).additionalContexts, undefined)
  assert.equal((await h.program()).additionalContexts, undefined)
  assert.match(h.text(await h.run('ego_query', { category: 'box' }, ok({ objects: [] }), { rootCallId: 'call-3' })), /^【反思】/)
  assert.equal((await h.program(ok(null), 'call-3')).additionalContexts, undefined, 'no checkpoint after a failure')
  assert.equal((await h.program()).additionalContexts, undefined)
  const text = h.text(await h.program())
  assert.match(text, /^【阶段回顾】这条指令下已运行 5 个程序，自上次反思以来的 3 个程序没有工具报错。/)
  assert.match(text, /- puzzle_piece ×8/)
})

test('a run of programs in which the robot does not act ends with a review of what they wait for', async () => {
  const h = harness({ idleEvery: 3 })
  h.claim('操作员任务：拼好拼图')
  await h.program(ok(null), 'call-1')
  await h.run('skill_run', { receipt: 'r1' }, ok({ skill: 'pick', state: 'succeeded' }), { rootCallId: 'call-2' })
  assert.equal((await h.program(ok(null), 'call-2')).additionalContexts, undefined, 'the robot acted')
  assert.equal((await h.program(failed('Error: x'), 'call-3')).additionalContexts.length, 1, 'a failure is reflected on first')
  assert.equal((await h.program()).additionalContexts, undefined)
  const text = h.text(await h.program())
  assert.match(text, /^【阶段回顾】这条指令下已运行 5 个程序，最近 3 个程序里机器人都没有动作（只在观察、计算或打印）。/)
  assert.ok(text.includes(IDLE_QUESTIONS[1]))
  assert.equal((await h.program()).additionalContexts, undefined, 'the run starts again')
})

test('each request thinks as long as the kind of step it follows', async () => {
  const h = harness({ checkpointEvery: 2 })
  assert.equal(await h.effort(), 'high', 'before any step the configured effort stands')
  h.claim('操作员任务：拼好拼图')
  assert.equal(await h.effort(), 'high', 'a new instruction')
  await h.program()
  assert.equal(await h.effort(), 'low', 'a routine program')
  await h.program(failed('Error: boom'), 'call-2')
  assert.equal(await h.effort(), 'high', 'a shallow reflection')
  await h.program(failed('Error: boom'), 'call-3')
  assert.equal(await h.effort(), 'max', 'a deep reflection')
  await h.program(failed('Error: boom'), 'call-4')
  assert.equal(await h.effort(), 'high', 'a failure past its reflections recovers')
  assert.match(h.text(await h.program()), /^【阶段回顾】/)
  assert.equal(await h.effort(), 'high', 'a checkpoint')
  await h.program()
  assert.equal(await h.effort(), 'low')
  const plain = harness({ effort: {} })
  await plain.program()
  assert.equal(await plain.effort(), 'high', 'an empty effort leaves the request as configured')
})
