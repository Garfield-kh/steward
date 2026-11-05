import assert from 'node:assert/strict'
import { test } from 'node:test'
import { apply, candidateLine, Config, failureOf, instructionOf, QUESTIONS, RECURRING_QUESTIONS, reflectionText } from '@steward/dsh-ego-reflection'

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

const LIMITS = { candidateLimit: 6, failureMaxChars: 400 }

test('a first reflection lists the facts, then asks the generic questions', () => {
  const text = reflectionText({ tool: 'ego_query', arguments: { category: 'box' }, message: 'no box in the ego state' },
    { instruction: '把拼图盒里的拼图倒在桌上', history: [], ordinal: 1 }, REFLECTION, LIMITS)
  const lines = text.split('\n')
  assert.deepEqual(lines.slice(0, 4), ['【反思】ego_query 失败：no box in the ego state', '操作员的原始指令：「把拼图盒里的拼图倒在桌上」',
    '失败归类：感知缺失：需要的物体不在 Ego State 里。常见对策：到能看清目标的位置环顾。', '可能就是指令所指、或名字存疑的其他物体（对照 box）：'])
  assert.ok(lines[4].startsWith('- e:5 picture'))
  assert.equal(lines[5], '这是这条指令下的第 1 次反思。下一次行动之前，先用文字逐条回答：')
  assert.deepEqual(lines.slice(6, 6 + QUESTIONS.length), QUESTIONS)
  for (const q of RECURRING_QUESTIONS) assert.ok(!lines.includes(q))
})

test('a later reflection quotes the earlier failures and asks to review the earlier hypotheses', () => {
  const text = reflectionText({ tool: 'reflect', arguments: {}, message: '只匹配上 9 块' },
    { instruction: '拼图', history: ['skill_run：pick failed: missed'], ordinal: 2 }, { ...REFLECTION, cause: { key: 'unknown', label: '', advice: '' } }, LIMITS)
  const lines = text.split('\n')
  assert.deepEqual(lines.slice(0, 4), ['【反思】你遇到的困难：只匹配上 9 块', '操作员的原始指令：「拼图」', '这条指令下之前的失败：',
    '- skill_run：pick failed: missed'])
  assert.ok(!text.includes('失败归类'))
  assert.ok(lines.includes(RECURRING_QUESTIONS[0]))
})

test('without Ego evidence the reflection still quotes the failure and the instruction', () => {
  const text = reflectionText({ tool: 'skill_run', arguments: {}, message: 'x'.repeat(50) }, { history: [], ordinal: 1 },
    { error: 'connect ECONNREFUSED' }, { candidateLimit: 6, failureMaxChars: 40 })
  assert.deepEqual(text.split('\n').slice(0, 3), [`【反思】skill_run 失败：${'x'.repeat(40)}…`,
    '操作员的原始指令：（本轮没有收到操作员消息）', '（无法取得 Ego 证据：connect ECONNREFUSED）'])
})

function harness(config = {}) {
  const handlers = {}
  const reports = []
  const registered = []
  const ctx = {
    ego: { reflect: async (report) => { reports.push(report); return REFLECTION } },
    on: (name, handler) => { handlers[name] = handler },
    effect: (fn) => fn(),
    tools: { register: (tool) => { registered.push(tool); return () => {} } },
  }
  apply(ctx, Config({ instructionMarker: '操作员任务：', ...config }))
  const agent = {}
  const claim = (text, kind = 'user') => handlers['agent/inbox/claimed']({ agent, message: { source: { kind }, content: [{ type: 'text', text }] }, turn: 1 })
  const run = (name, args, result, extra = {}) => handlers['tools/post-execute'](
    { name, arguments: args, agent, signal: new AbortController().signal, rootCallId: 'call-1', parent: {}, ...extra }, result,
    async () => ({ kind: 'accept' }))
  const ask = (args) => registered[0].execute(args, { agent, signal: new AbortController().signal })
  return { reports, claim, run, ask, registered }
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
  assert.match(message.content[0].text, /操作员的原始指令：「把拼图倒在桌上」/)
  assert.deepEqual(h.reports, [{ tool: 'ego_query', arguments: { category: 'box' }, message: 'no box in the ego state' }])
})

test('a failed skill_run is reported with the parameters of the receipt it ran', async () => {
  const h = harness()
  await h.run('ego_predict', { skill: 'pick', target: 'e:5' }, ok({ receipt: 'r7', feasible: true, issues: [] }))
  await h.run('skill_run', { receipt: 'r7' }, ok({ skill: 'pick', state: 'failed', message: 'missed the grasp' }))
  assert.deepEqual(h.reports, [{ tool: 'skill_run', arguments: { receipt: 'r7', params: { skill: 'pick', target: 'e:5' } },
    message: 'pick failed: missed the grasp' }])
})

test('repeated failures and failures beyond the budget pass without a reflection until the next instruction', async () => {
  const h = harness({ maxPerInstruction: 2 })
  const fail = (category) => h.run('ego_query', { category }, ok({ objects: [] }))
  assert.equal((await fail('box')).additionalContexts.length, 1)
  assert.equal((await fail('box')).additionalContexts, undefined)
  assert.equal((await fail('lid')).additionalContexts.length, 1)
  assert.equal((await fail('cup')).additionalContexts, undefined)
  h.claim('操作员任务：再试一次')
  assert.equal((await fail('box')).additionalContexts.length, 1)
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
  const top = { parent: undefined }
  assert.equal((await h.run('run_code', { code: '...' }, failed('Error: no box'), { ...top, rootCallId: 'call-1' })).additionalContexts, undefined)
  const decision = await h.run('run_code', { code: '...' }, failed('Error: only 9 pieces matched'), { ...top, rootCallId: 'call-2' })
  assert.match(decision.additionalContexts[0].content[0].text, /^【反思】run_code 失败：Error: only 9 pieces matched/)
  assert.deepEqual(h.reports.map(r => r.tool), ['ego_query', 'run_code'])
})

test('the reflect tool answers a reported difficulty with the history of the instruction', async () => {
  const h = harness()
  assert.equal(h.registered[0].name, 'reflect')
  h.claim('操作员任务：拼好拼图')
  await h.run('skill_run', { receipt: 'r1' }, ok({ skill: 'pick', state: 'failed', message: 'missed' }))
  const text = await h.ask({ difficulty: '桌上只找到 14 块，盒子上写着 16 片', name: 'puzzle_piece' })
  assert.match(text, /^【反思】你遇到的困难：桌上只找到 14 块/)
  assert.match(text, /- skill_run：pick failed: missed/)
  assert.match(text, /第 2 次反思/)
  assert.deepEqual(h.reports.at(-1), { tool: 'reflect', message: '桌上只找到 14 块，盒子上写着 16 片', name: 'puzzle_piece', size: undefined })
})
