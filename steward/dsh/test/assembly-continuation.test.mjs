import assert from 'node:assert/strict'
import { test as nodeTest } from 'node:test'
import { runInNewContext } from 'node:vm'
import { apply, Config } from '@steward/dsh-ego-reflection'

const test = process.env.STEWARD_ASSEMBLY_PROFILE === 'toy' ? nodeTest : nodeTest.skip

const text = value => [{ type: 'text', text: value }]
const tool = [{ type: 'tool-call', id: 'call', name: 'run_code', arguments: {} }]

async function harness(options = {}) {
  const hooks = new Map(), steering = [], calls = []
  const ctx = {
    on(name, fn) { const rows = hooks.get(name) ?? []; rows.push(fn); hooks.set(name, rows) },
    effect: fn => fn(), tools: { register: () => () => {} },
    ego: { assemblyProfile: async () => process.env.STEWARD_ASSEMBLY_PROFILE === 'toy' ? 'toy' : 'off', reflect: async () => ({ error: 'offline' }), summary: async () => ({}) },
  }
  await apply(ctx, Config({ checkpointEvery: 0, idleEvery: 0, ...options }))
  const agent = { session: {}, steer: message => steering.push(message) }
  const controller = new AbortController()
  const emit = async (name, ...args) => {
    let result
    for (const fn of hooks.get(name) ?? []) result = await fn(...args)
    return result
  }
  const event = (type, data) => emit('session/event', agent.session, { type, data })
  const begin = turn => event('turn/start', { turn })
  const message = (content, step, extra = {}) => event('assistant/message', {
    turn: 1, step, message: { content }, stream: [{ type: 'chunk', time: 0,
      chunk: { type: 'finish', reason: { kind: 'stop' } } }], ...extra,
  })
  const stopping = () => emit('agent/turn-stopping', { agent, turn: 1, signal: controller.signal })
  // Match the outer loop's documented boundary: no tool calls -> completed;
  // stopping listeners may steer next-step work through the SAME budget gate.
  async function loop(replies, { maxSteps = Infinity, expired = () => false } = {}) {
    await begin(1)
    let count = 0
    for (const content of replies) {
      if (count >= maxSteps || expired() || controller.signal.aborted) break
      steering.length = 0
      await message(content, ++count)
      if (content.some(block => block.type === 'tool-call')) {
        calls.push('run_code')
        await event('tool/result', { turn: 1, step: count, result: { isError: true } })
        await emit('tools/post-execute', { name: 'run_code', arguments: {}, agent,
          signal: controller.signal, rootCallId: String(count) },
        { isError: true, content: text("SyntaxError: Identifier 'value' has already been declared") },
        async () => ({ kind: 'accept' }))
      } else {
        await stopping()
        if (!steering.length) break
      }
    }
    return count
  }
  return { agent, begin, message, stopping, loop, steering, calls, controller, emit }
}

test('tool error then explanatory text continues to the next tool call', async () => {
  const replies = [tool, text('错误是重复声明；修正是使用局部作用域。'), tool, text('任务已完成。')]
  const before = (await harness({ maxConsecutiveTextReplies: 0 }))
  assert.equal(await before.loop(replies), 2)
  assert.equal(before.calls.length, 1)
  const after = (await harness())
  assert.equal(await after.loop(replies), 4)
  assert.equal(after.calls.length, 2)
})

test('consecutive text replies reach the configured cap and finish normally', async () => {
  assert.equal(await (await harness()).loop(Array.from({ length: 8 }, () => text('继续。'))), 3)
  for (const limit of [1, 2, 3, 5]) {
    const h = (await harness({ maxConsecutiveTextReplies: limit }))
    assert.equal(await h.loop(Array.from({ length: 8 }, () => text('接下来继续执行。'))), limit)
  }
})

test('an actual tool call resets the consecutive-text count', async () => {
  const h = (await harness({ maxConsecutiveTextReplies: 2 }))
  assert.equal(await h.loop([text('继续。'), tool, text('继续。'), tool, text('任务已完成。')]), 5)
})

test('explicit completion and abandonment still end the turn', async () => {
  for (const reply of ['任务已完成。', '我已经完成任务。', '确认无法继续。', '结束任务。',
    'The task is complete.', 'I have completed the task.', 'I cannot continue.', 'I give up.']) {
    assert.equal(await (await harness()).loop([text(reply), tool]), 1, reply)
  }
})

test('negated, quoted and code completion text is not a completion declaration', async () => {
  for (const reply of ['任务尚未完成。', '不要说“任务已完成”。', '> 任务已完成。\n我还在修复。', '```\n任务已完成。\n```\n需要重试。']) {
    assert.equal(await (await harness()).loop([text(reply), tool, text('任务已完成。')]), 3, reply)
  }
})

test('continuation uses the existing step/wall budget and cancellation', async () => {
  assert.equal(await (await harness()).loop([text('修复后继续。'), tool], { maxSteps: 1 }), 1)
  let checks = 0
  assert.equal(await (await harness()).loop([text('修复后继续。'), tool], { expired: () => ++checks > 1 }), 1)
  const h = (await harness()); await h.begin(1); await h.message(text('继续。'), 1)
  h.controller.abort(); await h.stopping(); assert.equal(h.steering.length, 0)
})

test('max-token and interrupted replies do not get a continuation', async () => {
  for (const extra of [{ interrupted: true }, { stream: [{ type: 'chunk', time: 0,
    chunk: { type: 'finish', reason: { kind: 'max-tokens' } } }] }]) {
    const h = (await harness()); await h.begin(1); await h.message(text('继续。'), 1, extra)
    await h.stopping(); assert.equal(h.steering.length, 0)
  }
})

test('terminal tool, duplicate stop notification and other sessions do not spuriously steer', async () => {
  const h = (await harness()); await h.begin(1); await h.message(tool, 1); await h.stopping()
  assert.equal(h.steering.length, 0)
  await h.message(text('继续。'), 2); await h.stopping(); await h.stopping()
  assert.equal(h.steering.length, 1)
  assert.equal(h.steering[0].source.kind, 'ego-continuation')
  await h.emit('agent/turn-stopping', { agent: { session: {}, steer() { throw Error('wrong session') } },
    turn: 1, signal: h.controller.signal })
})

test('new turn resets text-only budget', async () => {
  const h = (await harness({ maxConsecutiveTextReplies: 2 }))
  assert.equal(await h.loop([text('继续。'), text('继续。'), tool]), 2)
  await h.begin(2); await h.message(text('继续。'), 1, { turn: 2 })
  await h.emit('agent/turn-stopping', { agent: h.agent, turn: 2, signal: h.controller.signal })
  assert.equal(h.steering.length, 1)
})

test('redeclaration reflection gives executable local-scope advice', async () => {
  const h = (await harness())
  // Existing reflection middleware takes a downstream continuation.
  const decision = await h.emit('tools/post-execute', { name: 'run_code', arguments: {}, agent: h.agent,
    signal: h.controller.signal, rootCallId: 'first' },
  { isError: true, content: text("SyntaxError: Identifier 'value' has already been declared") },
  async () => ({ kind: 'accept' }))
  assert.match(decision.additionalContexts[0].content[0].text, /\{ \.\.\. \} 块内/)
  assert.match(decision.additionalContexts[0].content[0].text, /继续调用 run_code/)
  // Exercise the suggested scope form in one real JS context without changing the runtime.
  const context = {}
  runInNewContext('const value = 1', context)
  assert.throws(() => runInNewContext('const value = 2', context), /already been declared/)
  assert.equal(runInNewContext('{ const value = 2; value }', context), 2)
  assert.equal(runInNewContext('value', context), 1)
  assert.ok(decision)
})
