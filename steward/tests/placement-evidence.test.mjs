import assert from 'node:assert/strict'
import { test } from 'node:test'
import { egoTools } from '../dsh/packages/dsh-tool-ego/src/index.ts'

const exec = { signal: new AbortController().signal }

test('place evidence survives skill_run wait, async run and skill_status rendering', async () => {
  const postcondition = { ok: null, status: 'unknown', reason: 'fresh observation missing' }
  const job = { id: 'j', skill: 'place', state: 'succeeded', message: 'release executed', log: [], postcondition }
  const ego = { run: async () => job, wait: async () => job, job: async () => job }
  const tools = egoTools(ego)
  for (const [name, args] of [['skill_run', { receipt: 'r' }], ['skill_run', { receipt: 'r', wait: false }],
    ['skill_status', { job: 'j' }]]) {
    const tool = tools.find(t => t.name === name)
    const result = await tool.execute(args, exec)
    assert.deepEqual(result.postcondition, postcondition)
    assert.equal(result.state, 'succeeded') // transport completion is separate from outcome evidence
  }
})

test('jobs without evidence retain the parent response shape', async () => {
  const job = { id: 'j', skill: 'navigate', state: 'succeeded', message: '', log: [] }
  const tool = egoTools({ job: async () => job }).find(t => t.name === 'skill_status')
  assert.deepEqual(await tool.execute({ job: 'j' }, exec),
    { job: 'j', skill: 'navigate', state: 'succeeded', decision: { status: 'succeeded' } })
})
