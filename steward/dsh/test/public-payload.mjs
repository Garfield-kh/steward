// Run the production tool execute/render functions over offline HTTP response fixtures.
// Only dependency registration and the Ego transport are replaced; no model or network.
import { registerHooks, stripTypeScriptTypes } from 'node:module'
import { readFileSync } from 'node:fs'

const source = new URL('../packages/dsh-tool-ego/src/index.ts', import.meta.url)
const stub = code => `data:text/javascript,${encodeURIComponent(code)}`
registerHooks({
  resolve(specifier, context, next) {
    if (specifier === '@deepseek-ai/dsh-tools')
      return { url: stub('export const defineTool = x => x'), shortCircuit: true }
    if (specifier === '@steward/dsh-ego')
      return { url: stub('export const recordingId = x => x'), shortCircuit: true }
    return next(specifier, context)
  },
  load(url, context, next) {
    if (url === source.href) return { format: 'module', shortCircuit: true,
      source: stripTypeScriptTypes(readFileSync(source, 'utf8'), { mode: 'transform' }) }
    return next(url, context)
  },
})
const { egoTools } = await import(source.href)
const fixture = JSON.parse(readFileSync(0, 'utf8'))
const tools = egoTools({ summary: async () => fixture.summary,
  job: async () => fixture.job, run: async () => fixture.job, wait: async () => fixture.job })
const output = {}
for (const name of ['skill_run', 'skill_status', 'ego_state']) {
  const tool = tools.find(t => t.name === name)
  const args = { job: fixture.job.id, receipt: fixture.job.receipt }
  const value = await tool.execute(args, { signal: new AbortController().signal })
  output[name] = tool.output.render(args, value)[0].text
}
process.stdout.write(JSON.stringify(output))
