// Real schema compilation and plugin registration, with an offline Ego service.
import { registerHooks, stripTypeScriptTypes, createRequire } from 'node:module'
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import { Context } from '@deepseek-ai/cordis'
import { SystemPrompt } from '@deepseek-ai/dsh-system-prompt'
import { ToolRuntime, renderToolsSdk } from '@deepseek-ai/dsh-tools'
const require = createRequire(import.meta.url)
const dependency = Object.fromEntries(['@deepseek-ai/dsh-tools', '@deepseek-ai/schemastery', '@deepseek-ai/dsh-llm',
  '@steward/dsh-ego', '@steward/dsh-novelanything', '@steward/dsh-rgbd-video-gen'].map(name => [name, pathToFileURL(require.resolve(name)).href]))
registerHooks({
  resolve(name, context, next) {
    if (dependency[name]) return { url: dependency[name], shortCircuit: true }
    if (context.parentURL?.endsWith('/src/index.ts') && name === './continuation.js') {
      return { url: new URL('./continuation.ts', context.parentURL).href, shortCircuit: true }
    }
    return next(name, context)
  },
  load(url, context, next) {
    if (url.endsWith('.ts')) return { format: 'module', shortCircuit: true,
      source: stripTypeScriptTypes(readFileSync(new URL(url), 'utf8'), { mode: 'transform' }) }
    return next(url, context)
  },
})
export async function registeredSurface(moduleUrl, includeBundles = false) {
  const plugin = await import(moduleUrl)
  const calls = []
  const ego = { assemblyProfile: async () => process.env.STEWARD_ASSEMBLY_PROFILE === 'toy' ? 'toy' : 'off', predict: async (skill, params) => {
    calls.push({ skill, params })
    return { id: 'receipt', skill, params, plan: {}, risk: 0, maxRisk: 0.35, feasible: true, withinRiskLimit: true, issues: [] }
  } }
  const ctx = new Context()
  new SystemPrompt(ctx, SystemPrompt.Config({}))
  const registry = new ToolRuntime(ctx)
  const tools = []
  await plugin.apply({ ego, tools: { register(tool) { tools.push(tool); return registry.register(tool) } } })
  const prediction = registry.get('ego_predict')
  const validation = {}
  for (const skill of ['complete_assembly', 'regrasp_upright', 'assemble_onto', 'fasten_onto', 'seat_screw', 'drive_screw',
    'flip_assembly', 'insert_screw_into_hole', 'navigate', 'pick', 'place', 'pour', 'open', 'close', '__invalid__']) {
    try {
      await prediction.execute({ skill }, { signal: new AbortController().signal })
      validation[skill] = true
    } catch (error) {
      if (error.name !== 'ToolArgsError') throw error
      validation[skill] = false
    }
  }
  const result = { tools: tools.map(tool => ({ name: tool.name, description: tool.description,
    parameters: tool.parameters, output: tool.output.schema })), validation,
    sdk: renderToolsSdk(tools.map(tool => ({ ...tool, output: tool.output.schema }))) }
  if (includeBundles) {
    const packagesUrl = new URL('../../', moduleUrl)
    const face = moduleUrl.endsWith('.ts') ? 'src/index.ts' : 'lib/index.js'
    const extra = []
    for (const [name, factory] of [['dsh-tool-ego-capabilities', 'capabilityTools'],
      ['dsh-ego-reflection', 'reflectTool'], ['dsh-tool-rgbd-video-gen', 'rgbdVideoGenTools'],
      ['dsh-tool-novelanything', 'novelAnythingTools']]) {
      const module = await import(new URL(`${name}/${face}`, packagesUrl).href)
      if (name === 'dsh-ego-reflection') result.reflection = { config: module.Config({}), programQuestions: module.PROGRAM_QUESTIONS }
      const definitions = factory === 'reflectTool' ? [module[factory](async () => '')] : module[factory]({})
      for (const tool of definitions) {
        registry.register(tool)
        extra.push({ name: tool.name, description: tool.description, parameters: tool.parameters, output: tool.output.schema })
      }
    }
    result.extraTools = extra
    const world = await import(new URL(`dsh-tool-world/${face}`, packagesUrl).href)
    const worldTools = world.worldTools({})
    for (const tool of worldTools) registry.register(tool)
    result.worldTools = worldTools.map(tool => ({ name: tool.name, description: tool.description,
      parameters: tool.parameters, output: tool.output.schema }))
  }
  await ctx.fiber.dispose()
  return result
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  const moduleUrl = process.argv[2] ? pathToFileURL(process.argv[2]).href : import.meta.resolve('@steward/dsh-tool-ego')
  const result = await registeredSurface(moduleUrl, process.argv.includes('--all'))
  console.error(JSON.stringify({ moduleUrl, profile: process.env.STEWARD_ASSEMBLY_PROFILE ?? 'off', tools: result.tools.length }))
  console.log(JSON.stringify(result))
}
