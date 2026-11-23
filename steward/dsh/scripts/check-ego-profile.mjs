// Inspect the installed robot plugin before spending a model turn.
import { createHash } from 'node:crypto'
import { createRequire } from 'node:module'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const args = process.argv.slice(2)
const value = flag => args.includes(flag) ? args[args.indexOf(flag) + 1] : undefined
const expected = value('--expect') ?? (process.env.STEWARD_ASSEMBLY_PROFILE || 'off')
const profile = process.env.STEWARD_ASSEMBLY_PROFILE || 'off'
if (!['off', 'toy'].includes(expected) || profile !== expected) {
  throw new Error(`Harness profile ${JSON.stringify(profile)} does not match expected ${JSON.stringify(expected)}`)
}
const profileDir = value('--profile-dir')
const require = createRequire(profileDir ? resolve(profileDir, 'package.json') : import.meta.url)
const modulePath = value('--module') ? resolve(value('--module')) : require.resolve('@steward/dsh-tool-ego')
const plugin = await import(pathToFileURL(modulePath).href)
const tools = []
const ego = { assemblyProfile: async () => profile, predict: async (skill, params) => ({ id: 'profile-check', skill, params, plan: {}, issues: [],
  feasible: false, withinRiskLimit: false, risk: 1, maxRisk: 0.35, expiresAt: 0 }) }
await plugin.apply({ ego, tools: { register(tool) { tools.push(tool); return () => {} } } })
const prediction = tools.find(tool => tool.name === 'ego_predict')
const common = ['regrasp_upright', 'assemble_onto', 'fasten_onto', 'seat_screw', 'drive_screw', 'flip_assembly',
  'navigate', 'pick', 'place', 'pour', 'open', 'close']
const expectedSkills = [...common, profile === 'toy' ? 'complete_assembly' : 'insert_screw_into_hole'].sort()
const skills = prediction?.parameters?.properties?.skill?.enum
if (!Array.isArray(skills) || JSON.stringify([...skills].sort()) !== JSON.stringify(expectedSkills)
  || tools.length !== (profile === 'toy' ? 20 : 21)
  || tools.some(tool => tool.name === 'ego_screw_holes') !== (profile !== 'toy')) {
  throw new Error(`Loaded plugin ${modulePath} exposes the wrong ${profile} interface; rebuild and reinstall the robot profile`)
}
for (const skill of expectedSkills) {
  await prediction.execute({ skill }, { signal: new AbortController().signal })
}
const surface = tools.map(tool => ({ name: tool.name, description: tool.description,
  parameters: tool.parameters, output: tool.output.schema }))
console.log(JSON.stringify({ profile, modulePath, skills, tools: tools.map(tool => tool.name),
  surfaceSha256: createHash('sha256').update(JSON.stringify(surface)).digest('hex') }))
