// Offline dependency adapters only. Load the actual plugin source with Node's TS
// stripper; no package install, provider, HTTP, persistent runtime or simulation.
import { registerHooks, stripTypeScriptTypes } from 'node:module'
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const root = process.env.STEWARD_TEST_REFLECTION_ROOT
  ? pathToFileURL(process.env.STEWARD_TEST_REFLECTION_ROOT.replace(/\/$/, '') + '/')
  : new URL('../packages/dsh-ego-reflection/src/', import.meta.url)
const stub = source => `data:text/javascript,${encodeURIComponent(source)}`
const modules = {
  '@deepseek-ai/schemastery': stub(`
    function schema(parse = x => x) {
      let fallback, lower = -Infinity, upper = Infinity;
      const s = value => {
        const result = parse(value === undefined ? fallback : value);
        if (typeof result === 'number' && (result < lower || result > upper)) throw Error('range');
        return result;
      };
      s.default = value => { fallback = value; return s; };
      s.min = value => { lower = value; return s; };
      s.max = value => { upper = value; return s; };
      return s;
    }
    export default {
      string: () => schema(), natural: () => schema(), array: () => schema(),
      object: fields => schema(value => Object.fromEntries(
        Object.entries(fields).map(([key, field]) => [key, field(value?.[key])]))),
    };
  `),
  '@deepseek-ai/dsh-llm': stub(`
    export const createUserMessage = value => ({ role: 'user', ...value });
    export const ReasoningEffortId = value => value;
  `),
  '@deepseek-ai/dsh-tools': stub('export const defineTool = value => value;'),
}

registerHooks({
  resolve(specifier, context, next) {
    if (specifier === '@steward/dsh-ego-reflection') {
      return { url: new URL('index.ts', root).href, shortCircuit: true }
    }
    if (modules[specifier]) return { url: modules[specifier], shortCircuit: true }
    if (context.parentURL?.startsWith(root.href) && specifier.startsWith('./') && specifier.endsWith('.js')) {
      return { url: new URL(specifier.replace(/\.js$/, '.ts'), context.parentURL).href, shortCircuit: true }
    }
    return next(specifier, context)
  },
  load(url, context, next) {
    if (url.startsWith(root.href) && url.endsWith('.ts')) {
      return { format: 'module', source: stripTypeScriptTypes(readFileSync(new URL(url), 'utf8'),
        { mode: 'transform' }), shortCircuit: true }
    }
    return next(url, context)
  },
})
