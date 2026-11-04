// Run a shipped DSH profile in an isolated home and workspace.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
const cliPackage = createRequire(import.meta.url).resolve('@deepseek-ai/dsh/package.json')
const cli = join(dirname(cliPackage), 'lib/bin.js')

export async function launchProfile(args, cwd, env) {
  const child = spawn(process.execPath, [cli, ...args], { cwd, env, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] })
  let stdout = '', stderr = '', timedOut = false
  child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8')
  child.stdout.on('data', chunk => { stdout += chunk }); child.stderr.on('data', chunk => { stderr += chunk })
  const timer = setTimeout(() => { timedOut = true; child.kill() }, 180000)
  try {
    const code = await new Promise((ok, fail) => { child.once('error', fail); child.once('close', ok) })
    assert.equal(timedOut, false, `DSH timed out: ${stderr}`)
    assert.equal(code, 0, `${stderr}\n${stdout}`)
    return stdout
  } finally { clearTimeout(timer) }
}
