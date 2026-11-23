import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { registeredSurface } from './registered-profile-surface.mjs'

const fixture = JSON.parse(readFileSync(new URL('../../tests/expected/registered-tool-surfaces.json', import.meta.url)))
const profile = process.env.STEWARD_ASSEMBLY_PROFILE === 'toy' ? 'toy' : 'default'
for (const artifact of ['src/index.ts', 'lib/index.js']) {
  test(`${profile} ${artifact}: registered schemas, descriptions, SDK and validation match the reference`, async () => {
    const actual = await registeredSurface(new URL(`../packages/dsh-tool-ego/${artifact}`, import.meta.url).href, true)
    assert.deepEqual(actual, fixture[profile])
    assert.equal(actual.validation.complete_assembly, profile === 'toy')
    assert.equal(actual.validation.insert_screw_into_hole, profile !== 'toy')
  })
}
