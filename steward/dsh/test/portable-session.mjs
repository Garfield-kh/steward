// Normalize host-dependent recording metadata before checking a real Session into fixtures.
export function portableSession(raw, root, endpoint) {
  const rows = raw.trim().split('\n').map(line => JSON.parse(line))
  const session = rows[0].id
  function walk(value, key = '') {
    if (['time', 'createdAt'].includes(key) && typeof value === 'number') return 0
    if (Array.isArray(value)) return value.map(item => walk(item))
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([name, item]) => [name, walk(item, name)]))
    if (typeof value !== 'string') return value
    if (value.startsWith('{') || value.startsWith('[')) {
      let embedded
      try { embedded = JSON.parse(value) } catch { /* ordinary model text may start with punctuation */ }
      if (embedded !== undefined) return JSON.stringify(walk(embedded), null, 2)
    }
    return value.replaceAll(session, '{{sessionId}}').replaceAll(endpoint, '{{endpoint}}')
      .replaceAll(root, '{{cwd}}').replaceAll('\\', '/')
      .replace(/(state\/)[a-f0-9]{16}(\/)/g, '$1{{service}}$2')
  }
  return rows.map(row => JSON.stringify(walk(row))).join('\n') + '\n'
}
