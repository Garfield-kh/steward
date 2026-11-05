# Agent Note: Re-materialize an externally deleted sandbox private temp directory

Status: implemented

## Problem

`sandbox-local` materializes one private temp directory per live session/workspace pair (`mkdtempSync(join(tmpdir(), 'dsh-'))`) and caches the resulting `AclTempCapability` (directory + `tempWriteSid`) in `tempCapabilities`. Every later sandbox launch for that pair reuses the cached `temp.dir` without checking that the directory still exists. The Windows ACL runner (`sandbox-windows-acl/src/runner.ts`, `requireDirectory`) validates `--temp` and fails closed with `--temp is not an existing directory` — correct behaviour at the runner boundary — but nothing on the provider side ever re-materializes the grant.

Any external removal of that one directory therefore bricks the sandbox for the rest of the session: Temp cleaners, Storage Sense, disk-cleanup tools, or a reboot sweeper deleting `dsh-*` entries each trigger it. Observed in a 40+ minute agent session: after a third-party cleaner removed the live `dsh-q247zS` directory, every `run_code` call failed deterministically with `sandbox-unavailable` (6+ consecutive failures across trivial and real programs, `read-only` and `workspace-write`), while the 40 leftover `dsh-*` directories from earlier sessions were untouched — no bulk-cleanup code path is needed, a single per-directory deletion suffices. The agent had no recovery path and the session had to be resumed by the operator after a manual `mkdir`.

## Proposal

In `materializeAclGrant`, when reusing an existing `tempCapabilities` entry, verify `existsSync(temp.dir)`. If the directory is gone, revoke the stale capability (its ACEs died with the directory) and materialize a fresh directory with a new `tempWriteSid`, replacing the map entry. Later launches then pick up the new pair transparently and the session survives the deletion.

Recreating the directory bare (plain `mkdir`) is not sufficient as a fix: the private-temp ACE grant lived on the deleted directory, so a recreated path would pass `requireDirectory` but run without the intended confinement. The capability pair must be rebuilt, not just the path. This keeps the failure closed at the runner and puts recovery where the grant is owned.

## Alternatives considered

- **Recreate only the directory path.** Passes the runner's existence check but silently loses the private-temp ACEs — the exact confinement the SID exists to enforce.
- **Re-verify on every call from the runner.** The runner is fail-loud by design at the trust boundary; it cannot rebuild a grant it does not own. Recovery belongs in the provider that materialized the grant.
- **Keep failing and require an operator.** Status quo: a deterministic, session-wide sandbox outage from one deleted directory, with the only workaround (manual `mkdir`) leaving undefined ACE state.

## Verification

- Delete a live pair's temp directory externally, then run a `run_code` program: it succeeds, and the new directory carries a fresh `tempWriteSid` grant.
- Dispose revokes the replacement capability's ACEs as it does today.
- Existing sessions without deletion behave byte-for-byte as before (no extra syscalls on the happy path beyond one `existsSync`).

## Implementation

`LocalSandboxProvider.materializeAclGrant` (`packages/sandbox/sandbox-local/src/index.ts`) now re-checks `existsSync` on a cached temp capability before reuse. A missing directory deletes the stale map entry and disposes its grant (best-effort: the ACEs died with the directory), then materialization falls through to a fresh `mkdtemp` + `tempWriteSid` pair that later launches reuse as usual. `tests/acl-grants.spec.ts` covers the out-of-band deletion: the next confinement launches on a new directory and SID, the stale grant is disposed, the replacement is reused, and provider dispose still removes the replacement.
