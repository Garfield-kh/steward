#!/usr/bin/env bash
# Steward compatibility and regression checks (steward.md §7 P8):
#   1. Python: scene model, assets, WSim, Ego Runtime, Observer and the end-to-end mission.
#   2. DSH plugins: TypeScript build against the pinned @deepseek-ai packages, unit tests, and the
#      cross-language mission driven through the Ego API provider and robot tools.
#   3. Composed-configuration snapshots of the ego and world profiles. A diff after an upstream
#      upgrade lists new or changed rows (for example new upstream tools) for review.
# Set UPDATE_SNAPSHOTS=1 to accept the current composed configuration.
set -euo pipefail

STEWARD="$(cd "$(dirname "$0")/.." && pwd)"
DSH="$STEWARD/dsh/node_modules/.bin/dsh"
VAR="${STEWARD_DSH_VAR:-$STEWARD/var/dsh}"
SNAPSHOTS="$STEWARD/compat/snapshots"

echo "== python"
(cd "$STEWARD" && .venv/bin/python -m pytest -q)

echo "== dsh plugins"
(cd "$STEWARD/dsh" && pnpm run build && node --test test/*.test.mjs)

echo "== composed configuration"
if [ ! -d "$VAR/robot-1/profiles/ego" ] || [ ! -d "$VAR/world/profiles/world" ] || [ ! -d "$VAR/world/profiles/world-task" ]; then
  "$STEWARD/scripts/setup-dsh-profiles.sh"
fi
mkdir -p "$SNAPSHOTS"
status=0
for pair in "robot-1:ego" "world:world" "world:world-task"; do
  home="${pair%%:*}"
  profile="${pair##*:}"
  current="$(mktemp)"
  DSH_HOME="$VAR/$home" "$DSH" --profile "$profile" --dump-config > "$current"
  if [ "${UPDATE_SNAPSHOTS:-0}" = "1" ] || [ ! -f "$SNAPSHOTS/$profile.yml" ]; then
    cp "$current" "$SNAPSHOTS/$profile.yml"
    echo "snapshot $profile.yml written"
  elif ! diff -u "$SNAPSHOTS/$profile.yml" "$current"; then
    echo "composed configuration of profile $profile changed; review, then rerun with UPDATE_SNAPSHOTS=1" >&2
    status=1
  else
    echo "profile $profile unchanged"
  fi
  rm -f "$current"
done
exit "$status"
