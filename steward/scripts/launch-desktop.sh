#!/usr/bin/env bash
# Launch Steward + Ego Harness and open the control panel in the browser.
# Usage: launch-desktop.sh [scenario]
#   With no scenario argument, shows a macOS chooser (or uses STEWARD_SCENARIO).
set -euo pipefail
# Finder launches apps without a locale; Python and the dialogs need UTF-8 for the Chinese scenario text.
export LANG="${LANG:-en_US.UTF-8}" LC_ALL="${LC_ALL:-en_US.UTF-8}"

STEWARD="$(cd "$(dirname "$0")/.." && pwd)"
STEWARD_BIN="$STEWARD/.venv/bin/steward"
LOG_DIR="${STEWARD_LOG_DIR:-$HOME/Library/Logs/Steward}"
LOG="$LOG_DIR/desktop-launcher.log"
WCA_URL="http://127.0.0.1:7400"
EGO_URL="http://127.0.0.1:7420"
CONSOLE_URL="$WCA_URL/console"
TITLE="Steward"

mkdir -p "$LOG_DIR"

notify() {
  /usr/bin/osascript -e "display notification \"$1\" with title \"$TITLE\"" >/dev/null 2>&1 || true
}

fail() {
  /usr/bin/osascript -e "display dialog \"$1\" with title \"$TITLE\" buttons {\"OK\"} default button \"OK\" with icon caution" >/dev/null 2>&1 || true
  echo "FAIL: $1" >>"$LOG"
  exit 1
}

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG"
}

pick_scenario() {
  if [[ -n "${1:-}" ]]; then
    echo "$1"
    return
  fi
  if [[ -n "${STEWARD_SCENARIO:-}" ]]; then
    echo "$STEWARD_SCENARIO"
    return
  fi
  local choice
  choice="$(/usr/bin/osascript <<'APPLESCRIPT' 2>/dev/null || true
set opts to {"拼图·狗 (puzzle-dog)", "拼图·猫 (puzzle-cat)", "拼图·长城 (puzzle-greatwall)", "家务整理 (phase0-chores)", "循环搬运 (phase0-shuttle)", "杯子放到桌上 (phase0-kitchen)"}
set picked to choose from list opts with prompt "选择要启动的场景：" default items {"拼图·狗 (puzzle-dog)"} OK button name "启动" cancel button name "取消"
if picked is false then
  return ""
end if
return item 1 of picked
APPLESCRIPT
)"
  [[ -n "$choice" ]] || exit 0
  case "$choice" in
    *puzzle-dog*) echo puzzle-dog ;;
    *puzzle-cat*) echo puzzle-cat ;;
    *puzzle-greatwall*) echo puzzle-greatwall ;;
    *phase0-chores*) echo phase0-chores ;;
    *phase0-shuttle*) echo phase0-shuttle ;;
    *phase0-kitchen*) echo phase0-kitchen ;;
    *) echo "$choice" ;;
  esac
}

wait_http() {
  local url="$1" label="$2" seconds="${3:-90}"
  local i
  for i in $(seq 1 "$seconds"); do
    if /usr/bin/curl -fsS --max-time 1 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  fail "等待 $label 超时（$seconds 秒）：$url"
}

stop_existing() {
  # Ego Harness first, then steward up / ego API children.
  /usr/bin/pkill -f 'dsh --profile ego' 2>/dev/null || true
  /usr/bin/pkill -f 'steward up' 2>/dev/null || true
  /usr/bin/pkill -f 'steward ego' 2>/dev/null || true
  # Free fixed ports if a stale process still holds them.
  for port in 7400 7410 7420; do
    local pids
    pids="$(/usr/sbin/lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      # shellcheck disable=SC2086
      kill $pids 2>/dev/null || true
    fi
  done
  sleep 1
}

scenario_task() {
  local name="$1"
  local file="$STEWARD/var/steward/scenarios/${name}.json"
  if [[ ! -f "$file" ]]; then
    fail "找不到场景文件：$file"
  fi
  "$STEWARD/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["task"])' "$file"
}

robot_prompt() {
  local task="$1"
  "$STEWARD/.venv/bin/python" -c '
from steward.world.chat import ROBOT_BRIEF
import sys
print(ROBOT_BRIEF + sys.argv[1])
' "$task"
}

main() {
  {
    echo "----- $(date '+%Y-%m-%d %H:%M:%S') -----"
    echo "cwd=$STEWARD"
  } >>"$LOG"

  export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin${PATH:+:$PATH}"

  [[ -x "$STEWARD_BIN" ]] || fail "找不到 steward 可执行文件：${STEWARD_BIN}。请先在 steward/ 运行 uv sync。"
  [[ -f "$STEWARD/.env" ]] || fail "找不到 $STEWARD/.env（需要 DEEPSEEK_API_KEY）。"
  [[ -x "$STEWARD/dsh/node_modules/.bin/dsh" ]] || fail "找不到 dsh。请先运行 steward/scripts/setup-dsh-profiles.sh。"

  local scenario
  scenario="$(pick_scenario "${1:-}")"
  log "scenario=$scenario"
  notify "正在启动 Steward（${scenario}）…"

  stop_existing

  local task prompt
  task="$(scenario_task "$scenario")"
  prompt="$(robot_prompt "$task")"

  (
    cd "$STEWARD"
    # shellcheck disable=SC1091
    set -a; . .env; set +a
    exec "$STEWARD_BIN" up --scenario "$scenario" --speed max --freeze-idle
  ) >>"$LOG" 2>&1 &
  local steward_pid=$!
  log "steward_pid=$steward_pid"

  wait_http "$WCA_URL/console" "控制台" 90
  wait_http "$EGO_URL/ego/state" "Ego API" 90

  (
    cd "$STEWARD"
    exec ./scripts/run-ego.sh "$prompt"
  ) >>"$LOG" 2>&1 &
  log "ego_pid=$!"

  /usr/bin/open "$CONSOLE_URL"
  notify "Steward 已启动 · 控制面板已打开"
  log "opened $CONSOLE_URL"
}

main "$@"
