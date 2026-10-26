#!/usr/bin/env bash
# scripts/autocommit.sh — start/stop/status for the auto-commit daemon.
#
# This container has no init system (PID 1 is bash), so the daemon cannot be
# registered as a service. Run `scripts/autocommit.sh start` after boot, or wire
# it into whatever supervisor you use.
set -euo pipefail

ROOT="/root/autodl-tmp/vla-omni-bench"
PIDFILE="$ROOT/runs/autocommit.pid"
LOG="$ROOT/runs/autocommit.log"
PY="${PYTHON:-/root/miniconda3/bin/python}"
[[ -x "$PY" ]] || PY="python3"

pid() {
  [[ -f "$PIDFILE" ]] && cat "$PIDFILE" 2>/dev/null || true
}

current_pid() {
  local p
  p="$(pid)"
  if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then
    printf '%s\n' "$p"
    return 0
  fi
  # Fallback for a daemon started outside this wrapper (no/stale pidfile).
  pgrep -f 'python .*scripts/git_autocommit_daemon.py' 2>/dev/null | head -n1 || true
}

running() {
  [[ -n "$(current_pid)" ]]
}

start() {
  mkdir -p "$ROOT/runs"
  # Serialize concurrent starts (main serve + background service both load the plugin).
  exec 9>"$ROOT/runs/autocommit.lock"
  if ! flock -n 9; then
    echo "autocommit start already in progress"
    return 0
  fi
  if running; then
    echo "autocommit already running (pid $(current_pid))"
    return 0
  fi
  cd "$ROOT"
  nohup "$PY" scripts/git_autocommit_daemon.py >>"$LOG" 2>&1 9>&- &
  echo $! >"$PIDFILE"
  sleep 1
  if running; then
    echo "autocommit started (pid $(current_pid))"
  else
    echo "autocommit failed to start; see $LOG" >&2
    rm -f "$PIDFILE"
    return 1
  fi
}

stop() {
  if ! running; then
    echo "autocommit not running"
    rm -f "$PIDFILE"
    return 0
  fi
  local p
  p="$(current_pid)"
  kill "$p" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$p" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 "$p" 2>/dev/null; then
    kill -9 "$p" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
  echo "autocommit stopped (pid $p)"
}

status() {
  if running; then
    echo "autocommit running (pid $(current_pid))"
  else
    echo "autocommit not running"
    return 1
  fi
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
