#!/usr/bin/env bash
# GOAI FlashRT — 优雅停止 WS policy server
#
# 用法: bash server/stop_server.sh [--port N]
# 默认 port=3001。发送 SIGTERM 等待平滑退出, 超时(15s)才强制 kill。
set -uo pipefail

PORT="3001"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--port N]"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

PIDFILE="/tmp/ws_server_${PORT}.pid"
if [ ! -f "$PIDFILE" ]; then
  echo "[GOAI] no pidfile for port ${PORT} (${PIDFILE})"
  exit 0
fi

PID=$(cat "$PIDFILE")

# _alive PID: 0 if not running; zombie (Z) counts as gone (already exited).
_alive() {
  [ -d "/proc/$1" ] || return 1
  local st
  st=$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null)
  [ -n "$st" ] && [ "$st" != "Z" ] || return 1
  return 0
}

if ! _alive "$PID"; then
  echo "[GOAI] server pid=$PID not running (stale pidfile)"
  rm -f "$PIDFILE"
  exit 0
fi

echo "==[GOAI] stopping server port=${PORT} pid=${PID} =="
kill -TERM "$PID" 2>/dev/null || true

for _ in $(seq 1 30); do
  if ! _alive "$PID"; then
    echo "[GOAI] server exited cleanly"
    rm -f "$PIDFILE"
    exit 0
  fi
  sleep 0.5
done

echo "[WARN] server did not exit in 15s; forcing kill" >&2
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PIDFILE"
echo "[GOAI] forced stop"
