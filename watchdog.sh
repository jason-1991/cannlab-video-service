#!/bin/bash
# Idle watchdog: power off the environment after IDLE_LIMIT seconds
# without generation activity (tracked via ~/canvas-service/last_activity).
# Platform billing stops when the environment is off (verified by balance readback).
ACTIVITY_FILE="${ACTIVITY_FILE:-$HOME/canvas-service/last_activity}"
IDLE_LIMIT="${IDLE_LIMIT:-600}"   # 10 minutes
CHECK_EVERY=60

while true; do
  sleep "$CHECK_EVERY"
  now=$(date +%s)
  if [ -f "$ACTIVITY_FILE" ]; then
    last=$(cat "$ACTIVITY_FILE" 2>/dev/null || echo "$now")
  else
    last="$now"
  fi
  idle=$(( now - last ))
  if [ "$idle" -ge "$IDLE_LIMIT" ]; then
    echo "[watchdog] idle ${idle}s >= ${IDLE_LIMIT}s, powering off" >&2
    sync
    poweroff || shutdown -h now || sudo -n poweroff || sudo -n shutdown -h now
    exit 0
  fi
done
