#!/bin/bash
# Reverse SSH tunnel: expose local gen service (127.0.0.1:8900) on the
# Sydney anchor at 127.0.0.1:18900. Sydney then re-exposes it onto the
# tailnet (100.107.253.125:18900) via a local forwarder on that side.
# Dedicated key: ~/canvas-service/tunnel_key (pub added to Sydney by operator).
KEY="${TUNNEL_KEY:-$HOME/canvas-service/tunnel_key}"
REMOTE_USER="${TUNNEL_USER:-ubuntu}"
REMOTE_HOST="${TUNNEL_HOST:-3.25.167.44}"
LOCAL_PORT="${LOCAL_PORT:-8900}"
REMOTE_PORT="${REMOTE_PORT:-18900}"

exec ssh -N \
  -i "$KEY" \
  -o StrictHostKeyChecking=accept-new \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R "127.0.0.1:${REMOTE_PORT}:127.0.0.1:${LOCAL_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}"
