#!/usr/bin/env bash
#
# nginx.sh — Start nginx alone in the foreground with the local dev config.
#
# Mirrors how scripts/serve.sh launches nginx (same prefix, config, and
# pre-created directories) — keep the two in sync.
#
# Usage: make nginx  (or ./scripts/nginx.sh from anywhere)

set -e

REPO_ROOT="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd -P)"
cd "$REPO_ROOT"

mkdir -p logs
mkdir -p temp/client_body_temp temp/proxy_temp temp/fastcgi_temp temp/uwsgi_temp temp/scgi_temp

GATEWAY_PORT="${GATEWAY_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
NGINX_PORT="${NGINX_PORT:-2026}"
NGINX_CONFIG="$REPO_ROOT/logs/nginx.local.generated.conf"

sed \
    -e "s/127\.0\.0\.1:8001/127.0.0.1:$GATEWAY_PORT/g" \
    -e "s/127\.0\.0\.1:3000/127.0.0.1:$FRONTEND_PORT/g" \
    -e "s/listen 2026;/listen $NGINX_PORT;/g" \
    -e "s/listen \[::\]:2026;/listen [::]:$NGINX_PORT;/g" \
    "$REPO_ROOT/docker/nginx/nginx.local.conf" > "$NGINX_CONFIG"

exec nginx -g 'daemon off;' -c "$NGINX_CONFIG" -p "$REPO_ROOT"
