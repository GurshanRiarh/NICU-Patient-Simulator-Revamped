#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_NAME="neosim-dashboard.service"
UNIT_PATH="$UNIT_DIR/$UNIT_NAME"
HOST="${NEOSIM_HOST:-127.0.0.1}"
PORT="${NEOSIM_PORT:-8010}"

if [[ ! -x "$PROJECT_DIR/.venv/bin/neosim" ]]; then
  echo "error: $PROJECT_DIR/.venv/bin/neosim not found; run 'uv sync' first" >&2
  exit 1
fi

install -d "$UNIT_DIR"
sed \
  -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
  -e "s|__HOST__|$HOST|g" \
  -e "s|__PORT__|$PORT|g" \
  "$PROJECT_DIR/deploy/neosim-dashboard.service" > "$UNIT_PATH"
chmod 0644 "$UNIT_PATH"

systemctl --user daemon-reload
systemctl --user enable --now "$UNIT_NAME"
systemctl --user --no-pager --lines=0 status "$UNIT_NAME"
