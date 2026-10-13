#!/usr/bin/env bash
# Stop the machine when the GUI has been idle for a while (cloud VMs bill by the hour).
# Install on the host as a cron job, e.g. every 5 minutes:
#   */5 * * * * RAPIDTOOLS_GUI_TOKEN=<token> /srv/rapidtools/idle-shutdown.sh
# "Idle" means: no job running, and no browser has used the app for IDLE_MINUTES.
# The GUI reports both in /api/state (`busy` and `idle_s`); this script's own
# polling is marked with X-Rapidtools-Probe so it never counts as activity.
# A stopped VM keeps its disk; start it again from the cloud console or a scheduler.
set -euo pipefail
IDLE_MINUTES="${IDLE_MINUTES:-30}"
URL="${RAPIDTOOLS_URL:-http://127.0.0.1:8765}"
TOKEN="${RAPIDTOOLS_GUI_TOKEN:-}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-/sbin/shutdown -h now}"

state=$(curl -fsS -H "X-Rapidtools-Token: ${TOKEN}" -H "X-Rapidtools-Probe: 1" \
    "${URL}/api/state" 2>/dev/null || echo '')
if [ -z "$state" ]; then exit 0; fi   # GUI not up yet; do nothing

read -r busy idle_s < <(printf '%s' "$state" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("1" if d.get("busy") else "0", int(float(d.get("idle_s", 0))))
')
if [ "$busy" = "1" ]; then exit 0; fi
if [ "$idle_s" -ge $((IDLE_MINUTES * 60)) ]; then
  logger -t rapidtools "idle for ${idle_s}s, shutting down" || true
  exec $SHUTDOWN_CMD
fi
