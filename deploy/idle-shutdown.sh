#!/usr/bin/env bash
# Stop the machine when the GUI has been idle for a while (cloud VMs bill by the hour).
# Install on the host as a cron job, e.g. every 5 minutes:
#   */5 * * * * /srv/rapidtools/idle-shutdown.sh
# "Idle" means: no job running, and nobody has polled the page for IDLE_MINUTES.
# A stopped VM keeps its disk; start it again from the cloud console or a scheduler.
set -euo pipefail
IDLE_MINUTES="${IDLE_MINUTES:-30}"
URL="${RAPIDTOOLS_URL:-http://127.0.0.1:8765}"
TOKEN="${RAPIDTOOLS_GUI_TOKEN:-}"
STAMP=/var/tmp/rapidtools-last-active

state=$(curl -fsS -H "X-Rapidtools-Token: ${TOKEN}" "${URL}/api/state" 2>/dev/null || echo '')
if [ -z "$state" ]; then exit 0; fi   # GUI not up yet; do nothing
busy=$(printf '%s' "$state" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("1" if d["busy"] else "0")')
if [ "$busy" = "1" ]; then date +%s > "$STAMP"; exit 0; fi
# Any recent HTTP activity counts as "someone is looking": the container log grows.
last_log=$(docker logs --since "${IDLE_MINUTES}m" "$(docker ps -q -f name=gui)" 2>&1 | wc -l)
if [ "$last_log" -gt 0 ]; then date +%s > "$STAMP"; exit 0; fi
now=$(date +%s); last=$(cat "$STAMP" 2>/dev/null || echo "$now")
if [ $((now - last)) -ge $((IDLE_MINUTES * 60)) ]; then
  logger -t rapidtools "idle for ${IDLE_MINUTES} min, shutting down"
  /sbin/shutdown -h now
fi
