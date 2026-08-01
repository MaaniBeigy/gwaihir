#!/usr/bin/env bash
# Smoke-test the local dev coach-api compose with a single /chat call.
#
# Usage:
#   scripts/smoke_chat.sh [host] [playerId] [campaignId] [sessionId]
#
# Defaults to http://localhost:8003, player=1, campaign=1, session=smoke-1.
# Replaces the previous start_interview.py auto-session sidecar.

set -euo pipefail

HOST="${1:-http://localhost:8003}"
PLAYER_ID="${2:-1}"
CAMPAIGN_ID="${3:-1}"
SESSION_ID="${4:-smoke-1}"

CURRENT_DATE="$(date -u +%Y-%m-%d)"
CURRENT_DAYNAME="$(date -u +%a | tr '[:upper:]' '[:lower:]' | cut -c1-3)"
END_DATE="$(date -u -d "+6 days" +%Y-%m-%d 2>/dev/null || date -u -v+6d +%Y-%m-%d)"

read -r -d '' BODY <<JSON || true
{
  "playerId": ${PLAYER_ID},
  "campaignId": ${CAMPAIGN_ID},
  "sessionId": "${SESSION_ID}",
  "message": "Start interview about habits and routines.",
  "context": {
    "challenges": [],
    "scheduledActivities": [],
    "schedulingWindow": {
      "scheduleStartDate": "${CURRENT_DATE}",
      "scheduleEndDate": "${END_DATE}",
      "schedulingPeriodWeeks": 1,
      "currentDate": "${CURRENT_DATE}",
      "currentDayname": "${CURRENT_DAYNAME}"
    }
  }
}
JSON

echo "POST ${HOST}/chat"
curl -sS -X POST "${HOST}/chat" \
  -H "Content-Type: application/json" \
  -H "X-Coach-Request-Id: smoke-$(date +%s)" \
  --data "${BODY}"
echo
