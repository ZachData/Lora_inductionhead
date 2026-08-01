#!/usr/bin/env bash
# Foreground wrapper for one work cycle: run Claude Code on the next
# board row, wait for CI, terminate self on a clean finish.
# No auto-relaunch — that's a manual step for now (see project notes).

set -euo pipefail

REGION="us-east-2"
REPO_DIR="$HOME/Lora_inductionhead"
IMDS_TOKEN="$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")"
INSTANCE_ID="$(curl -s -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}" http://169.254.169.254/latest/meta-data/instance-id)"

cd "$REPO_DIR"

# --- 1. Run Claude Code in the foreground on the next task ---
# You are watching this run. It should: read PROJECT.md, take the
# first unfinished status-board row, TDD it, commit, push.
claude

# --- 2. Refuse to proceed toward terminate on an unclean repo state ---
# (This is what teardown.sh used to check, before terminate replaced
#  stop as the end-of-cycle action. terminate destroys the volume, so
#  this check matters more here than it did there.)
if [ -n "$(git status --porcelain)" ]; then
  echo "Uncommitted changes present. Not terminating."
  git status --short
  exit 1
fi
UNPUSHED="$(git log @{u}.. --oneline 2>/dev/null || echo "no-upstream")"
if [ -n "${UNPUSHED}" ] && [ "${UNPUSHED}" != "no-upstream" ]; then
  echo "Unpushed commits present. Not terminating."
  exit 1
fi

# --- 3. Wait on the GitHub Actions result for the pushed commit ---
SHA="$(git rev-parse HEAD)"
echo "Waiting on CI for ${SHA}..."

STATUS=""
for i in $(seq 1 60); do
  STATUS="$(gh run list --commit "$SHA" --limit 1 --json status,conclusion \
            --jq '.[0].conclusion // .[0].status')"
  if [[ "$STATUS" == "success" || "$STATUS" == "failure" ]]; then
    break
  fi
  sleep 10
done

if [[ "$STATUS" != "success" ]]; then
  echo "CI did not pass (status: ${STATUS}). Not terminating."
  echo "Row should already be flagged; inspect before relaunching manually."
  exit 1
fi

# --- 4. Transcript backup: no S3 bucket exists yet. Transcript will
#         not survive this terminate. Revisit once a bucket exists —
#         see infra notes.
echo "No transcript backup configured (no S3 bucket yet) — .claude/ will be lost on terminate."

# --- 5. Terminate. Whether there's more work is a decision for the
#         next manual launch, not this script. ---
echo "CI green. Terminating ${INSTANCE_ID}."
aws ec2 terminate-instances --region "${REGION}" --instance-ids "${INSTANCE_ID}"
