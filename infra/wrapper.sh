#!/usr/bin/env bash
# Foreground wrapper for one work cycle. Handles two shapes of "dot":
#   - simple: Claude Code does the work directly, same as before.
#   - sweep: Claude Code decides the split, writes sweep_manifest.json,
#     and this script launches N worker instances to run it concurrently.
# Either way, this instance (the orchestrator) still terminates itself
# on a clean finish. No auto-relaunch — manual, as before.

set -euo pipefail

REGION="us-east-2"
REPO_DIR="$HOME/Lora_inductionhead"
WORKER_TEMPLATE="research-vm-worker-template"
WORKER_BOOTSTRAP_TEMPLATE="$REPO_DIR/infra/worker-bootstrap.sh"

IMDS_TOKEN="$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")"
INSTANCE_ID="$(curl -s -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}" http://169.254.169.254/latest/meta-data/instance-id)"

cd "$REPO_DIR"

# --- 1. Run Claude Code on the next row ---
claude "Read PROJECT.md. Take the first unfinished (○) row in phase order \
from the status board. Follow CLAUDE.md's development rules: write the \
test first, implement, run pytest tests/unit, and only proceed if it's \
green. If this row is a sweep (multiple independent cells — rank/lr/seed/ \
arm combinations), do not run the cells yourself: decide the split, write \
sweep_manifest.json (schema: {worker_id, arm, rank, lr, seed} per cell), \
create an empty file NEEDS_WORKERS at repo root, mark the row ⏳, commit, \
and stop there. If this row is not a sweep, do the work directly as before: \
mark ⏳ on start, ✓ or ✗ on finish. If a decision was made, append it to §9. \
If something surprising turned up, append it to §13. Commit with a real \
message and push."

# --- 2. Refuse to proceed on an unclean repo state ---
if [ -n "$(git status --porcelain)" ]; then
  echo "Uncommitted changes present. Not proceeding."
  git status --short
  exit 1
fi
UNPUSHED="$(git log @{u}.. --oneline 2>/dev/null || echo "no-upstream")"
if [ -n "${UNPUSHED}" ] && [ "${UNPUSHED}" != "no-upstream" ]; then
  echo "Unpushed commits present. Not proceeding."
  exit 1
fi

# --- 3. Branch: sweep or simple dot ---
if [ -f "NEEDS_WORKERS" ]; then
  echo "Sweep detected. Launching workers per sweep_manifest.json..."

  WORKER_IDS=$(python3 -c "import json; print(' '.join(str(w['worker_id']) for w in json.load(open('sweep_manifest.json'))['workers']))")
  INSTANCE_IDS=()

  for WID in $WORKER_IDS; do
    sed "s/__WORKER_ID__/${WID}/" "$WORKER_BOOTSTRAP_TEMPLATE" > "/tmp/worker-${WID}-userdata.sh"
    WORKER_INSTANCE_ID=$(aws ec2 run-instances \
      --region "${REGION}" \
      --launch-template LaunchTemplateName="${WORKER_TEMPLATE}" \
      --user-data "file:///tmp/worker-${WID}-userdata.sh" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=research-vm},{Key=Role,Value=worker},{Key=WorkerId,Value=${WID}}]" \
      --query 'Instances[0].InstanceId' --output text)
    echo "Launched worker ${WID}: ${WORKER_INSTANCE_ID}"
    INSTANCE_IDS+=("${WORKER_INSTANCE_ID}")
  done

  echo "Waiting on ${#INSTANCE_IDS[@]} workers to terminate..."
  while true; do
    STATES=$(aws ec2 describe-instances --region "${REGION}" \
      --instance-ids "${INSTANCE_IDS[@]}" \
      --query 'Reservations[].Instances[].State.Name' --output text)
    if ! echo "$STATES" | grep -qv "terminated"; then
      echo "All workers terminated."
      break
    fi
    if echo "$STATES" | grep -q "stopped"; then
      echo "At least one worker stopped instead of terminating — a failure"
      echo "or hard-cap path fired. Inspect before proceeding:"
      aws ec2 describe-instances --region "${REGION}" --instance-ids "${INSTANCE_IDS[@]}" \
        --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output table
      exit 1
    fi
    sleep 30
  done

  git pull --ff-only
  rm -f NEEDS_WORKERS

  # --- Re-invoke Claude Code to close out the row using worker results ---
  claude "Workers for the sweep on the current row have all finished and \
pushed their results. Read the results, mark the row ✓ or ✗ per the \
pre-registered criteria in PROJECT.md, append findings to §13 if anything \
surprising turned up, commit, and push."

  if [ -n "$(git status --porcelain)" ] || { [ -n "$(git log @{u}.. --oneline 2>/dev/null)" ] && [ "$(git log @{u}.. --oneline 2>/dev/null)" != "" ]; }; then
    echo "Uncommitted or unpushed changes after closing out the sweep. Not terminating."
    exit 1
  fi
fi

# --- 4. Wait on the GitHub Actions result for the pushed commit ---
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

# --- 5. Transcript backup: no S3 bucket exists yet. Transcript will
#         not survive this terminate. Revisit once a bucket exists —
#         see infra notes.
echo "No transcript backup configured (no S3 bucket yet) — .claude/ will be lost on terminate."

# --- 6. Terminate the orchestrator. Whether there's more work is a
#         decision for the next manual launch, not this script. ---
echo "CI green. Terminating ${INSTANCE_ID}."
aws ec2 terminate-instances --region "${REGION}" --instance-ids "${INSTANCE_ID}"
