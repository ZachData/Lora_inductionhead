#!/usr/bin/env bash
# Foreground wrapper for one work cycle. Handles two shapes of "dot":
#   - simple: Claude Code does the work directly, same as before.
#   - sweep: Claude Code decides the split, writes sweep_manifest.json,
#     and this script launches N worker instances to run it concurrently.
# Either way, this instance (the orchestrator) still terminates itself
# on a clean finish. No auto-relaunch — manual, as before.

set -euo pipefail

REGION="us-east-2"
REPO_DIR="/home/ubuntu/Lora_inductionhead"
WORKER_TEMPLATE="research-vm-worker-template"
WORKER_BOOTSTRAP_TEMPLATE="$REPO_DIR/infra/worker-bootstrap.sh"
BRANCH="$(cd "$REPO_DIR" && git rev-parse --abbrev-ref HEAD)"

IMDS_TOKEN="$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")"
INSTANCE_ID="$(curl -s -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}" http://169.254.169.254/latest/meta-data/instance-id)"

# Activate the project venv so pytest, torch, and transformer_lens are
# available to whatever Claude Code invokes, and to any python3 calls
# below. Without this, imports fail against system Python, which has
# neither package (blocked by PEP 668 / externally-managed-environment).
source /home/ubuntu/venv/bin/activate

cd "$REPO_DIR"

# Reconcile the AMI-baked venv against whatever this commit's
# pyproject.toml actually declares. The venv is a snapshot from
# image-build time; without this, a dependency fix merged to the repo
# (e.g. a version pin) silently never reaches instances launched from
# an older AMI — the exact failure mode that motivated adding this line.
pip install -e ".[dev]"

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
    # Per-worker "cmd" and "instance_type" overrides, both optional (see
    # infra/prompt.md). Substituted in Python, not sed: a command contains
    # slashes ("python scripts/foo.py"), which would need escaping in a
    # sed s/// and silently mangles the userdata if it isn't.
    WORKER_CMD="$(python3 -c "
import json,sys
w=[x for x in json.load(open('sweep_manifest.json'))['workers'] if str(x['worker_id'])==sys.argv[1]][0]
print(w.get('cmd','python scripts/g0_sweep.py --manifest sweep_manifest.json --worker-id '+str(w['worker_id'])))
" "${WID}")"
    WORKER_TYPE="$(python3 -c "
import json,sys
w=[x for x in json.load(open('sweep_manifest.json'))['workers'] if str(x['worker_id'])==sys.argv[1]][0]
print(w.get('instance_type',''))
" "${WID}")"
    python3 -c "
import sys
src=open(sys.argv[1]).read()
for k,v in (('__WORKER_ID__',sys.argv[2]),('__BRANCH__',sys.argv[3]),('__WORKER_CMD__',sys.argv[4])):
    src=src.replace(k,v)
open(sys.argv[5],'w').write(src)
" "$WORKER_BOOTSTRAP_TEMPLATE" "${WID}" "${BRANCH}" "${WORKER_CMD}" "/tmp/worker-${WID}-userdata.sh"

    # The launch template is t4g.small for every version. A cell that needs
    # more memory than that (two checkpoints live at once, say) must say so
    # in the manifest, or it will be OOM-killed exactly as it would be on
    # the orchestrator. Empty -> inherit the template's type.
    TYPE_ARG=()
    if [ -n "${WORKER_TYPE}" ]; then
      TYPE_ARG=(--instance-type "${WORKER_TYPE}")
      echo "Worker ${WID}: overriding instance type -> ${WORKER_TYPE}"
    fi
    echo "Worker ${WID}: cmd = ${WORKER_CMD}"
    # One-time spot request: cheaper than on-demand, and a reclaim
    # terminates the instance (AWS only allows stop/hibernate-on-
    # interruption for *persistent* requests, which can auto-restart
    # later from a stopped state -- our worker only ever runs its work
    # from --user-data on first boot, so an auto-restarted persistent
    # instance would come back up idle rather than resuming, a zombie
    # that costs money and does nothing). Terminate-on-interruption
    # means a reclaimed worker looks identical, at the instance-state
    # level, to one that finished and pushed successfully -- see the
    # git-log check after this loop, which is the actual safety net.
    # Version=$Default is explicit on purpose. Only v4 carries the spot
    # options; v1-v3 predate them, so pinning an older version yields a
    # silently on-demand worker at several times the price. Naming the
    # version here means a change to the template's default is a visible
    # change to this line's behaviour rather than an invisible one.
    WORKER_INSTANCE_ID=$(aws ec2 run-instances \
      --region "${REGION}" \
      --launch-template LaunchTemplateName="${WORKER_TEMPLATE}",Version='$Default' \
      "${TYPE_ARG[@]}" \
      --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
      --user-data "file:///tmp/worker-${WID}-userdata.sh" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=research-vm},{Key=Role,Value=worker},{Key=WorkerId,Value=${WID}}]" \
      --query 'Instances[0].InstanceId' --output text)
    echo "Launched worker ${WID}: ${WORKER_INSTANCE_ID}"

    # Assert the thing we actually paid for. The flag above and the
    # template default both say spot, but neither is self-verifying, and
    # an on-demand worker is indistinguishable from a spot one in every
    # other respect -- it just costs several times more and nothing ever
    # says so. InstanceLifecycle is "spot" for a spot instance and absent
    # (rendered "None") otherwise. Kill it immediately rather than let a
    # sweep's worth of mispriced workers run to completion.
    LIFECYCLE=$(aws ec2 describe-instances --region "${REGION}" \
      --instance-ids "${WORKER_INSTANCE_ID}" \
      --query 'Reservations[].Instances[].InstanceLifecycle' --output text)
    if [ "${LIFECYCLE}" != "spot" ]; then
      echo "Worker ${WID} (${WORKER_INSTANCE_ID}) launched as '${LIFECYCLE}', not spot."
      echo "Terminating it and aborting the sweep -- see CLAUDE.md 'Spot only'."
      aws ec2 terminate-instances --region "${REGION}" --instance-ids "${WORKER_INSTANCE_ID}" >/dev/null || true
      if [ "${#INSTANCE_IDS[@]}" -gt 0 ]; then
        aws ec2 terminate-instances --region "${REGION}" --instance-ids "${INSTANCE_IDS[@]}" >/dev/null || true
      fi
      exit 1
    fi
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

  # A worker's instance state alone cannot be trusted as a completion
  # signal now that workers are spot: on-demand, only two things ever
  # terminate a worker (a successful push, or a spot reclaim never
  # happens) -- so "terminated" implied "pushed." A spot reclaim also
  # terminates the instance, with no push, indistinguishable from
  # success at the instance-state level. Check the actual evidence
  # instead: every worker's push carries the commit message
  # worker-bootstrap.sh always uses, so its absence means that worker's
  # result never arrived, regardless of why its instance is gone.
  MISSING_WORKERS=()
  for WID in $WORKER_IDS; do
    if ! git log --oneline --grep="^Worker ${WID} result$" | grep -q .; then
      MISSING_WORKERS+=("${WID}")
    fi
  done
  if [ "${#MISSING_WORKERS[@]}" -gt 0 ]; then
    echo "Worker(s) ${MISSING_WORKERS[*]} have no result commit -- terminated"
    echo "(likely a spot reclaim) without pushing. Not proceeding; re-launch"
    echo "just these worker IDs once capacity is available."
    exit 1
  fi

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

# --- 5. Transcript backup. The bucket does exist (it is the same one
#         worker-bootstrap.sh has always used as its result backstop);
#         this step claimed otherwise for long enough that every
#         orchestrator transcript so far was lost on terminate. Best
#         effort: never block a clean shutdown on a backup failure.
S3_BUCKET="research-vm-shared-176048535722"
if [ -d "${REPO_DIR}/.claude" ]; then
  aws s3 cp "${REPO_DIR}/.claude" \
    "s3://${S3_BUCKET}/transcripts/${INSTANCE_ID}-$(date -u +%Y%m%dT%H%M%SZ)/" \
    --recursive --region "${REGION}" >/dev/null \
    && echo "Transcript backed up to s3://${S3_BUCKET}/transcripts/" \
    || echo "Transcript backup failed — continuing to terminate anyway."
else
  echo "No .claude/ directory to back up."
fi

# --- 6. Delete this instance's own idle alarm before it's gone —
#         otherwise it lingers in INSUFFICIENT_DATA forever, referencing
#         a dead instance ID.
aws cloudwatch delete-alarms --region "${REGION}" --alarm-names "research-vm-idle-${INSTANCE_ID}" || true

# --- 7. Terminate the orchestrator. Whether there's more work is a
#         decision for the next manual launch, not this script. ---
echo "CI green. Terminating ${INSTANCE_ID}."
aws ec2 terminate-instances --region "${REGION}" --instance-ids "${INSTANCE_ID}"
