#!/bin/bash
# Passed as --user-data at launch, with __WORKER_ID__ substituted by
# wrapper.sh before launch. Not a repo file with a fixed path — a
# template that wrapper.sh fills in per instance.
#
# Success path: git push succeeds -> cancel hard cap -> shutdown -h now
#   -> launch template's Shutdown-behavior=Terminate does the rest.
#   No AWS API call needed for this path at all.
# Failure path (push never succeeds, or hard cap fires first): stop,
#   not terminate — needs the tag-scoped self-stop permission on
#   research-vm-worker-role, so the state is inspectable afterward.
set -euo pipefail

TARGET_USER="ubuntu"
TARGET_HOME="/home/${TARGET_USER}"
REPO_DIR="${TARGET_HOME}/Lora_inductionhead"
REPO_SSH_URL="git@github.com:ZachData/Lora_inductionhead.git"
BRANCH="__BRANCH__"
SSM_KEY_PARAM="/research-vm/github-deploy-key"
REGION="us-east-2"
WORKER_ID="__WORKER_ID__"
# Durability backstop, not the source of truth (git push still is): synced
# after every checkpoint via g0_sweep.py's sync_to_s3, so a worker killed
# mid-run (e.g. OOM, PROJECT.md 2026-08-12) doesn't lose everything it had
# already computed just because it never reached its final push. Empty ->
# g0_sweep.py no-ops, unchanged from before this existed.
S3_BUCKET="research-vm-shared-176048535722"
HOURS=3   # hard cap per worker cell — shorter than the orchestrator's, tune per sweep.
          # G0 (2026-08-13): 7 workers (vCPU-quota ceiling incl. orchestrator, both
          # t4g families are 2 vCPU regardless of small/medium) * ~21-22 checkpoints
          # * 5.5min ~= 1.9-2.0h; 3h leaves buffer for download-time variance.

IMDS_TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}" http://169.254.169.254/latest/meta-data/instance-id)

# --- deploy key + clone (same pattern as bootstrap-user-data.sh) ---
mkdir -p "${TARGET_HOME}/.ssh"
aws ssm get-parameter --name "${SSM_KEY_PARAM}" --with-decryption --region "${REGION}" \
  --query "Parameter.Value" --output text > "${TARGET_HOME}/.ssh/github_deploy_key"
chmod 600 "${TARGET_HOME}/.ssh/github_deploy_key"
chown "${TARGET_USER}:${TARGET_USER}" "${TARGET_HOME}/.ssh/github_deploy_key"
ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> "${TARGET_HOME}/.ssh/known_hosts" 2>/dev/null
chown "${TARGET_USER}:${TARGET_USER}" "${TARGET_HOME}/.ssh/known_hosts"
cat > "${TARGET_HOME}/.ssh/config" << EOF
Host github.com
  IdentityFile ${TARGET_HOME}/.ssh/github_deploy_key
  IdentitiesOnly yes
EOF
chown "${TARGET_USER}:${TARGET_USER}" "${TARGET_HOME}/.ssh/config"
chmod 600 "${TARGET_HOME}/.ssh/config"
if [ -d "${REPO_DIR}/.git" ]; then
  # AMI ships with the repo pre-cloned (baked in at image build time) —
  # a plain `git clone` into a non-empty dir fails fatally, and since
  # this runs before the push loop's own failure handling, the instance
  # is left running forever with nothing to stop it. Reset in place instead.
  su - "${TARGET_USER}" -c "cd ${REPO_DIR} && git remote set-url origin ${REPO_SSH_URL} && git fetch origin ${BRANCH} && git checkout -B ${BRANCH} origin/${BRANCH} && git reset --hard origin/${BRANCH} && git clean -fd"
else
  su - "${TARGET_USER}" -c "git clone -b ${BRANCH} ${REPO_SSH_URL} ${REPO_DIR}"
fi

# --- hard cap: stop (not terminate) if this worker never finishes ---
# `time` is GNU time, for the peak-RSS measurement below -- the shell
# builtin `time` cannot report maximum resident set size.
apt install -y at time >/dev/null 2>&1 || true
systemctl enable --now atd
HARD_CAP_JOB=$(echo "aws ec2 stop-instances --region ${REGION} --instance-ids ${INSTANCE_ID}" | at now + "${HOURS}" hours 2>&1 | grep -o 'job [0-9]*' | awk '{print $2}')

# --- swap: a checkpoint load sits right at a t4g.small's ceiling ---
# PROJECT.md §11 records this twice (integration test 2026-08-12, G1 forward
# pass): torch+transformers imports alone cost ~700MB, and the HF->TL weight
# conversion peaks well above what 1.8GB with no swap can absorb, so the
# process is OOM-killed with no error attributable to the code. Swap turns
# that hard kill into slow-but-correct. Idempotent, and a no-op on an
# instance sized large enough to never touch it.
if [ ! -f /swapfile ]; then
  fallocate -l 3G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile || true
fi

# --- run this worker's assigned cell ---
# The command is substituted by wrapper.sh from the manifest entry's "cmd"
# key, falling back to the g0_sweep default when the entry doesn't set one
# (infra/prompt.md documents that override; before this it was documented
# but not implemented, so every worker ran g0_sweep.py no matter what the
# manifest said — which silently made workers unusable for anything except
# a checkpoint-grid sweep).
#
# The `pip install` reconciles the AMI-baked venv against whatever this
# commit's pyproject.toml actually declares — the venv is a snapshot
# from image-build time, and without this a dependency fix merged to
# the repo (e.g. a version pin) silently never reaches a worker
# launched from an older AMI.
#
# Wrapped in `/usr/bin/time -v` so every worker records its own peak RSS
# to data/worker_<id>_resources.txt, which is committed with the results.
# Sizing otherwise only ever ratchets upward: someone hits an OOM, bumps
# the instance type, and nothing afterwards ever establishes that the
# smaller size would now do. CLAUDE.md ("Smallest thing that works")
# requires reading this file before sizing the next run. `-o` sends the
# report to the file and leaves the program's own stderr alone.
RESOURCE_LOG="${REPO_DIR}/data/worker_${WORKER_ID}_resources.txt"
su - "${TARGET_USER}" -c "mkdir -p ${REPO_DIR}/data"
su - "${TARGET_USER}" -c "cd ${REPO_DIR} && source /home/ubuntu/venv/bin/activate && pip install -e '.[dev]' && export G0_S3_BUCKET='${S3_BUCKET}' && /usr/bin/time -v -o '${RESOURCE_LOG}' __WORKER_CMD__"

# --- S3 backstop before the push ---
# git is the source of truth; this exists because a spot reclaim can kill
# the instance between the work finishing and the push landing. Scoped to
# result-shaped files: a blind recursive copy of data/ would also ship
# checkpoint blobs, which are large and already in S3.
su - "${TARGET_USER}" -c "aws s3 cp ${REPO_DIR}/data s3://${S3_BUCKET}/worker-results/${WORKER_ID}/ --recursive --exclude '*' --include '*.jsonl' --include '*.json' --include 'worker_*_resources.txt' --region ${REGION}" || true

# --- push result, retrying through concurrent-worker collisions ---
cd "${REPO_DIR}"
PUSHED=false
for i in $(seq 1 5); do
  su - "${TARGET_USER}" -c "cd ${REPO_DIR} && git add -A && git commit -m 'Worker ${WORKER_ID} result' --allow-empty-message || true"
  if su - "${TARGET_USER}" -c "cd ${REPO_DIR} && git pull --rebase origin ${BRANCH} && git push origin HEAD:${BRANCH}"; then
    PUSHED=true
    break
  fi
  sleep $((RANDOM % 10 + 5))   # jittered backoff, several workers may be colliding at once
done

if [ "${PUSHED}" = true ]; then
  atrm "${HARD_CAP_JOB}" 2>/dev/null || true
  shutdown -h now
else
  echo "Push failed after retries — stopping for inspection, not terminating."
  aws ec2 stop-instances --region "${REGION}" --instance-ids "${INSTANCE_ID}"
fi
