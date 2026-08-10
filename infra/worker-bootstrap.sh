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
HOURS=2   # hard cap per worker cell — shorter than the orchestrator's, tune per sweep

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
apt install -y at >/dev/null 2>&1 || true
systemctl enable --now atd
HARD_CAP_JOB=$(echo "aws ec2 stop-instances --region ${REGION} --instance-ids ${INSTANCE_ID}" | at now + "${HOURS}" hours 2>&1 | grep -o 'job [0-9]*' | awk '{print $2}')

# --- run this worker's assigned cell ---
# train.py does not exist yet — this is the call shape it needs to
# support once it does. Update this line when it's built.
su - "${TARGET_USER}" -c "cd ${REPO_DIR} && python -m indbw.train --manifest sweep_manifest.json --worker-id ${WORKER_ID}"

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
