#!/bin/bash
# OPTIONAL — not part of the main work cycle (wrapper.sh handles that,
# and terminates rather than stops). Keep this only if you want a manual
# "pause without losing the volume" command for stepping away mid-session
# without finishing a full row and without waiting on the idle-CPU alarm.
#
# Run manually, as the ubuntu user.
# Refuses to run if there are uncommitted changes — commit with a real
# message first (ask Claude Code to summarize the diff and commit it).
#
# Pushes, then STOPS (not terminates) the instance — stop keeps the EBS
# volume and instance ID, so a bad pause doesn't cost you the environment.
# Terminate manually and separately once you're sure you no longer need
# this instance at all.
#
# Requires: research-vm-ssm-role has ec2:StopInstances scoped to its own
# instance ARN — already granted (self-stop inline policy).
set -euo pipefail
REPO_DIR="$HOME/Lora_inductionhead"
REGION="us-east-2"
cd "${REPO_DIR}"
if [ -n "$(git status --porcelain)" ]; then
  echo "Uncommitted changes present. Commit with a descriptive message before running teardown:"
  git status --short
  exit 1
fi
UNPUSHED=$(git log @{u}.. --oneline 2>/dev/null || echo "no-upstream")
if [ -n "${UNPUSHED}" ] && [ "${UNPUSHED}" != "no-upstream" ]; then
  git push
else
  echo "Nothing to push."
fi
IMDS_TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}" http://169.254.169.254/latest/meta-data/instance-id)
echo "Stopping instance ${INSTANCE_ID}..."
aws ec2 stop-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}"
