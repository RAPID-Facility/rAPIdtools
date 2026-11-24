#!/usr/bin/env bash
# Remove the rapidtools GUI stack from AWS.
#
#   deploy/aws/teardown.sh [--stack NAME] [--region REGION] [--delete-snapshot] [--yes]
#
# Deleting the stack removes the machine, its system disk, the address, the
# network, the functions and the stored secrets. The data disk
# (/srv/rapidtools: imagery, results, model weights) is saved as an EBS
# snapshot first. That snapshot is the one thing that stays and is billed
# (about 0.05 USD per GiB actually used, per month) until you delete it:
# pass --delete-snapshot to remove it right away, or keep it and start a new
# stack from it later with launch.sh --snapshot <id>.
set -euo pipefail

STACK="rapidtools-gui"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
DELETE_SNAPSHOT=0
ASSUME_YES=0
die() { echo "ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --stack) [ $# -ge 2 ] || die "--stack needs a value"; STACK="$2"; shift ;;
    --region) [ $# -ge 2 ] || die "--region needs a value"; REGION="$2"; shift ;;
    --delete-snapshot) DELETE_SNAPSHOT=1 ;;
    --yes | -y) ASSUME_YES=1 ;;
    -h | --help) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
  shift
done

command -v aws >/dev/null || die "the AWS CLI is not installed"
[ -n "$REGION" ] || REGION=$(aws configure get region 2>/dev/null || true)
[ -n "$REGION" ] || die "no region: pass --region"
aws_r() { aws --region "$REGION" "$@"; }

aws_r cloudformation describe-stacks --stack-name "$STACK" >/dev/null 2>&1 ||
  die "no stack named $STACK in $REGION"
volume=$(aws_r cloudformation describe-stack-resources --stack-name "$STACK" --logical-resource-id DataVolume \
  --query 'StackResources[0].PhysicalResourceId' --output text 2>/dev/null || true)
case "$volume" in vol-*) ;; *) volume="" ;; esac

echo "This deletes stack $STACK in $REGION: the machine, its address and the stored token."
if [ -n "$volume" ]; then
  if [ "$DELETE_SNAPSHOT" = 1 ]; then
    echo "The data disk $volume (imagery, results, model weights) will be DELETED for good."
  else
    echo "The data disk $volume is saved as a snapshot, which stays (and is billed) until you delete it."
  fi
fi
[ "$ASSUME_YES" = 1 ] || { read -r -p "Type the stack name to confirm: " a; [ "$a" = "$STACK" ] || die "not confirmed"; }

aws_r cloudformation delete-stack --stack-name "$STACK"
echo "Deleting (5 to 15 minutes; the snapshot of a full disk takes longest)..."
if ! aws_r cloudformation wait stack-delete-complete --stack-name "$STACK"; then
  echo "The stack was not deleted completely. What is left:" >&2
  aws_r cloudformation describe-stack-events --stack-name "$STACK" --output text \
    --query "StackEvents[?ResourceStatus=='DELETE_FAILED'].[LogicalResourceId, ResourceStatusReason]" | head -n 8 >&2 || true
  die "open the stack in the CloudFormation console, choose Delete again, and retry"
fi
echo "Stack $STACK is gone."

# Secrets are removed by the stack; this covers a stack that failed before it could.
for name in gui-token hf-token smtp-password; do
  aws_r ssm delete-parameter --name "/rapidtools/$STACK/$name" >/dev/null 2>&1 || true
done

[ -n "$volume" ] || exit 0
snapshots=$(aws_r ec2 describe-snapshots --owner-ids self --filters "Name=volume-id,Values=$volume" \
  --query 'Snapshots[].SnapshotId' --output text 2>/dev/null || true)
if [ -z "$snapshots" ] || [ "$snapshots" = None ]; then
  echo "No snapshot of $volume was found."
  exit 0
fi
for snap in $snapshots; do
  if [ "$DELETE_SNAPSHOT" = 1 ]; then
    aws_r ec2 delete-snapshot --snapshot-id "$snap" && echo "Deleted snapshot $snap. Nothing is billed any more."
  else
    echo "Kept snapshot $snap of the data disk. It is billed until you delete it:"
    echo "  aws ec2 delete-snapshot --region $REGION --snapshot-id $snap"
    echo "To start a new stack with this data: deploy/aws/launch.sh --region $REGION --snapshot $snap"
  fi
done
