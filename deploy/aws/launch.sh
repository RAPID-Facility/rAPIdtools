#!/usr/bin/env bash
# Launch (or update) the rapidtools GUI on AWS from the command line.
#
#   deploy/aws/launch.sh                         GPU machine (g5.xlarge), open to any address
#   deploy/aws/launch.sh --cpu --my-ip           small CPU machine, reachable from this network only
#   deploy/aws/launch.sh --idle-minutes 30       change one setting of an existing stack
#
# Options (all optional; on an existing stack only the ones you give change):
#   --stack NAME          stack name (default rapidtools-gui)
#   --region REGION       AWS region (default: AWS_REGION, then your CLI default)
#   --instance-type TYPE  g5.xlarge (default), g5.2xlarge, g6.xlarge, g4dn.xlarge, t3.large, ...
#   --cpu                 shorthand for --instance-type t3.large
#   --allowed-cidr CIDR   network that may open the GUI (default 0.0.0.0/0)
#   --my-ip               shorthand for --allowed-cidr <your current public address>/32
#   --domain NAME         your own DNS name instead of the automatic sslip.io name
#   --hosted-zone ID      Route 53 zone of that name; the stack then creates the record
#   --data-gb N           size of the data disk in GiB (default 100)
#   --idle-minutes N      stop after N idle minutes, 0 = never (default 60)
#   --daily-stop HOUR     also stop every day at HOUR (0 to 23), with --timezone ZONE
#   --timezone ZONE       IANA zone for --daily-stop (default America/Los_Angeles)
#   --no-wake-url         do not create the wake link
#   --image-tag TAG       container image tag (default latest; "build" forces a source build)
#   --git-ref REF         branch, tag or commit for a source build (default main)
#   --bucket NAME         existing S3 bucket the machine may read and write
#   --snapshot ID         start the data disk from a snapshot
#   --key-name NAME       EC2 key pair, with --ssh-cidr CIDR, to open SSH (not needed)
#   --ssh-cidr CIDR       network allowed to use SSH
#   --az ZONE             Availability Zone, for example us-west-2b
#   --param KEY=VALUE     any other template parameter (repeatable)
#   --yes                 do not ask for confirmation
#
# Secrets are read from the environment so they stay out of your shell history:
#   RAPIDTOOLS_GUI_TOKEN      access token (blank: one is generated)
#   HF_TOKEN                  Hugging Face token for the gated SAM 3 and Gemma weights
#   RAPIDTOOLS_SMTP_PASSWORD  with --param SmtpHost=... SmtpUser=... SmtpFrom=...
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$HERE/rapidtools-gui.yaml"
STACK="rapidtools-gui"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
INSTANCE_TYPE=""
AZ=""
ASSUME_YES=0
PARAMS=()

usage() { sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "ERROR: $*" >&2; exit 1; }
need() { [ $# -ge 2 ] || die "$1 needs a value"; }
add() { PARAMS+=("$1=$2"); }

while [ $# -gt 0 ]; do
  case "$1" in
    --stack) need "$@"; STACK="$2"; shift ;;
    --region) need "$@"; REGION="$2"; shift ;;
    --instance-type) need "$@"; INSTANCE_TYPE="$2"; shift ;;
    --cpu) INSTANCE_TYPE="t3.large" ;;
    --allowed-cidr) need "$@"; add AllowedCidr "$2"; shift ;;
    --my-ip)
      ip=$(curl -fsS -m 10 https://checkip.amazonaws.com | tr -d '[:space:]') || die "cannot find your public address"
      echo "Restricting the GUI to $ip/32"
      add AllowedCidr "$ip/32" ;;
    --domain) need "$@"; add DomainName "$2"; shift ;;
    --hosted-zone) need "$@"; add HostedZoneId "$2"; shift ;;
    --data-gb) need "$@"; add DataVolumeSizeGiB "$2"; shift ;;
    --idle-minutes) need "$@"; add IdleShutdownMinutes "$2"; shift ;;
    --daily-stop) need "$@"; add DailyStopHour "$2"; shift ;;
    --timezone) need "$@"; add ScheduleTimezone "$2"; shift ;;
    --no-wake-url) add EnableWakeUrl no ;;
    --image-tag) need "$@"; add ImageTag "$2"; shift ;;
    --git-ref) need "$@"; add GitRef "$2"; shift ;;
    --bucket) need "$@"; add DataBucketName "$2"; shift ;;
    --snapshot) need "$@"; add DataSnapshotId "$2"; shift ;;
    --key-name) need "$@"; add KeyName "$2"; shift ;;
    --ssh-cidr) need "$@"; add SshCidr "$2"; shift ;;
    --az) need "$@"; AZ="$2"; shift ;;
    --param) need "$@"; PARAMS+=("$2"); shift ;;
    --yes | -y) ASSUME_YES=1 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown option $1" ;;
  esac
  shift
done

[ -z "${RAPIDTOOLS_GUI_TOKEN:-}" ] || add AccessToken "$RAPIDTOOLS_GUI_TOKEN"
[ -z "${HF_TOKEN:-}" ] || add HfToken "$HF_TOKEN"
[ -z "${RAPIDTOOLS_SMTP_PASSWORD:-}" ] || add SmtpPassword "$RAPIDTOOLS_SMTP_PASSWORD"

command -v aws >/dev/null || die "the AWS CLI is not installed: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
[ -n "$REGION" ] || REGION=$(aws configure get region 2>/dev/null || true)
[ -n "$REGION" ] || die "no region: pass --region, for example --region us-west-2"
[ -f "$TEMPLATE" ] || die "template not found: $TEMPLATE"
account=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) ||
  die "the AWS CLI has no working credentials: run 'aws configure' or 'aws sso login' first"
aws_r() { aws --region "$REGION" "$@"; }

exists=0
if status=$(aws_r cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null); then
  exists=1
  case "$status" in
    ROLLBACK_COMPLETE | CREATE_FAILED | DELETE_FAILED)
      die "stack $STACK is in state $status from an earlier attempt. Remove it with deploy/aws/teardown.sh --stack $STACK --region $REGION and launch again." ;;
  esac
fi

if [ "$exists" = 0 ] && [ -z "$INSTANCE_TYPE" ]; then INSTANCE_TYPE="g5.xlarge"; fi
if [ -n "$INSTANCE_TYPE" ]; then
  add InstanceType "$INSTANCE_TYPE"
  # GPU machines need a vCPU quota that new accounts do not have (default 0).
  case "$INSTANCE_TYPE" in
    g*)
      case "$INSTANCE_TYPE" in *.4xlarge) vcpu=16 ;; *.2xlarge) vcpu=8 ;; *) vcpu=4 ;; esac
      quota=$(aws_r service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
        --query 'Quota.Value' --output text 2>/dev/null || echo unknown)
      if [ "$quota" != unknown ] && [ "${quota%.*}" -lt "$vcpu" ]; then
        echo "WARNING: your quota 'Running On-Demand G and VT instances' in $REGION is ${quota%.*} vCPUs;"
        echo "         $INSTANCE_TYPE needs $vcpu. The launch will fail until AWS raises it. Request it at"
        echo "         https://$REGION.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E81BA"
        echo "         or launch with --cpu for now (see \"Switching between CPU and GPU\" in the README)."
        [ "$ASSUME_YES" = 1 ] || { read -r -p "Try anyway? [y/N] " a; [ "$a" = y ] || [ "$a" = Y ] || exit 1; }
      fi ;;
  esac
  # Pick an Availability Zone that offers the machine type (new stacks only: the zone is fixed afterwards).
  if [ "$exists" = 0 ] && [ -z "$AZ" ]; then
    zones=$(aws_r ec2 describe-instance-type-offerings --location-type availability-zone \
      --filters "Name=instance-type,Values=$INSTANCE_TYPE" --query 'InstanceTypeOfferings[].Location' --output text 2>/dev/null || true)
    if [ -n "$zones" ]; then
      # shellcheck disable=SC2086
      AZ=$(printf '%s\n' $zones | sort | head -n 1)
    else
      echo "WARNING: could not confirm that $INSTANCE_TYPE is offered in $REGION."
    fi
  fi
fi
[ -z "$AZ" ] || add AvailabilityZone "$AZ"

echo
echo "Account $account, region $REGION, stack $STACK ($([ "$exists" = 1 ] && echo "update of the existing stack" || echo "new"))"
shown=()
for p in ${PARAMS[@]+"${PARAMS[@]}"}; do
  case "$p" in AccessToken=* | HfToken=* | SmtpPassword=*) shown+=("${p%%=*}=<hidden>") ;; *) shown+=("$p") ;; esac
done
echo "Settings: ${shown[*]:-(unchanged)}"
if [ "$exists" = 0 ]; then
  echo "This creates billable resources. A first launch takes 10 to 15 minutes when the container"
  echo "image is published and 25 to 40 minutes when the machine has to build it."
fi
[ "$ASSUME_YES" = 1 ] || { read -r -p "Continue? [y/N] " a; [ "$a" = y ] || [ "$a" = Y ] || exit 1; }

args=(cloudformation deploy --stack-name "$STACK" --template-file "$TEMPLATE"
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --disable-rollback)
[ ${#PARAMS[@]} -eq 0 ] || args+=(--parameter-overrides "${PARAMS[@]}")
if ! aws_r "${args[@]}"; then
  echo
  echo "The stack did not come up. Failed steps:"
  aws_r cloudformation describe-stack-events --stack-name "$STACK" --output text \
    --query "StackEvents[?contains(ResourceStatus, 'FAILED')].[LogicalResourceId, ResourceStatusReason]" | head -n 8 || true
  echo
  echo "The machine was kept so you can look at /var/log/rapidtools-bootstrap.log (see Troubleshooting"
  echo "in deploy/aws/README.md). Fix the cause, then remove the stack with"
  echo "  deploy/aws/teardown.sh --stack $STACK --region $REGION --delete-snapshot"
  echo "and launch again."
  exit 1
fi

out() { aws_r cloudformation describe-stacks --stack-name "$STACK" --output text --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue"; }
url=$(out GuiUrl)
token=$(aws_r ssm get-parameter --name "/rapidtools/$STACK/gui-token" --with-decryption \
  --query Parameter.Value --output text 2>/dev/null || true)
wake=$(out WakeLink)
echo
echo "rapidtools GUI is ready."
if [ -n "$token" ]; then
  echo "  Share link:   ${url}?token=${token}"
else
  echo "  GUI:          $url   (token: $(out GetTokenCommand))"
fi
[ -z "$wake" ] || [ "$wake" = None ] || echo "  Wake link:    $wake   (starts the machine after it stopped itself)"
echo "  Shell:        $(out ConnectCommand)"
echo "  Elastic IP:   $(out ElasticIp)"
echo "  Setup report: $(out BootstrapResult)"
echo "Stop paying for the machine: $(out StopCommand)"
echo "Remove everything:           deploy/aws/teardown.sh --stack $STACK --region $REGION"
