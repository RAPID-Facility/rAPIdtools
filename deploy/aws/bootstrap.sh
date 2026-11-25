#!/usr/bin/env bash
# rapidtools GUI on AWS: instance bootstrap and admin helper.
#
# The CloudFormation template (rapidtools-gui.yaml) embeds this file in the
# instance user data, writes the stack settings to /etc/rapidtools/stack.env
# and runs `bootstrap.sh boot` on the first boot and on every later boot. It
# is also installed as `rapidtools-admin` for day-to-day tasks:
#
#   sudo rapidtools-admin status          containers, GPU, disk, share link
#   sudo rapidtools-admin token           print the share link
#   sudo rapidtools-admin rotate-token    new access token (also stored in SSM)
#   sudo rapidtools-admin update          pull or rebuild the image, restart
#   sudo rapidtools-admin restart         restart the containers
#   sudo rapidtools-admin boot            apply changed stack settings or secrets now
#   sudo rapidtools-admin logs [service]  follow the container logs
#   sudo rapidtools-admin pull-data       s3://<bucket>/data  -> /srv/rapidtools/data
#   sudo rapidtools-admin push-results    /srv/rapidtools/output -> s3://<bucket>/output
#
# Everything is idempotent: running `boot` again only reconciles the setup.
# After editing this file run build-template.py so the template embeds it.
set -Eeuo pipefail

RT_ETC="${RT_ETC:-/etc/rapidtools}"
# shellcheck disable=SC1091
if [ -f "$RT_ETC/stack.env" ]; then . "$RT_ETC/stack.env"; fi

RT_STACK="${RT_STACK:-rapidtools-gui}"
RT_REGION="${RT_REGION:-us-west-2}"
RT_HOSTNAME="${RT_HOSTNAME:-}"            # DNS name; empty derives an sslip.io name from the address
RT_EIP="${RT_EIP:-}"                      # the Elastic IP the stack attached
RT_VOLUME_ID="${RT_VOLUME_ID:-}"          # EBS data volume to mount on RT_DATA
RT_PARAM_PREFIX="${RT_PARAM_PREFIX:-/rapidtools/$RT_STACK}"
RT_GIT_REPO="${RT_GIT_REPO:-https://github.com/RAPID-Facility/rAPIdtools.git}"
RT_GIT_REF="${RT_GIT_REF:-main}"
RT_IMAGE_REPO="${RT_IMAGE_REPO:-ghcr.io/rapid-facility/rapidtools-gui}"
RT_IMAGE_TAG="${RT_IMAGE_TAG:-latest}"    # "build" skips the registry and builds RT_GIT_REF
RT_IDLE_MINUTES="${RT_IDLE_MINUTES:-0}"   # 0 disables the idle shutdown
RT_BUCKET="${RT_BUCKET:-}"
RT_SIGNAL_URL="${RT_SIGNAL_URL:-}"        # CloudFormation wait handle
RT_SMTP_HOST="${RT_SMTP_HOST:-}"
RT_SMTP_PORT="${RT_SMTP_PORT:-587}"
RT_SMTP_USER="${RT_SMTP_USER:-}"
RT_SMTP_FROM="${RT_SMTP_FROM:-}"
RT_SMTP_SSL="${RT_SMTP_SSL:-0}"

RT_HOME="${RT_HOME:-/opt/rapidtools}"     # compose files, .env, source checkout
RT_DATA="${RT_DATA:-/srv/rapidtools}"     # data volume: data/ output/ cache/ caddy/
RT_LOG="${RT_LOG:-/var/log/rapidtools-bootstrap.log}"
RT_LOCK="${RT_LOCK:-/run/rapidtools-bootstrap.lock}"
RT_GUI_URL="${RT_GUI_URL:-http://127.0.0.1:8765}"
RT_HEALTH_TIMEOUT="${RT_HEALTH_TIMEOUT:-900}"
RT_SYSTEMD_DIR="${RT_SYSTEMD_DIR:-/etc/systemd/system}"
RT_BIN_DIR="${RT_BIN_DIR:-/usr/local/bin}"
RT_PERBOOT_DIR="${RT_PERBOOT_DIR:-/var/lib/cloud/scripts/per-boot}"
RT_FSTAB="${RT_FSTAB:-/etc/fstab}"
RT_IMDS="${RT_IMDS:-http://169.254.169.254}"
RT_CERT_TRIES="${RT_CERT_TRIES:-30}"       # ten seconds apart
RT_MIN_UPTIME="${RT_MIN_UPTIME:-600}"     # never idle-stop sooner than this after boot
SHUTDOWN_CMD="${SHUTDOWN_CMD:-/sbin/shutdown -h now}"
export DEBIAN_FRONTEND=noninteractive

log() { echo "[$(date -u +%FT%TZ)] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# ---------------------------------------------------------------- helpers

imds() {  # imds <path under /latest/>, IMDSv2 only
  local t
  t=$(curl -sf -m 5 -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 120' "$RT_IMDS/latest/api/token") || return 1
  curl -sf -m 5 -H "X-aws-ec2-metadata-token: $t" "$RT_IMDS/latest/$1"
}

sslip_host() { printf 'rapidtools-%s.sslip.io\n' "${1//./-}"; }

resolve_hostname() {  # custom name, else an sslip.io name for the public address
  local ip="$RT_EIP"
  if [ -n "$RT_HOSTNAME" ]; then echo "$RT_HOSTNAME"; return; fi
  [ -n "$ip" ] || ip=$(imds meta-data/public-ipv4) || die "no public address to derive a host name from"
  sslip_host "$ip"
}

has_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

new_token() { python3 -c 'import secrets; print(secrets.token_urlsafe(18))'; }

env_get() {  # env_get KEY: value of KEY in the .env file, quotes removed
  sed -n "s/^$1='\\(.*\\)'\$/\\1/p" "$RT_HOME/.env" 2>/dev/null | tail -n 1
}

get_param() {  # prints the value; returns 0 found, 1 not found, 2 other error
  local out
  if out=$(aws ssm get-parameter --region "$RT_REGION" --name "$RT_PARAM_PREFIX/$1" \
      --with-decryption --query Parameter.Value --output text 2>&1); then
    printf '%s' "$out"
    return 0
  fi
  case "$out" in *ParameterNotFound*) return 1 ;; *) return 2 ;; esac
}

put_token() {  # JSON input, so a token that starts with "-" is not read as an option
  aws ssm put-parameter --region "$RT_REGION" --cli-input-json "$(python3 -c 'import json, sys
print(json.dumps({"Name": sys.argv[1], "Value": sys.argv[2], "Type": "SecureString", "Overwrite": True}))' \
    "$RT_PARAM_PREFIX/gui-token" "$1")" >/dev/null
}

load_token() {  # SSM is the source of truth; create the token when the stack did not
  local tok rc
  for _ in $(seq 1 12); do
    rc=0
    tok=$(get_param gui-token) || rc=$?
    if [ "$rc" = 0 ] && [ -n "$tok" ]; then echo "$tok"; return; fi
    [ "$rc" = 1 ] && break
    sleep 10   # the instance role can take a moment to become usable
  done
  [ "${rc:-2}" = 1 ] || die "cannot read $RT_PARAM_PREFIX/gui-token from Parameter Store"
  tok=$(new_token)
  put_token "$tok" || die "cannot store the access token in Parameter Store"
  echo "$tok"
}

signal() {  # signal SUCCESS|FAILURE <reason> [data]: answer the CloudFormation wait handle once
  [ -n "$RT_SIGNAL_URL" ] || return 0
  [ ! -e "$RT_HOME/.signaled" ] || return 0
  local body
  body=$(python3 -c 'import json, sys
s, r, d, u = sys.argv[1:5]
print(json.dumps({"Status": s, "Reason": r[-1000:], "UniqueId": u, "Data": d[:1000]}))' \
    "$1" "$2" "${3:-}" "$RT_STACK")
  if curl -sf -m 30 -X PUT -H 'Content-Type:' --data-binary "$body" "$RT_SIGNAL_URL" >/dev/null; then
    touch "$RT_HOME/.signaled"
  else
    log "could not signal CloudFormation"
  fi
}

on_exit() {  # EXIT trap of `boot`: report a failed first boot to CloudFormation
  local code=$?
  trap - EXIT ERR
  [ "$code" != 0 ] || return 0
  log "bootstrap failed (exit $code); full log in $RT_LOG"
  sleep 1   # let tee flush the log
  signal FAILURE "rapidtools bootstrap failed. Last log lines: $(tail -n 6 "$RT_LOG" 2>/dev/null | tr '\n"' ' .')" || true
  exit "$code"
}

dc() {  # docker compose with this deployment's files
  local files=(-f "$RT_HOME/docker-compose.yml")
  [ -e "$RT_HOME/.use-gpu" ] && files+=(-f "$RT_HOME/docker-compose.gpu.yml")
  [ -e "$RT_HOME/.use-build" ] && files+=(-f "$RT_HOME/docker-compose.build.yml")
  docker compose --project-directory "$RT_HOME" "${files[@]}" "$@"
}

# ------------------------------------------------------------- boot steps

install_packages() {
  local apt=(apt-get -y -q -o DPkg::Lock::Timeout=600)
  if ! command -v git >/dev/null || ! command -v curl >/dev/null || ! command -v unzip >/dev/null; then
    "${apt[@]}" update
    "${apt[@]}" install git curl unzip ca-certificates
  fi
  if ! command -v docker >/dev/null; then
    log "installing Docker"
    "${apt[@]}" update
    "${apt[@]}" install docker.io docker-compose-v2 || curl -fsSL https://get.docker.com | sh
  fi
  systemctl enable --now docker
  if ! docker compose version >/dev/null 2>&1; then
    log "installing the Docker Compose plugin"
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
      "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)"
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    docker compose version
  fi
  if ! command -v aws >/dev/null; then
    log "installing the AWS CLI"
    snap install aws-cli --classic || {
      curl -fsSL -o /tmp/awscliv2.zip "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip"
      unzip -q -o /tmp/awscliv2.zip -d /tmp && /tmp/aws/install --update
    }
  fi
  # GPU machines: make sure Docker knows the NVIDIA runtime (the Deep Learning AMI already does).
  if has_gpu && command -v nvidia-ctk >/dev/null && ! docker info 2>/dev/null | grep -qi nvidia; then
    nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
  fi
}

mount_data() {  # format (first time only), mount and grow the EBS data volume
  mkdir -p "$RT_DATA"
  if [ -n "$RT_VOLUME_ID" ]; then
    # Nitro machines expose EBS volumes as NVMe devices whose stable name carries the volume id.
    local link="${RT_DATA_DEVICE:-/dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_${RT_VOLUME_ID/-/}}" rc
    if ! mountpoint -q "$RT_DATA"; then
      for _ in $(seq 1 60); do
        [ -b "$link" ] && break
        sleep 5
      done
      [ -b "$link" ] || die "data volume $RT_VOLUME_ID did not appear as $link"
      # Format only when blkid positively reports "no filesystem" (exit 2), never on an error.
      rc=0
      blkid -p "$link" >/dev/null 2>&1 || rc=$?
      if [ "$rc" = 2 ]; then
        log "formatting the new data volume $RT_VOLUME_ID"
        mkfs.ext4 -q -L rapidtools "$link"
      elif [ "$rc" != 0 ]; then
        die "cannot inspect data volume $link (blkid exit $rc)"
      fi
      grep -q "[[:space:]]${RT_DATA}[[:space:]]" "$RT_FSTAB" ||
        echo "$link $RT_DATA ext4 defaults,nofail,x-systemd.device-timeout=60s 0 2" >>"$RT_FSTAB"
      mount "$RT_DATA"
    fi
    resize2fs "$link" >/dev/null 2>&1 || true   # follows a volume enlarged in the stack
  fi
  mkdir -p "$RT_DATA/data" "$RT_DATA/output" "$RT_DATA/cache" "$RT_DATA/caddy/data" "$RT_DATA/caddy/config"
}

wait_for_address() {  # Caddy needs the Elastic IP in place before it asks for a certificate
  [ -n "$RT_EIP" ] || return 0
  for _ in $(seq 1 60); do
    [ "$(imds meta-data/public-ipv4 || true)" = "$RT_EIP" ] && return 0
    sleep 5
  done
  log "WARNING: the instance does not report Elastic IP $RT_EIP yet"
}

write_env() {  # write_env <hostname> <token> <hf token> <smtp password> <image> <base image> <torch index>
  local f="$RT_HOME/.env"
  umask 077
  {
    echo "# Written by rapidtools-admin on every boot; change settings in the CloudFormation stack."
    echo "RAPIDTOOLS_DOMAIN='$1'"
    echo "RAPIDTOOLS_GUI_TOKEN='$2'"
    [ -z "$3" ] || echo "HF_TOKEN='$3'"
    echo "RAPIDTOOLS_SMTP_HOST='$RT_SMTP_HOST'"
    echo "RAPIDTOOLS_SMTP_PORT='$RT_SMTP_PORT'"
    echo "RAPIDTOOLS_SMTP_USER='$RT_SMTP_USER'"
    echo "RAPIDTOOLS_SMTP_PASSWORD='$4'"
    echo "RAPIDTOOLS_SMTP_FROM='$RT_SMTP_FROM'"
    echo "RAPIDTOOLS_SMTP_SSL='$RT_SMTP_SSL'"
    echo "RAPIDTOOLS_IMAGE='$5'"
    echo "BASE_IMAGE='$6'"
    echo "TORCH_INDEX='$7'"
    echo "RT_DATA='$RT_DATA'"
  } >"$f.new"
  mv "$f.new" "$f"
  umask 022
}

write_compose() {
  cat >"$RT_HOME/docker-compose.yml" <<'EOF'
# Written by rapidtools-admin (deploy/aws/bootstrap.sh). Same layout as
# deploy/docker-compose.yml, with the state kept on the data volume.
name: rapidtools
services:
  gui:
    image: ${RAPIDTOOLS_IMAGE}
    restart: unless-stopped
    env_file: .env
    environment:
      RAPIDTOOLS_PUBLIC_URL: https://${RAPIDTOOLS_DOMAIN}/
    volumes:
      - ${RT_DATA}/data:/data
      - ${RT_DATA}/output:/output
      - ${RT_DATA}/cache:/cache
    ports:
      - "127.0.0.1:8765:8765"
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS -H \"X-Rapidtools-Token: $$RAPIDTOOLS_GUI_TOKEN\" -H 'X-Rapidtools-Probe: 1' http://127.0.0.1:8765/api/state >/dev/null || exit 1"]
      interval: 30s
      timeout: 5s
      start_period: 120s
    logging: &logging
      driver: json-file
      options: {max-size: "20m", max-file: "3"}
  caddy:
    image: caddy:2
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    environment:
      RAPIDTOOLS_DOMAIN: ${RAPIDTOOLS_DOMAIN}
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - ${RT_DATA}/caddy/data:/data
      - ${RT_DATA}/caddy/config:/config
    depends_on:
      - gui
    logging: *logging
EOF
  cat >"$RT_HOME/docker-compose.gpu.yml" <<'EOF'
services:
  gui:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
EOF
  cat >"$RT_HOME/docker-compose.build.yml" <<'EOF'
services:
  gui:
    build:
      context: ./src
      dockerfile: deploy/Dockerfile
      args:
        BASE_IMAGE: ${BASE_IMAGE}
        TORCH_INDEX: ${TORCH_INDEX}
EOF
  cat >"$RT_HOME/Caddyfile" <<'EOF'
# Caddy obtains and renews the certificate for the name by itself (ports 80 and 443).
{$RAPIDTOOLS_DOMAIN} {
    reverse_proxy gui:8765 {
        transport http {
            read_timeout 10m
            write_timeout 10m
        }
    }
    encode gzip
    log
}
EOF
}

fetch_source() {  # check out RT_GIT_REF (branch, tag or full commit id) into $RT_HOME/src
  local src="$RT_HOME/src"
  if [ ! -d "$src/.git" ]; then
    git init -q "$src"
    git -C "$src" remote add origin "$RT_GIT_REPO"
  fi
  git -C "$src" remote set-url origin "$RT_GIT_REPO"
  git -C "$src" fetch -q --depth 1 origin "$RT_GIT_REF"
  git -C "$src" checkout -q -f FETCH_HEAD
  if [ ! -f "$src/deploy/Dockerfile" ]; then  # refs older than the deploy/ folder
    mkdir -p "$src/deploy"
    cat >"$src/deploy/Dockerfile" <<'EOF'
ARG BASE_IMAGE=pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime
FROM ${BASE_IMAGE}
ARG TORCH_INDEX=""
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/cache/huggingface RAPIDTOOLS_GUI_NO_BROWSER=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
RUN if [ -n "$TORCH_INDEX" ]; then pip install torch torchvision --index-url "$TORCH_INDEX"; fi
WORKDIR /src
COPY . /src
RUN pip install .
RUN mkdir -p /data /output /cache/rapidtools /cache/huggingface
ENV HOME=/cache
EXPOSE 8765
CMD ["rapidtools-gui", "--host", "0.0.0.0", "--port", "8765", "--no-browser", "--data-root", "/data", "--output-dir", "/output"]
EOF
  fi
  log "source at $(git -C "$src" rev-parse --short HEAD) ($RT_GIT_REF)"
}

choose_image() {  # sets IMAGE, BASE_IMAGE, TORCH_INDEX and the .use-gpu / .use-build markers
  local suffix="" force="${1:-0}"
  BASE_IMAGE="pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime"
  TORCH_INDEX=""
  if has_gpu; then
    touch "$RT_HOME/.use-gpu"
  else
    rm -f "$RT_HOME/.use-gpu"
    suffix="-cpu"
    BASE_IMAGE="python:3.11-slim"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  fi
  IMAGE="rapidtools-gui:local$suffix"
  # A machine that already built this reference keeps its build across restarts;
  # only `update` looks at the registry again.
  if [ "$force" != 1 ] && [ -e "$RT_HOME/.use-build" ] &&
    [ "$(cat "$RT_HOME/.built" 2>/dev/null)" = "$RT_GIT_REPO $RT_GIT_REF $IMAGE" ] &&
    docker image inspect "$IMAGE" >/dev/null 2>&1; then
    return 0
  fi
  rm -f "$RT_HOME/.use-build"
  if [ "$RT_IMAGE_TAG" != build ]; then
    IMAGE="$RT_IMAGE_REPO:$RT_IMAGE_TAG$suffix"
    if [ "$force" != 1 ] && docker image inspect "$IMAGE" >/dev/null 2>&1; then return 0; fi
    log "pulling $IMAGE"
    if docker pull -q "$IMAGE"; then return 0; fi
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
      log "pull failed; keeping the local copy of $IMAGE"
      return 0
    fi
    log "$IMAGE is not published; building from $RT_GIT_REPO at $RT_GIT_REF instead"
  fi
  IMAGE="rapidtools-gui:local$suffix"
  touch "$RT_HOME/.use-build"
}

start_stack() {  # start_stack <force>: build when needed, then start or reconcile the containers
  local force="${1:-0}" stamp="$RT_HOME/.built"
  if [ -e "$RT_HOME/.use-build" ]; then
    local want="$RT_GIT_REPO $RT_GIT_REF $IMAGE"
    if [ "$force" = 1 ] || [ "$(cat "$stamp" 2>/dev/null)" != "$want" ] ||
      ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
      fetch_source
      log "building $IMAGE (15 to 25 minutes for the CUDA image)"
      dc build gui
      echo "$want" >"$stamp"
    fi
  fi
  dc up -d --no-build --remove-orphans
}

install_units() {  # idle-shutdown timer, per-boot hook and the rapidtools-admin command
  ln -sf "$RT_HOME/bootstrap.sh" "$RT_BIN_DIR/rapidtools-admin"
  mkdir -p "$RT_PERBOOT_DIR"
  cat >"$RT_PERBOOT_DIR/rapidtools.sh" <<EOF
#!/bin/bash
# Re-run the stack's current user data on every boot so that a changed stack
# parameter takes effect after the stop and start CloudFormation performs.
T=\$(curl -sf -m 5 -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 120' $RT_IMDS/latest/api/token) &&
  curl -sf -m 10 -H "X-aws-ec2-metadata-token: \$T" $RT_IMDS/latest/user-data -o /run/rapidtools-userdata.sh &&
  exec bash /run/rapidtools-userdata.sh
exec $RT_HOME/bootstrap.sh boot
EOF
  chmod 755 "$RT_PERBOOT_DIR/rapidtools.sh"
  cat >"$RT_SYSTEMD_DIR/rapidtools-idle.service" <<EOF
[Unit]
Description=Stop this machine when the rapidtools GUI is idle

[Service]
Type=oneshot
ExecStart=$RT_HOME/bootstrap.sh idle-check
EOF
  cat >"$RT_SYSTEMD_DIR/rapidtools-idle.timer" <<EOF
[Unit]
Description=Check every five minutes whether the rapidtools GUI is idle

[Timer]
OnBootSec=10min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
EOF
  systemctl daemon-reload
  if [ "$RT_IDLE_MINUTES" -gt 0 ]; then
    systemctl enable --now rapidtools-idle.timer
  else
    systemctl disable --now rapidtools-idle.timer 2>/dev/null || true
  fi
}

gui_state() { curl -fsS -m 10 -H "X-Rapidtools-Token: $1" -H "X-Rapidtools-Probe: 1" "$RT_GUI_URL/api/state" 2>/dev/null; }

wait_healthy() {  # wait_healthy <token>: until the GUI answers /api/state
  local deadline=$((SECONDS + RT_HEALTH_TIMEOUT))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if gui_state "$1" >/dev/null; then log "GUI is answering"; return 0; fi
    sleep 5
  done
  dc logs --tail 40 gui || true
  die "the GUI did not answer on $RT_GUI_URL/api/state within ${RT_HEALTH_TIMEOUT}s"
}

wait_certificate() {  # wait_certificate <hostname>: up to five minutes, never fatal
  for _ in $(seq 1 "$RT_CERT_TRIES"); do
    if curl -fsS -m 10 -o /dev/null --resolve "$1:443:127.0.0.1" "https://$1/" 2>/dev/null; then
      echo "ready"
      return 0
    fi
    sleep 10
  done
  echo "certificate pending (see the troubleshooting section)"
}

# ------------------------------------------------------------ subcommands

cmd_boot() {
  local force="${1:-0}" host token hf smtp cert first=0
  [ "$(id -u)" = 0 ] || die "run as root (sudo)"
  mkdir -p "$RT_HOME" "$(dirname "$RT_LOG")"
  exec 9>"$RT_LOCK"
  flock -n 9 || die "another bootstrap run is in progress (see $RT_LOG)"
  exec > >(tee -a "$RT_LOG") 2>&1
  [ -e "$RT_HOME/.installed" ] || first=1
  # The wait handle is answered once, so only the first boot reports to CloudFormation.
  trap 'log "command failed at line $LINENO"' ERR
  trap on_exit EXIT
  log "rapidtools bootstrap starting (stack $RT_STACK, ref $RT_GIT_REF, image tag $RT_IMAGE_TAG)"
  if [ "$(readlink -f "$0")" != "$(readlink -f "$RT_HOME/bootstrap.sh")" ]; then
    install -m 700 "$0" "$RT_HOME/bootstrap.sh"
  fi
  install_packages
  mount_data
  wait_for_address
  host=$(resolve_hostname)
  token=$(load_token)
  hf=$(get_param hf-token) || hf=""
  smtp=$(get_param smtp-password) || smtp=""
  write_compose
  choose_image "$force"
  write_env "$host" "$token" "$hf" "$smtp" "$IMAGE" "$BASE_IMAGE" "$TORCH_INDEX"
  start_stack "$force"
  install_units
  wait_healthy "$token"
  cert=$(wait_certificate "$host")
  touch "$RT_HOME/.installed"
  log "ready: https://$host/ (HTTPS: $cert, GPU: $(has_gpu && echo yes || echo no), image: $IMAGE)"
  [ "$first" = 0 ] || signal SUCCESS "rapidtools GUI is up" "https://$host/ HTTPS: $cert; image: $IMAGE"
}

cmd_idle_check() {  # stop the machine after RT_IDLE_MINUTES without a job or a visitor
  local token state busy idle_s up
  [ "$RT_IDLE_MINUTES" -gt 0 ] || return 0
  up=$(cut -d. -f1 /proc/uptime)
  [ "$up" -ge "$RT_MIN_UPTIME" ] || return 0
  # A running bootstrap (first build, update) holds the lock: not idle.
  exec 8>"$RT_LOCK"
  flock -n 8 || return 0
  flock -u 8
  token=$(env_get RAPIDTOOLS_GUI_TOKEN)
  state=$(gui_state "$token" || true)
  [ -n "$state" ] || return 0   # GUI not up; do nothing
  read -r busy idle_s < <(printf '%s' "$state" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("1" if d.get("busy") else "0", int(float(d["idle_s"])) if "idle_s" in d else -1)
')
  [ "$busy" = 0 ] || return 0
  if [ "$idle_s" -lt 0 ]; then
    # Older GUI builds do not report idle_s: fall back to Caddy's access log.
    if dc logs --no-log-prefix --since "${RT_IDLE_MINUTES}m" caddy 2>/dev/null | grep -q '"request"'; then
      return 0
    fi
    idle_s=$up
  fi
  if [ "$idle_s" -ge $((RT_IDLE_MINUTES * 60)) ]; then
    logger -t rapidtools "idle for ${idle_s}s, shutting down" 2>/dev/null || true
    # shellcheck disable=SC2086
    exec $SHUTDOWN_CMD
  fi
}

cmd_token() { echo "https://$(env_get RAPIDTOOLS_DOMAIN)/?token=$(env_get RAPIDTOOLS_GUI_TOKEN)"; }

cmd_rotate_token() {
  local tok="${1:-}"
  [ -n "$tok" ] || tok=$(new_token)
  case "$tok" in *[!A-Za-z0-9_.~-]*) die "use only letters, digits and - _ . ~ in the token" ;; esac
  [ "${#tok}" -ge 12 ] || die "use at least 12 characters"
  put_token "$tok" || die "cannot store the token in Parameter Store"
  sed -i "s|^RAPIDTOOLS_GUI_TOKEN=.*|RAPIDTOOLS_GUI_TOKEN='$tok'|" "$RT_HOME/.env"
  dc up -d --no-build gui
  echo "New share link:"
  cmd_token
}

cmd_status() {
  dc ps
  if has_gpu; then nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; else echo "GPU: none"; fi
  df -h / "$RT_DATA" | sed 's/^/  /'
  echo "Idle shutdown: $RT_IDLE_MINUTES min (0 = off)"
  cmd_token
}

cmd_sync() {  # cmd_sync pull-data|push-results [extra aws s3 sync arguments]
  [ -n "$RT_BUCKET" ] || die "no S3 bucket was given to the stack (parameter DataBucketName)"
  local what="$1"
  shift
  if [ "$what" = pull-data ]; then
    aws s3 sync --region "$RT_REGION" "s3://$RT_BUCKET/data/" "$RT_DATA/data/" "$@"
  else
    aws s3 sync --region "$RT_REGION" "$RT_DATA/output/" "s3://$RT_BUCKET/output/" "$@"
  fi
}

main() {
  local cmd="${1:-status}"
  [ $# -eq 0 ] || shift
  case "$cmd" in
    boot) cmd_boot 0 ;;
    update) cmd_boot 1 ;;
    idle-check) cmd_idle_check ;;
    token) cmd_token ;;
    rotate-token) cmd_rotate_token "$@" ;;
    status) cmd_status ;;
    restart) dc restart ;;
    logs) dc logs -f --tail 100 "$@" ;;
    pull-data | push-results) cmd_sync "$cmd" "$@" ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
  esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
