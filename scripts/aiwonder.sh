#!/usr/bin/env bash
# MNLM remote-host wrapper. Runs one command on aiwonder over SSH with the
# sanctioned key, or probes the host. Never embeds a password. See docs/AIWONDER.md.
#
#   scripts/aiwonder.sh --probe                 # hardware, GPU residents, disk, our paths
#   scripts/aiwonder.sh 'nvidia-smi'            # any single command (quote it)
#   scripts/aiwonder.sh --put <local> <remote>  # scp a file to aiwonder
#   scripts/aiwonder.sh --get <remote> <local>  # scp a file from aiwonder
set -euo pipefail

AIWONDER_HOST="${AIWONDER_HOST:?MNLM_AIWONDER_HOST_UNSET: export AIWONDER_HOST}"
AIWONDER_USER="${AIWONDER_USER:?MNLM_AIWONDER_USER_UNSET: export AIWONDER_USER}"
AIWONDER_KEY="${AIWONDER_KEY:?MNLM_AIWONDER_KEY_UNSET: export AIWONDER_KEY}"
SSH_OPTS=(-i "$AIWONDER_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)

if [[ ! -f "$AIWONDER_KEY" ]]; then
  printf '%s\n' "{\"ok\":false,\"code\":\"MNLM_AIWONDER_KEY_MISSING\",\"message\":\"$AIWONDER_KEY not found; see docs/AIWONDER.md\"}" >&2
  exit 1
fi

case "${1:-}" in
  --probe)
    exec ssh "${SSH_OPTS[@]}" "$AIWONDER_USER@$AIWONDER_HOST" '
      echo "== host =="; hostname; uptime; cat /etc/os-release | grep PRETTY_NAME
      echo "== gpu =="; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
      echo "== gpu residents (not ours unless the lease says so) =="; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
      echo "== mem =="; free -g | head -2
      echo "== disk =="; df -h / | tail -1; zfs list -o name,used,avail,mountpoint 2>/dev/null | grep -E "^(hotpool|archive)(/mnlm)?[[:space:]]"
      echo "== mnlm paths =="; for p in /zfs/hot/mnlm /zfs/archive/mnlm /home/croyse/mnlm; do [ -e "$p" ] && echo "present $p" || echo "absent  $p"; done
      echo "== lease =="; cat /home/croyse/mnlm/.lease/gpu 2>/dev/null || echo "no lease"
    '
    ;;
  --put)
    [[ $# -eq 3 ]] || { echo "usage: $0 --put <local> <remote>" >&2; exit 2; }
    exec scp "${SSH_OPTS[@]}" "$2" "$AIWONDER_USER@$AIWONDER_HOST:$3"
    ;;
  --get)
    [[ $# -eq 3 ]] || { echo "usage: $0 --get <remote> <local>" >&2; exit 2; }
    exec scp "${SSH_OPTS[@]}" "$AIWONDER_USER@$AIWONDER_HOST:$2" "$3"
    ;;
  "")
    echo "usage: $0 --probe | --put <local> <remote> | --get <remote> <local> | '<command>'" >&2
    exit 2
    ;;
  *)
    exec ssh "${SSH_OPTS[@]}" "$AIWONDER_USER@$AIWONDER_HOST" "$@"
    ;;
esac
