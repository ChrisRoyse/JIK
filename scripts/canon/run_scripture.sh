#!/usr/bin/env bash
# Stage 1 runner for the scripture partition (issue #9): waits for each source's receipts, canonicalises under the
# scripture-v1 contract, signs, then runs the document layers, the suffix-array layer, the token count and the FSV.
#
#   VERSION=scripture-v1 nohup scripts/canon/run_scripture.sh > <log> 2>&1 &
#   SOURCES=a,b STEP=canon ...   # only canonicalise+sign the named sources (resumable; sealed shards are skipped)
#
# Env: VERSION (default scripture-v1); CENSUS (default = VERSION); SOURCES (comma list, default: every canonicalised
#      source of the partition); STEP (canon|layers|fsv|all, default all); WORKERS (24); THREADS (32); WAIT_MAX_S (6h).
set -uo pipefail
P=${P:-$HOME/mnlm/.venv/bin/python}
R=${R:-/zfs/archive/mnlm}
O=${O:-/zfs/hot/mnlm}
TOOL=${TOOL:-$HOME/mnlm/tools/deduplicate-text-datasets}
V=${VERSION:-scripture-v1}
CENSUS=${CENSUS:-$V}
export MNLM_CANON_VERSION=${MNLM_CANON_VERSION:-$V}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-32}
WAIT_MAX_S=${WAIT_MAX_S:-21600}
STEP=${STEP:-all}
ISSUE=${ISSUE:-9}  # evidence directory: 9 for scripture-v1, 10 for identity-v1 (SOURCES=identity-v1 VERSION=identity-v1 CENSUS=identity-v1 ISSUE=10)
ALL=unfoldingword-uhb,unfoldingword-ugnt,tanach-us-uxlc,sblgnt,nestle1904,byztxt-rp2018,crosswire-sword-pd,sefaria-export,sefaria-export-nc,ebible-english,berean-bsb,open-english-bible,unfoldingword-ult,unfoldingword-ust,openbible-crossrefs,ubsicap-versification,stepbible-data
SOURCES=${SOURCES:-$ALL}
cd "$HOME/mnlm" || exit 2
ulimit -Sn 100000

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

take_lease() {  # path label: O_EXCL create; a live holder refuses, a dead holder's lease is retired; released on exit
  local lease=$1 label=$2
  mkdir -p "$(dirname "$lease")"
  if ! ( set -C; echo "$$ $(ts) $label" > "$lease" ) 2>/dev/null; then
    local pid; pid=$(cut -d" " -f1 "$lease" 2>/dev/null)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "MNLM_STAGE1_LEASE_HELD $lease: $(cat "$lease")"; exit 4
    fi
    mkdir -p "$(dirname "$lease")/archive"; mv "$lease" "$(dirname "$lease")/archive/$(basename "$lease").stale-${pid:-nopid}-$(date -u +%Y%m%dT%H%M%SZ)"
    ( set -C; echo "$$ $(ts) $label" > "$lease" ) || { echo "MNLM_STAGE1_LEASE_RACE $lease"; exit 4; }
  fi
  trap 'rm -f "'"$lease"'"' EXIT
}
step() { echo "[$(ts)] == $*"; }
STEPLOGS="$R/evidence/fsv/$ISSUE/steps"; mkdir -p "$STEPLOGS"
keep() { tee "$STEPLOGS/$V-$1.log" | tail -n "$2"; }

receipts_complete() {  # source -> 0 when every census manifest entry has an accepted receipt (the NC alias shares sefaria-export's)
  "$P" -I - "$1" "$R" "$CENSUS" <<'EOF'
import json, sys
src, root, census = sys.argv[1], sys.argv[2], sys.argv[3]
raw = {"sefaria-export-nc": "sefaria-export"}.get(src, src)
man = [json.loads(l) for l in open(f"{root}/census/{census}/{raw}/manifest.jsonl") if l.strip()]
rec = {}
try:
    for l in open(f"{root}/raw/{raw}/receipts.jsonl"):
        r = json.loads(l); rec[r["path"]] = r
except FileNotFoundError:
    pass
acc = lambda r: r.get("verified") is True or (r.get("verified") is None and r.get("attestation") == "content-length" and r.get("size_match") is True)
n = sum(acc(rec.get(e["path"], {})) for e in man)
print(f"{src}: {n}/{len(man)} accepted")
sys.exit(0 if n == len(man) else 1)
EOF
}

wait_for() {
  local t0=$SECONDS
  until receipts_complete "$1"; do
    if (( SECONDS - t0 > WAIT_MAX_S )); then echo "MNLM_STAGE1_WAIT_TIMEOUT $1"; return 1; fi
    sleep 60
  done
}

step "scripture Stage 1 run $V start on $(hostname) sources=$SOURCES step=$STEP"
take_lease "$HOME/mnlm/.lease/scripture-$V" "scripture $V"

if [[ "$STEP" == "canon" || "$STEP" == "all" ]]; then
  IFS=, read -r -a SRCS <<< "$SOURCES"
  for src in "${SRCS[@]}"; do
    wait_for "$src" || continue
    step "canonicalise $src"
    "$P" -I scripts/canon/canonicalise.py --root "$R" --out "$O" --version "$V" --census "$CENSUS" --source "$src" --workers "$WORKERS" | keep "canon-$src" 3
    step "minhash $src"
    "$P" -I scripts/canon/minhash.py --root "$R" --out "$O" --version "$V" --source "$src" --workers "$WORKERS" | keep "minhash-$src" 2
  done
fi

rc=0
if [[ "$STEP" == "layers" || "$STEP" == "all" ]]; then
  step "dedup layers over $SOURCES"
  "$P" -I scripts/canon/dedup.py --root "$R" --out "$O" --version "$V" --sources "$SOURCES"
  step "suffix-array layer"
  "$P" -I scripts/canon/sa_dedup.py --root "$R" --out "$O" --version "$V" --tool "$TOOL" --threads "$THREADS" --concurrent 4
  step "token count"
  "$P" -I scripts/canon/count_tokens.py --root "$R" --out "$O" --version "$V" --workers "$WORKERS" | keep tokens 1
fi
if [[ "$STEP" == "fsv" || "$STEP" == "layers" || "$STEP" == "all" ]]; then
  step "FSV"
  "$P" -I scripts/canon/fsv_stage1.py --root "$R" --out "$O" --version "$V" --census "$CENSUS" --issue "$ISSUE" ${FSV_FLAGS:---all-shards --workers 16}
  rc=$?
fi
step "scripture Stage 1 run $V end rc=$rc"
exit $rc
