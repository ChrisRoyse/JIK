#!/usr/bin/env bash
# Stage 1 runner for aiwonder (issue #8): waits for each source's acquisition receipts, canonicalises, signs, then runs
# the document layers, the suffix-array layer and the FSV. Every step appends to the run log with wall-clock timings;
# steps whose output is long keep it in full under <root>/evidence/fsv/8/steps/ and only their tail in the run log.
#
#   VERSION=p0micro-v1 FINEWEB_SHARDS=000_00000 NEMOTRON_FIRST=1 nohup scripts/canon/run_stage1.sh > <log> 2>&1 &
#   VERSION=p0-v1 FINEWEB_SHARDS=all NEMOTRON_FIRST=0 ...
#
# Env: VERSION (required); FINEWEB_SHARDS (comma list or "all"); NEMOTRON_FIRST (1 = only the first actual file);
#      WAIT_MAX_S (default 6h per source); WORKERS (default 24); THREADS (default 32);
#      MNLM_CANON_VERSION (reuse another version's sealed canon/sig shards, e.g. p0micro-v1 for p0-v1).
set -euo pipefail
P=${P:-$HOME/mnlm/.venv/bin/python}
R=${R:-/zfs/archive/mnlm}
O=${O:-/zfs/hot/mnlm}
TOOL=${TOOL:-$HOME/mnlm/tools/deduplicate-text-datasets}
V=${VERSION:?VERSION required}
WORKERS=${WORKERS:-24}
THREADS=${THREADS:-32}
WAIT_MAX_S=${WAIT_MAX_S:-21600}
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
STEPLOGS="$R/evidence/fsv/8/steps"; mkdir -p "$STEPLOGS"
keep() {  # name lines: full step output goes to a per-step log (progress visible mid-run); the run log keeps the tail
  tee "$STEPLOGS/$V-$1.log" | tail -n "$2"
}

receipts_complete() {  # source -> 0 when every manifest entry has a verified receipt (or the subset we need)
  "$P" -I - "$1" "$2" <<'EOF'
import json, sys
src, need = sys.argv[1], sys.argv[2]
man = [json.loads(l) for l in open(f"/zfs/archive/mnlm/census/p0-v1/{src}/manifest.jsonl") if l.strip()]
rec = {}
try:
    for l in open(f"/zfs/archive/mnlm/raw/{src}/receipts.jsonl"):
        r = json.loads(l); rec[r["path"]] = r
except FileNotFoundError:
    pass
paths = [e["path"] for e in man]
if need == "first":
    paths = paths[:1]
acc = lambda r: r.get("verified") is True or (r.get("verified") is None and r.get("attestation") == "content-length" and r.get("size_match") is True)
ok = all(acc(rec.get(p, {})) for p in paths)
print(f"{src}: {sum(acc(rec.get(p, {})) for p in paths)}/{len(paths)} verified")
sys.exit(0 if ok else 1)
EOF
}

wait_for() {  # source need
  local t0=$SECONDS
  until receipts_complete "$1" "$2"; do
    if (( SECONDS - t0 > WAIT_MAX_S )); then echo "MNLM_STAGE1_WAIT_TIMEOUT $1"; return 1; fi
    sleep 60
  done
}

canon_and_sign() {  # source [shards]
  local src=$1 shards=${2:-}
  local extra=()
  [[ -n "$shards" && "$shards" != "all" ]] && extra=(--shards "$shards")
  step "canonicalise $src ${shards:-all}"
  "$P" -I scripts/canon/canonicalise.py --root "$R" --out "$O" --version "$V" --source "$src" --workers "$WORKERS" "${extra[@]}" | keep "canon-$src" 3
  step "minhash $src"
  local chunk=()
  [[ "$src" == wikidata5m ]] && chunk=(--chunk-docs 320000)  # one 4.8 M-doc shard: measured 31 s chunked against 236 s whole; FineWeb measured slower chunked
  "$P" -I scripts/canon/minhash.py --root "$R" --out "$O" --version "$V" --source "$src" --workers "$WORKERS" "${chunk[@]}" | keep "minhash-$src" 2
}

first_nemotron_shard() {
  "$P" -I -c 'import json,sys; e=json.loads(open("/zfs/archive/mnlm/census/p0-v1/nemotron-cc-high-actual-slice/manifest.jsonl").readline()); print(e["path"].rsplit("/",1)[-1].split(".")[0])'
}

step "Stage 1 run $V start on $(hostname)"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
take_lease "$HOME/mnlm/.lease/stage1-$V" "stage1 $V"

canon_and_sign fineweb-edu-sample-10BT "${FINEWEB_SHARDS:-000_00000}"
wait_for wikipedia-en-20231101 all
canon_and_sign wikipedia-en-20231101
wait_for wikidata5m all
canon_and_sign wikidata5m
wait_for opencodeinstruct all
canon_and_sign opencodeinstruct
wait_for gdelt-events-90d all
canon_and_sign gdelt-events-90d
if [[ "${NEMOTRON_FIRST:-1}" == "1" ]]; then
  wait_for nemotron-cc-high-actual-slice first
  canon_and_sign nemotron-cc-high-actual-slice "$(first_nemotron_shard)"
else
  wait_for nemotron-cc-high-actual-slice all
  canon_and_sign nemotron-cc-high-actual-slice all
  wait_for nemotron-cc-high-synthetic-twins all
  canon_and_sign nemotron-cc-high-synthetic-twins all
fi
if [[ "${FINEWEB_SHARDS:-}" == "all" ]]; then
  wait_for fineweb-edu-sample-10BT all
  canon_and_sign fineweb-edu-sample-10BT all
fi

SOURCES=wikipedia-en-20231101,wikidata5m,opencodeinstruct,nemotron-cc-high-actual-slice,fineweb-edu-sample-10BT,gdelt-events-90d
step "dedup layers over $SOURCES"
"$P" -I scripts/canon/dedup.py --root "$R" --out "$O" --version "$V" --sources "$SOURCES"
step "suffix-array layer"
"$P" -I scripts/canon/sa_dedup.py --root "$R" --out "$O" --version "$V" --tool "$TOOL" --threads "$THREADS" --concurrent 4
step "token count"
"$P" -I scripts/canon/count_tokens.py --root "$R" --out "$O" --version "$V" --workers "$WORKERS" | keep tokens 1
step "FSV"
"$P" -I scripts/canon/fsv_stage1.py --root "$R" --out "$O" --version "$V" --issue 8 ${FSV_FLAGS:---all-shards --workers 16}
rc=$?
step "Stage 1 run $V end rc=$rc"
exit $rc
