#!/usr/bin/env bash
# Context-length sweep: for each config, start the server once, run one discarded warm-up pass, then PASSES probe
# passes at each prompt length. A failed pass (e.g. HTTP 400) is recorded and the sweep continues.
#
#   STRATA_DIR=... XOLOTL_DIR=... ./scripts/sweep.sh
#
# Env: CONTEXTS ("4000 16000 32000 64000", the probe's --prompt target; the real token count is ~0.87x),
#      PASSES (5), DECODE (1024), CONFIGS ("A-strata B-xolotl-same C-xolotl-tuned"), PORT (8098),
#      WARM_FILES (files to read into the page cache first), EVICT_FILES (files to drop from the page cache before each
#      config and every poll while it loads: keeps a big model's page cache from filling RAM next to the engine's own
#      copy), SWEEP_OUT (results/sweep), READY_TIMEOUT (600 s).
# A config may have a configs/<name>.bench file (shell) setting DIR, SERVER (script relative to DIR) and PYBIN, for
# servers other than <checkout>/serve/server.py (e.g. Xolotl's tools/v41/serve/serve_ds41.py).
# Output: $SWEEP_OUT/<config>/<prompt>-<pass>.json, <config>.log (engine), <config>.server.out.
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
STRATA_DIR=${STRATA_DIR:-$HOME/Strata}
XOLOTL_DIR=${XOLOTL_DIR:-$HOME/Xolotl}
PY=${PY:-$STRATA_DIR/.venv/bin/python}
CONTEXTS=${CONTEXTS:-"4000 16000 32000 64000"}
PASSES=${PASSES:-5}
DECODE=${DECODE:-1024}
CONFIGS=${CONFIGS:-"A-strata B-xolotl-same C-xolotl-tuned"}
PORT=${PORT:-8098}; URL=http://127.0.0.1:$PORT
OUT=${SWEEP_OUT:-$HERE/results/sweep}
READY_TIMEOUT=${READY_TIMEOUT:-600}
mkdir -p "$OUT"

gpus_free() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
evict() { [ -n "${EVICT_FILES:-}" ] && for f in $EVICT_FILES; do dd if="$f" iflag=nocache count=0 status=none 2>/dev/null; done; true; }

if [ -n "${WARM_FILES:-}" ]; then
  echo "warming page cache"; cat $WARM_FILES > /dev/null
fi

for N in $CONFIGS; do
  CFG=$HERE/configs/$N.json
  case $N in A-*|*-strata*) DIR=$STRATA_DIR ;; *) DIR=$XOLOTL_DIR ;; esac
  SERVER=serve/server.py; PYBIN=$PY
  [ -f "$HERE/configs/$N.bench" ] && . "$HERE/configs/$N.bench"
  mkdir -p "$OUT/$N"
  evict
  for _ in $(seq 1 60); do gpus_free && break; sleep 1; done
  gpus_free || { echo "$N: GPUs busy, skipping"; continue; }

  # the engine log goes where the config says; point it into the sweep folder
  TMPCFG=$OUT/$N/config.json
  python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); c["log"]=sys.argv[2]; json.dump(c,open(sys.argv[3],"w"),indent=1)' \
    "$CFG" "$OUT/$N.log" "$TMPCFG"

  cd "$DIR"
  setsid "$PYBIN" "$SERVER" --engine strata --config "$TMPCFG" --port "$PORT" > "$OUT/$N.server.out" 2>&1 < /dev/null &
  P=$!
  t0=$(date +%s)
  while :; do
    curl -sf --max-time 2 "$URL/v1/models" >/dev/null 2>&1 && break
    kill -0 $P 2>/dev/null || { echo "$N: server died"; tail -20 "$OUT/$N.server.out"; continue 2; }
    if (( $(date +%s) - t0 > READY_TIMEOUT )); then
      echo "$N: not ready after ${READY_TIMEOUT}s, skipping"; kill -KILL -- -$P 2>/dev/null; continue 2
    fi
    evict
    sleep 2
  done
  echo "$N: ready after $(( $(date +%s) - t0 ))s"

  echo "$N: warm-up (discarded)"
  python3 "$XOLOTL_DIR/tools/throughput.py" "$URL" --prompt 4000 --decode 256 --json /dev/null > /dev/null 2>&1

  for ctx in $CONTEXTS; do
    for r in $(seq 1 "$PASSES"); do
      f=$OUT/$N/$ctx-$r.json
      if line=$(python3 "$XOLOTL_DIR/tools/throughput.py" "$URL" --prompt "$ctx" --decode "$DECODE" --json "$f" 2>&1 | tail -1) \
         && [ -s "$f" ]; then
        echo "$N ctx=$ctx pass=$r: $line"
      else
        echo "{\"failed\": true, \"error\": $(printf '%s' "$line" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()[-300:]))')}" > "$f"
        echo "$N ctx=$ctx pass=$r: FAILED: $line"
      fi
      sleep 1   # the probe picks its text offset from the clock; never start two passes in the same second
    done
  done

  kill -INT -- -$P 2>/dev/null
  for _ in $(seq 1 30); do kill -0 $P 2>/dev/null || break; sleep 1; done
  kill -KILL -- -$P 2>/dev/null
  for _ in $(seq 1 60); do gpus_free && break; sleep 1; done
  echo "$N: stopped"
done
