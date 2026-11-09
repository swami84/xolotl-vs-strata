#!/usr/bin/env bash
# Benchmark one config: start the server, wait until it answers, run Xolotl's throughput probe twice, stop it.
#
#   ./scripts/bench.sh NAME CONFIG SERVER_DIR PYTHON
#
#   NAME        label for the output files (results/NAME.*)
#   CONFIG      server config JSON (from make_configs.py)
#   SERVER_DIR  checkout whose serve/server.py runs the config (Strata for A, Xolotl for B and C)
#   PYTHON      interpreter with the server's packages (regex, jinja2, numpy, ...). Strata's .venv works for both.
#
# Env: XOLOTL_DIR (for tools/throughput.py, default ~/Xolotl), PORT (8098), PROMPT (16000), DECODE (1024).
set -u
N=$1; CFG=$(readlink -f "$2"); DIR=$3; PY=$4
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
OUT=$HERE/results
XOLOTL_DIR=${XOLOTL_DIR:-$HOME/Xolotl}
PORT=${PORT:-8098}; URL=http://127.0.0.1:$PORT
PROMPT=${PROMPT:-16000}; DECODE=${DECODE:-1024}
mkdir -p "$OUT"

cd "$DIR"
setsid "$PY" serve/server.py --engine strata --config "$CFG" --port "$PORT" > "$OUT/$N.server.out" 2>&1 < /dev/null &
P=$!
for i in $(seq 1 300); do
  curl -sf --max-time 2 "$URL/v1/models" >/dev/null 2>&1 && break
  kill -0 $P 2>/dev/null || { echo "$N: server died"; tail -20 "$OUT/$N.server.out"; exit 1; }
  sleep 3
done
echo "$N: ready after $((i * 3))s"          # 3 s polling: load times are +/- 3 s
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
for r in 1 2; do
  python3 "$XOLOTL_DIR/tools/throughput.py" "$URL" --prompt "$PROMPT" --decode "$DECODE" --json "$OUT/$N.run$r.json" 2>&1 | tail -6
done
kill -INT -- -$P 2>/dev/null
for _ in $(seq 1 30); do kill -0 $P 2>/dev/null || break; sleep 1; done
kill -KILL -- -$P 2>/dev/null
# the engine can hold its VRAM for a while after the server exits; the next run would fail to allocate
for _ in $(seq 1 60); do
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
  sleep 1
done
echo "$N: stopped"
