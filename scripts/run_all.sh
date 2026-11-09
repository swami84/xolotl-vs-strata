#!/usr/bin/env bash
# Run A, B and C back to back. Edit the four paths for your machine.
#
#   STRATA_DIR=... XOLOTL_DIR=... BASE_CONFIG=... ./scripts/run_all.sh
#
# Nothing else may be using the GPUs or port 8098 while this runs.
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)

STRATA_DIR=${STRATA_DIR:-$HOME/Strata}
XOLOTL_DIR=${XOLOTL_DIR:-$HOME/Xolotl}
BASE_CONFIG=${BASE_CONFIG:-$STRATA_DIR/strata-iq3_s.json}
PY=${PY:-$STRATA_DIR/.venv/bin/python}   # has regex/jinja2/numpy; system python3 lacks `regex`
export XOLOTL_DIR

python3 "$HERE/scripts/make_configs.py" "$BASE_CONFIG" "$XOLOTL_DIR" "$HERE/configs"

"$HERE/scripts/bench.sh" A-strata       "$HERE/configs/A-strata.json"       "$STRATA_DIR" "$PY"
"$HERE/scripts/bench.sh" B-xolotl-same  "$HERE/configs/B-xolotl-same.json"  "$XOLOTL_DIR" "$PY"
"$HERE/scripts/bench.sh" C-xolotl-tuned "$HERE/configs/C-xolotl-tuned.json" "$XOLOTL_DIR" "$PY"

grep -h 'tok/s' "$HERE"/results/*.log | grep -v ' 1 generated' || true
