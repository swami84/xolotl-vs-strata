#!/usr/bin/env python3
"""Generate the three benchmark configs from a working Strata server config.

    python3 scripts/make_configs.py BASE_CONFIG XOLOTL_DIR [OUT_DIR]

BASE_CONFIG is the JSON your Strata install already serves the model with (here: strata-iq3_s.json,
Qwen3.8-Flash-Next IQ3_S on two GPUs). The three configs differ only where stated:

  A-strata        BASE_CONFIG unchanged (bare Strata engine)                    -- baseline
  B-xolotl-same   same args, engine binary swapped for XOLOTL_DIR/build-xolotl/strata
  C-xolotl-tuned  B + Xolotl's recommended QFN settings: --ple-io direct and
                  STRATA_ROUTE_RESIDENT=0.8 / STRATA_ROUTE_RESIDENT_RANKS=5-9 (from Xolotl's configs/xol-qfn.json)

All three serve on port 8098 and log to OUT_DIR/../results/<name>.log.
"""
import json
import os
import sys

PORT = 8098


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    base_path, xolotl = sys.argv[1], os.path.abspath(sys.argv[2])
    out = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "configs"))
    results = os.path.abspath(os.path.join(out, "..", "results"))
    os.makedirs(out, exist_ok=True)
    base = json.load(open(base_path))
    base.pop("_comment", None)

    a = dict(base, port=PORT, log=os.path.join(results, "A-strata.log"))

    b = dict(a, exe=os.path.join(xolotl, "build-xolotl", "strata"), cwd=xolotl,
             log=os.path.join(results, "B-xolotl-same.log"))

    args = list(b["args"])
    i = args.index("--ple-gguf")
    args[i + 2:i + 2] = ["--ple-io", "direct"]
    c = dict(b, args=args, log=os.path.join(results, "C-xolotl-tuned.log"),
             env={"STRATA_ROUTE_RESIDENT": "0.8", "STRATA_ROUTE_RESIDENT_RANKS": "5-9"})

    for name, cfg in (("A-strata", a), ("B-xolotl-same", b), ("C-xolotl-tuned", c)):
        p = os.path.join(out, name + ".json")
        json.dump(cfg, open(p, "w"), indent=1)
        print(p)


if __name__ == "__main__":
    main()
