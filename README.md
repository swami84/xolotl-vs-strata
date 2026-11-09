# Xolotl vs Strata: Qwen3.8-Flash-Next on two RTX 3090s

[Xolotl](https://github.com/Indras-Mirror/Xolotl) is a fork of the [Strata](https://github.com/Niko1221/Strata)
MoE inference engine. This repo measures whether swapping Strata's engine for Xolotl's makes the same model faster
on the same machine. The model, quant, pack, MTP draft, expert profile and flags are all unchanged.

**Result:** with identical flags, Xolotl decoded about **30–36% faster** (128 → 166–174 tok/s). Its prefill was
also 37% higher in this test, but most of that gap is probably a cold page cache in the Strata run (see
[Caveats](#caveats)). Xolotl's recommended QFN flags did not make it faster than the identical-flags run, but they
cut load time from 42 s to 27 s.

## Setup

| | |
| --- | --- |
| GPUs | 2x NVIDIA RTX 3090 (24 GB each), layer split across both (`"gpu": [1, 0]`, `"layer_split": "auto"`) |
| CPU / RAM | AMD EPYC 7C13 64-core, 503 GB RAM |
| Software | Ubuntu, kernel 6.8.0, NVIDIA driver 580.178.04, CUDA 12.8 |
| Model | Qwen3.8-Flash-Next GSQ-RCO **IQ3_S** (125B MoE, two GGUF shards, 46.8 GiB) |
| Strata | 0.1.39, commit `6f32ec0` (prebuilt engine from `setup.sh`) |
| Xolotl | commit `8486ead` (7 Nov 2026), built from source for sm_86 against llama.cpp `6050b467` + Xolotl's ggml patch |

Engine flags, identical for A and B:

```
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <mtp/rt>
--max-context 131072 --kv int8 --kv-resident 32768 --remote-expert-opt
```

## The three runs

| Run | Engine | What changes |
| --- | --- | --- |
| **A** | Strata | Nothing: the working Strata config as-is (baseline) |
| **B** | Xolotl | The engine binary only |
| **C** | Xolotl | B + Xolotl's QFN settings from its `configs/xol-qfn.json`: `--ple-io direct`, `STRATA_ROUTE_RESIDENT=0.8`, `STRATA_ROUTE_RESIDENT_RANKS=5-9` |

Each run starts the server, waits for it to answer, runs Xolotl's `tools/throughput.py` twice, then stops the
server. One probe pass:

1. **Prefill:** a fresh ~14K-token prompt of source code, 1 token out.
2. **Decode:** on that context, ask for a 1,024-token code answer, thinking off, temperature 0.
3. **Agent turn:** add a short user message on top and time the prefill, which reuses the cached prefix.

All speeds are the server's own timings, not wall-clock time.

## Results

| Run | Prefill, pass 1 / 2 | Decode, pass 1 / 2 | Tokens decoded | Draft acceptance | Load |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: Strata | 1,861 / 2,051 tok/s | 120.0 / 135.6 tok/s | 335 / 312 | 78% / 84% | 54 s |
| B: Xolotl, same flags | 2,705 / 2,671 tok/s | 165.7 / 182.9 tok/s | 1,024 / 1,024 | 80% / 84% | 42 s |
| C: Xolotl, tuned | 2,624 / 2,717 tok/s | 167.3 / 164.7 tok/s | 1,024 / 1,024 | 84% / 78% | 27 s |

Means:

| Run | Prefill | Decode | vs A (prefill / decode) |
| --- | ---: | ---: | ---: |
| A | 1,956 tok/s | 127.8 tok/s | — |
| B | 2,688 tok/s | 174.3 tok/s | +37% / +36% |
| C | 2,671 tok/s | 166.0 tok/s | +37% / +30% |

Raw numbers are in [results/summary.json](results/summary.json). The engine log lines are in
[results/log-excerpts.txt](results/log-excerpts.txt).

### Why C loads faster

Xolotl has a residency policy for the model's 26.8 GiB PLE (per-layer embedding) table. With 485 GiB of RAM free,
B's engine chose `--ple-io ram` and spent **15.2 s** locking the table into RAM:

```
strata generate: PLE table locked in RAM (--ple-io ram chosen by the residency policy) in 15.2 s
```

C forces `--ple-io direct`, so the engine reads table rows from the NVMe as tokens need them. It skips that step,
and 42 s − 15 s ≈ 27 s. The trade-off: B keeps 26.8 GiB of RAM in use while it serves, and C touches the drive on
every token. Decode speed was the same here, but C could slow down if other jobs load the drive heavily.

## Caveats

- **Strata's prefill number is probably low.** A ran first, on a cold page cache: Strata loaded the weights at
  1.86 GiB/s, while B and C loaded them at about 15 GiB/s from the cache A had warmed. The same Strata config, logged
  earlier in normal use on similar 13–15K-token prompts, prefilled at **2,670–2,930 tok/s**. That is close to
  Xolotl's 2,670–2,720. Treat the 37% prefill gain as unproven until A is rerun warm.
- **Decode lengths differ.** Strata stopped after about 320 tokens, while Xolotl wrote all 1,024. Draft acceptance
  was similar (78–84% in every run), and Strata's earlier logged decode (116–141 tok/s) matches run A. So the decode
  gain looks real, but the exact percentage needs a matched-length run.
- **Load times aren't comparable between A and B** for the same cold-cache reason. The 3 s polling interval also
  makes each load time accurate only to ±3 s.
- **Agent-turn times aren't comparable.** Xolotl's longer answers meant its follow-up turn had about 1,230 new tokens
  to read, against about 200 for Strata.
- **Two passes per run.** The spread between passes is up to 10% for decode.
- **Only the full server logs were lost.** They sat in a temp folder that a reboot cleared. The numbers above come
  from the probe output and engine log lines captured during the runs.
- **Xolotl's own headline numbers** (150–199 tok/s decode) are for a different quant, the
  [XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored) build, measured on one
  RTX 4090. Its RTX 40 speed switches (`STRATA_SM89_PERF`) do not apply to 3090s. This test used neither.

## Reproduce

1. Build Xolotl's engine. This takes about 5 minutes on 64 cores.

   ```bash
   XOLOTL_DIR=~/Xolotl CUDA_ARCH=86 ./scripts/build_xolotl.sh
   ```

   `ninja` must be on `PATH`. Strata's `.venv` has it, or run `pip install ninja`.

2. Run all three benchmarks. Start from a working Strata install and the config it serves the model with.

   ```bash
   STRATA_DIR=~/Strata XOLOTL_DIR=~/Xolotl BASE_CONFIG=~/Strata/strata-iq3_s.json ./scripts/run_all.sh
   ```

   `make_configs.py` writes `configs/A-strata.json`, `configs/B-xolotl-same.json` and `configs/C-xolotl-tuned.json`
   from the base config. `bench.sh` runs each one on port 8098 and writes logs and probe JSON to `results/`.
   Nothing else may be using the GPUs or port 8098 during the run.

   To get a fair prefill comparison, warm the page cache before run A, for example with
   `cat <model shards> > /dev/null`. Or run A twice and use the second result.

The Xolotl server needs Python's `regex` package. Strata's `.venv` interpreter works for both servers, and
`run_all.sh` uses it by default.

The configs in `configs/` are the ones generated for this machine and are kept for reference. Their paths are
specific to it.

## Files

```
scripts/build_xolotl.sh   clone + build Xolotl's strata engine
scripts/make_configs.py   base Strata config -> A/B/C configs
scripts/bench.sh          start server, probe twice, stop
scripts/run_all.sh        A, B, C back to back
configs/                  the configs used here
results/                  summary.json, log-excerpts.txt
```
