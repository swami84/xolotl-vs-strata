# Xolotl vs Strata: Qwen3.8-Flash-Next on two RTX 3090s

[Xolotl](https://github.com/Indras-Mirror/Xolotl) is a fork of the [Strata](https://github.com/Niko1221/Strata)
MoE inference engine. This repo measures whether swapping Strata's engine for Xolotl's makes the same model faster
on the same machine. The model, quant, pack, MTP draft, expert profile and flags are all unchanged.

**Result:** with identical flags, Xolotl was about **40% faster** at both prefill (1,889 → 2,698 tok/s) and decode
(124 → 175 tok/s). Two sessions a day apart, one on a cold page cache and one on a warm one, gave the same picture.
Xolotl's recommended QFN flags did not add speed on this machine.

## Setup

| | |
| --- | --- |
| GPUs | 2x NVIDIA RTX 3090 (24 GB each), layer split across both (`"gpu": [1, 0]`, `"layer_split": "auto"`) |
| CPU / RAM | AMD EPYC 7C13 64-core (AVX2, no AVX-512), 503 GB RAM |
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

Main session: 8 Nov 2026. The model, pack and MTP files were read into the page cache first, so every run started
warm.

| Run | Prefill, pass 1 / 2 | Decode, pass 1 / 2 | Tokens decoded | Draft acceptance | Load |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: Strata | 1,793 / 1,984 tok/s | 115.4 / 132.5 tok/s | 348 / 594 | 73% / 81% | 24 s |
| B: Xolotl, same flags | 2,725 / 2,671 tok/s | 165.5 / 183.6 tok/s | 1,024 / 1,024 | 81% / 81% | 30 s |
| C: Xolotl, tuned | 2,625 / 2,676 tok/s | 163.0 / 168.1 tok/s | 1,024 / 1,024 | 80% / 78% | 27 s |

Means, with the first session for comparison:

| Run | Prefill, warm (8 Nov) | Decode, warm (8 Nov) | Prefill, cold (7 Nov) | Decode, cold (7 Nov) |
| --- | ---: | ---: | ---: | ---: |
| A: Strata | 1,889 tok/s | 124.0 tok/s | 1,956 tok/s | 127.8 tok/s |
| B: Xolotl, same flags | **2,698 tok/s (+43%)** | **174.6 tok/s (+41%)** | 2,688 tok/s (+37%) | 174.3 tok/s (+36%) |
| C: Xolotl, tuned | 2,650 tok/s (+40%) | 165.6 tok/s (+34%) | 2,671 tok/s (+37%) | 166.0 tok/s (+30%) |

Data:

- [results/2026-11-08-warm/](results/2026-11-08-warm/): engine logs and probe JSON for each run.
- [results/2026-11-07-cold/](results/2026-11-07-cold/): probe numbers and engine log lines from the first session.
  Its full logs were lost when a reboot cleared the temp folder they were in.

### What made Xolotl faster

This test doesn't isolate the cause. Two differences show up in the logs:

- Xolotl turns on nine bit-exact speed switches for Ampere (sm_86) cards by default:

  ```
  strata: sm_86 (Ampere): 9 exact speed switches on by default (STRATA_SM86_PERF=0 turns them off ...)
  ```

- Xolotl's prompt pipeline uses a 384-slot ring, against Strata's 96 slots (`prompt chunk auto: 8192 tokens, a
  384-slot ring`).

Rerunning B with `STRATA_SM86_PERF=0` would show how much of the gain comes from the speed switches.

### Load time

With a warm cache, the three runs loaded in 24–30 s, and Strata was fastest. Each load time is accurate only to
±3 s because of the polling interval.

In the cold session, C loaded in 27 s against B's 42 s. The cause was B's engine: it picked `--ple-io ram` and read
the model's 26.8 GiB PLE (per-layer embedding) table from disk into RAM, which took 15.2 s. With a warm cache the
same step took 2.5 s. `--ple-io direct` only saves load time when the table isn't already in the page cache. The
trade-off is RAM against disk: `ram` keeps 26.8 GiB of RAM in use while the server runs, and `direct` reads the
drive on every token.

## Caveats

- **Decode lengths differ.** Strata stopped after 348–594 tokens, while Xolotl wrote all 1,024 every time. Draft
  acceptance was similar (73–81%), and Strata's speed didn't rise with longer answers (594 tokens at 132.5 tok/s, in
  line with its other runs). So the decode gain looks real, but the exact percentage would need matched-length
  output.
- **Strata did better on other prompts.** The same Strata config, logged earlier in normal use on 13–15K-token
  prompts, prefilled at 2,670–2,930 tok/s. On this probe's source-code prompt, Strata stayed at 1,793–2,084 tok/s in
  five passes across both sessions, warm and cold. Prompt content seems to matter. Xolotl wasn't measured on those
  earlier prompts.
- **One Strata pass failed.** In a first attempt on 8 Nov, Strata's server rejected generated code with
  `malformed tool call` partway through decode, and the probe got HTTP 400. That attempt is kept in
  `results/2026-11-08-warm/attempt1/`. Its clean second pass (2,084 / 122.0 tok/s) agrees with the table. C failed
  to start in the same attempt because B's engine still held GPU memory. `bench.sh` now waits for the GPUs to clear
  between runs.
- **Agent-turn times aren't comparable.** Xolotl's longer answers meant its follow-up turn had about 1,210 new tokens
  to read, against about 185 for Strata.
- **Two passes per run.** Passes differ by up to 10% for decode. Both sessions agree within a few percent.
- **Xolotl's own headline numbers** (150–199 tok/s decode) are for a different quant, the
  [XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored) build, measured on one
  RTX 4090. This test used the IQ3_S quant on two 3090s.

## Reproduce

1. Build Xolotl's engine. This takes about 5 minutes on 64 cores.

   ```bash
   XOLOTL_DIR=~/Xolotl CUDA_ARCH=86 ./scripts/build_xolotl.sh
   ```

   `ninja` must be on `PATH`. Strata's `.venv` has it, or run `pip install ninja`.

2. Warm the page cache so load times are comparable.

   ```bash
   cat <model dir>/*.gguf <pack dir>/*.bin <mtp dir>/* > /dev/null
   ```

3. Run all three benchmarks. Start from a working Strata install and the config it serves the model with.

   ```bash
   STRATA_DIR=~/Strata XOLOTL_DIR=~/Xolotl BASE_CONFIG=~/Strata/strata-iq3_s.json ./scripts/run_all.sh
   ```

   `make_configs.py` writes `configs/A-strata.json`, `configs/B-xolotl-same.json` and `configs/C-xolotl-tuned.json`
   from the base config. `bench.sh` runs each one on port 8098 and writes logs and probe JSON to `results/`.
   Nothing else may be using the GPUs or port 8098 during the run.

The Xolotl server needs Python's `regex` package. Strata's `.venv` interpreter works for both servers, and
`run_all.sh` uses it by default.

The configs in `configs/` are the ones generated for this machine and are kept for reference. Their paths are
specific to it.

## Files

```
scripts/build_xolotl.sh   clone + build Xolotl's strata engine
scripts/make_configs.py   base Strata config -> A/B/C configs
scripts/bench.sh          start server, probe twice, stop, wait for GPUs to free
scripts/run_all.sh        A, B, C back to back
configs/                  the configs used here
results/2026-11-08-warm/  main session: engine logs + probe JSON
results/2026-11-07-cold/  first session: probe numbers + log excerpts
```
