# Xolotl vs Strata: Qwen3.8-Flash-Next on two RTX 3090s

[Xolotl](https://github.com/Indras-Mirror/Xolotl) is a fork of the [Strata](https://github.com/Niko1221/Strata)
MoE inference engine. This repo measures whether swapping Strata's engine for Xolotl's makes the same model faster
on the same machine. The model, quant, pack, MTP draft, expert profile and flags are all unchanged.

**Result, with identical flags:**

- **Decode is about 20% faster** when both engines write the same number of tokens: 138 → 164 tok/s at 3.3K
  prompt tokens and 132 → 164 tok/s at 54K. Where Strata's answers stopped early, the gap was 32–38%.
- **Prefill is 23–39% faster from 14K prompt tokens up, but 10% slower at 3.3K.** The gain peaks at 14K, which
  happens to be Strata's weakest point, and shrinks as prompts get longer.
- **Xolotl's recommended QFN flags didn't help on this machine.** They were within a few percent of the
  identical-flags run either way.

![Prefill and decode speed vs prompt length](docs/speed_vs_context.png)

![Xolotl speedup over Strata vs prompt length](docs/speedup_vs_context.png)

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

## The three configs

| Config | Engine | What changes |
| --- | --- | --- |
| **A** | Strata | Nothing: the working Strata config as-is (baseline) |
| **B** | Xolotl | The engine binary only |
| **C** | Xolotl | B + Xolotl's QFN settings from its `configs/xol-qfn.json`: `--ple-io direct`, `STRATA_ROUTE_RESIDENT=0.8`, `STRATA_ROUTE_RESIDENT_RANKS=5-9` |

## Method

The test uses Xolotl's `tools/throughput.py` probe. One pass:

1. **Prefill:** a fresh prompt of source code, 1 token out. The text comes from a new offset in the corpus each time,
   so nothing is reused from cache.
2. **Decode:** on that context, ask for a 1,024-token code answer, thinking off, temperature 0.
3. **Agent turn:** add a short user message on top. This step isn't reported, because the turn size depends on how
   long the previous answer was.

All speeds are the server's own timings, not wall-clock time.

**Context sweep (8 Nov 2026):** prompts of 3.3K, 14K, 27K and 54K tokens (the probe's 4K/16K/32K/64K targets), with
5 passes at each length. The model, pack and MTP files were read into the page cache first. Each config's server
started once and ran one warm-up pass, which was discarded. Configs ran one after another, never at the same time.

**Confidence intervals:** speeds are means with a 95% CI from the t distribution. Speedups are the ratio of the two
means, with a 95% bootstrap CI from 10,000 resamples.

## Results

### Prefill (tokens/s, mean ± 95% CI)

| Prompt | A: Strata | B: Xolotl, same flags | B / A | C: Xolotl, tuned | C / A |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3.3K | 1,646 ± 14 | 1,487 ± 71 | **0.90** (0.88–0.93) | 1,559 ± 14 | 0.95 (0.94–0.95) |
| 14K | 1,920 ± 76 | 2,676 ± 28 | **1.39** (1.37–1.44) | 2,728 ± 55 | 1.42 (1.39–1.47) |
| 27K | 2,573 ± 38 | 3,263 ± 35 | **1.27** (1.26–1.28) | 3,248 ± 120 | 1.26 (1.23–1.29) |
| 54K | 3,201 ± 31 | 3,920 ± 59 | **1.23** (1.21–1.24) | 3,752 ± 118 | 1.17 (1.15–1.20) |

### Decode (tokens/s, mean ± 95% CI)

| Prompt | A: Strata (tokens written) | B: Xolotl, same flags | B / A | C: Xolotl, tuned | C / A |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3.3K | 137.8 ± 5.7 (1,024) | 164.1 ± 2.4 | **1.19** (1.16–1.23) | 166.3 ± 8.9 | 1.21 (1.16–1.26) |
| 14K | 129.9 ± 5.9 (285) | 178.4 ± 6.0 | 1.37 (1.33–1.43) | 171.0 ± 4.6 | 1.32 (1.27–1.36) |
| 27K | 127.7 ± 4.1 (315) | 176.4 ± 3.7 | 1.38 (1.35–1.41) | 171.5 ± 5.0 | 1.34 (1.31–1.37) |
| 54K | 131.7 ± 2.2 (1,024) | 163.6 ± 6.7 | **1.24** (1.21–1.28) | 157.5 ± 6.5 | 1.20 (1.16–1.23) |

Xolotl wrote all 1,024 tokens in every pass. Strata wrote all 1,024 at 3.3K and 54K, but stopped early (285–315
tokens on average) at 14K and 27K. So only the 3.3K and 54K rows compare equal-length answers, and they are the
bold ones. Strata's own speed barely changed with answer length (128–138 tok/s), which suggests the 14K and 27K
gaps are roughly right. The safe claim is still the matched one: about +20%.

All numbers are in [results/sweep/summary.csv](results/sweep/summary.csv), and each pass has its own JSON under
`results/sweep/<config>/`.

### Reading the results

- **Prefill depends on prompt length.** Both engines get faster with longer prompts, but in different ways.
  Strata gains little from 3.3K to 14K (1,646 → 1,920) and then climbs. Xolotl starts lower, jumps by 14K
  (1,487 → 2,676) and then rises more slowly. Its biggest advantage, about 40%, lands at 14K, where Strata is
  weakest. An earlier test of 14K prompts alone therefore overstated the overall prefill gain.
- **The decode gain is smaller with longer prompts.** Xolotl was 19–24% faster at the two matched lengths. Its own
  decode peaked at 14–27K (176–178 tok/s) and dropped at 54K (164 tok/s).
- **The tuned flags (C) were a wash.** They were 5% faster than B on 3.3K prefill and 4% slower on 54K prefill.
  Decode was the same within the CIs.

### What made Xolotl faster

This test doesn't isolate the cause. Two differences show up in the logs:

- Xolotl turns on nine bit-exact speed switches for Ampere (sm_86) cards by default. The Strata 0.1.39 binary
  doesn't have them.

  ```
  strata: sm_86 (Ampere): 9 exact speed switches on by default (STRATA_SM86_PERF=0 turns them off ...)
  ```

- Xolotl's prompt pipeline uses a 384-slot ring, against Strata's 96 slots (`prompt chunk auto: 8192 tokens, a
  384-slot ring`).

Rerunning B with `STRATA_SM86_PERF=0` would show how much of the gain comes from the speed switches.

## Earlier runs at 14K only

Before the sweep, two sessions ran 2 passes per config at ~14K tokens only. They agree with the sweep's 14K row.

| Session | A prefill / decode | B prefill / decode | C prefill / decode |
| --- | ---: | ---: | ---: |
| 7 Nov, cold page cache | 1,956 / 127.8 | 2,688 / 174.3 | 2,671 / 166.0 |
| 8 Nov, warm page cache | 1,889 / 124.0 | 2,698 / 174.6 | 2,650 / 165.6 |

Data is in [results/2026-11-07-cold/](results/2026-11-07-cold/) and
[results/2026-11-08-warm/](results/2026-11-08-warm/). The first session's full logs were lost when a reboot
cleared the temp folder they were in.

### Load time

Load time is the time until the server first answered. With a warm cache, the configs loaded in 20–30 s, and Strata
was fastest. In the cold session, C loaded in 27 s against B's 42 s. The cause was B's engine: it picked
`--ple-io ram` and read the model's 26.8 GiB PLE (per-layer embedding) table from disk into RAM, which took 15.2 s.
With a warm cache the same step took 2.5 s. `--ple-io direct` only saves load time when the table isn't already in
the page cache. The trade-off is RAM against disk: `ram` keeps 26.8 GiB of RAM in use while the server runs, and
`direct` reads the drive on every token.

## Caveats

- **Strata stopped early at 14K and 27K.** See the decode table: only 3.3K and 54K compare equal-length answers.
- **Strata failed 2 of its 20 sweep passes.** One failed at 27K and one at 54K, each with
  `the engine reported an error: malformed tool call` from Strata's server (HTTP 400). Those points use 4 passes.
  The same error happened once before the sweep. Xolotl had no failures.
- **The probe prompt is source code.** In earlier normal use, the same Strata config prefilled 13–15K-token prompts
  at 2,670–2,930 tok/s, against 1,920 on the probe at 14K. Prompt content changes prefill speed, so other workloads
  may show a smaller or larger gap.
- **B's 14K points come from a second server session.** The project folder was renamed during the first sweep and
  those results weren't saved, so B was restarted for that one length. Its 14K numbers match the earlier sessions
  (2,676 vs 2,688–2,698 prefill).
- **Xolotl's own headline numbers** (150–199 tok/s decode) are for a different quant, the
  [XQ 45 GB](https://huggingface.co/IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored) build, measured on one
  RTX 4090. This test used the IQ3_S quant on two 3090s.

## Reproduce

1. Build Xolotl's engine. This takes about 5 minutes on 64 cores.

   ```bash
   XOLOTL_DIR=~/Xolotl CUDA_ARCH=86 ./scripts/build_xolotl.sh
   ```

   `ninja` must be on `PATH`. Strata's `.venv` has it, or run `pip install ninja`.

2. Generate the three configs from your working Strata config.

   ```bash
   python3 scripts/make_configs.py ~/Strata/strata-iq3_s.json ~/Xolotl
   ```

3. Run the context sweep. It takes about 40 minutes for three configs, 4 lengths and 5 passes. `WARM_FILES` reads
   the model into the page cache first.

   ```bash
   STRATA_DIR=~/Strata XOLOTL_DIR=~/Xolotl \
   WARM_FILES="<model dir>/*.gguf <pack dir>/*.bin <mtp dir>/*" \
   ./scripts/sweep.sh | tee results/sweep-console.log
   ```

4. Make the plots and `summary.csv`.

   ```bash
   python3 -m venv .venv && .venv/bin/pip install matplotlib numpy
   .venv/bin/python scripts/plot.py
   ```

Nothing else may be using the GPUs or port 8098 during a run. Don't rename or move the repo folder while a sweep is
running. The Xolotl server needs Python's `regex` package. Strata's `.venv` interpreter works for both servers, and
the scripts use it by default.

For a quick check at one length, use `scripts/run_all.sh`. It runs 2 passes per config at ~14K tokens.

The configs in `configs/` were generated for this machine and are kept for reference. Their paths are specific to
it.

## Files

```
scripts/build_xolotl.sh   clone + build Xolotl's strata engine
scripts/make_configs.py   base Strata config -> A/B/C configs
scripts/sweep.sh          context sweep: each config x prompt lengths x passes
scripts/plot.py           charts + summary.csv from the sweep (95% CIs)
scripts/bench.sh          one config: start server, probe twice, stop
scripts/run_all.sh        quick check: A, B, C at ~14K, 2 passes each
configs/                  the configs used here
docs/                     the charts
results/sweep/            context sweep: per-pass JSON, engine logs, summary.csv
results/2026-11-0*/       earlier 14K-only sessions
```
