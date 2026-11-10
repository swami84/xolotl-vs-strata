#!/usr/bin/env python3
"""Plot the context-length sweep: prefill and decode speed vs prompt length, mean with 95% CI, plus Xolotl's speedup.

    .venv/bin/python scripts/plot.py [SWEEP_DIR] [--series "cfg=Label;cfg=Label;..."] [--prefix P] [--title T]

Reads <sweep>/<config>/<prompt>-<pass>.json (written by sweep.sh) and writes:
  <sweep>/summary.csv            one row per config x prompt length (the table view of the charts)
  docs/<P>speed_vs_context.png   prefill and decode tok/s vs prompt tokens, mean +/- 95% CI (t distribution)
  docs/<P>speedup_vs_context.png each series / the first series, ratio of means, 95% bootstrap CI
The first series is the baseline (Strata). Defaults are the QFN sweep. Needs matplotlib and numpy.
"""
import argparse
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("sweep", nargs="?", default=os.path.join(HERE, "results", "sweep"))
ap.add_argument("--series", default="A-strata=Strata;B-xolotl-same=Xolotl (same flags);C-xolotl-tuned=Xolotl (tuned)")
ap.add_argument("--prefix", default="")
ap.add_argument("--title", default="Qwen3.8-Flash-Next IQ3_S on 2x RTX 3090: Strata vs Xolotl")
ap.add_argument("--speedup-legend", default="upper right", help="legend position on the speedup chart")
ap.add_argument("--split-labels", action="store_true", help="speedup chart: label the second series' end point on the left")
ARGS = ap.parse_args()
SWEEP = os.path.abspath(ARGS.sweep)
DOCS = os.path.join(HERE, "docs")

# Series: fixed order, fixed color, fixed marker (identity never color-alone). Palette validated with the dataviz
# skill's validate_palette.js (light): all checks pass; aqua is < 3:1 on the surface, so every line is direct-labelled.
STYLES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]
SERIES = [(cfg, label, *STYLES[i]) for i, (cfg, label) in
          enumerate(s.split("=", 1) for s in ARGS.series.split(";"))]
BASE = SERIES[0][0]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def load():
    data = {}  # cfg -> ctx -> list of rows
    for cfg, *_ in SERIES:
        for f in glob.glob(os.path.join(SWEEP, cfg, "*-*.json")):
            name = os.path.basename(f)[:-5]
            if not name.split("-")[0].isdigit():
                continue
            ctx = int(name.split("-")[0])
            row = json.load(open(f))
            data.setdefault(cfg, {}).setdefault(ctx, []).append(row)
    return data


def mean_ci(xs):
    xs = np.asarray(xs, float)
    n = len(xs)
    if n == 0:
        return np.nan, np.nan, 0
    m = xs.mean()
    if n == 1:
        return m, np.nan, 1
    return m, T95.get(n - 1, 1.96) * xs.std(ddof=1) / np.sqrt(n), n


def ratio_ci(num, den, rng, b=10000):
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) < 2 or len(den) < 2:
        return num.mean() / den.mean(), np.nan, np.nan
    r = rng.choice(num, (b, len(num))).mean(1) / rng.choice(den, (b, len(den))).mean(1)
    return num.mean() / den.mean(), *np.percentile(r, [2.5, 97.5])


def style(ax, title, ylabel, ticks):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="semibold", pad=10)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_xlabel("prompt length (tokens)", color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [f"{t / 1000:.1f}K" if t < 10000 else f"{t / 1000:.0f}K" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(min(ticks) / 1.4, max(ticks) * 1.5)


def main():
    data = load()
    if not data:
        sys.exit(f"no sweep results under {SWEEP}")
    os.makedirs(DOCS, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    stats = {}  # (cfg, metric) -> list of (x, mean, ci)
    for cfg, label, *_ in SERIES:
        for ctx in sorted(data.get(cfg, {})):
            ok = [r for r in data[cfg][ctx] if not r.get("failed")]
            failed = len(data[cfg][ctx]) - len(ok)
            x = np.mean([r["prefill_tokens"] for r in ok]) if ok else ctx * 0.87
            pm, pc, n = mean_ci([r["prefill_tok_s"] for r in ok])
            dm, dc, _ = mean_ci([r["decode_tok_s"] for r in ok])
            dt = np.mean([r["decode_tokens"] for r in ok]) if ok else np.nan
            stats.setdefault((cfg, "prefill"), []).append((x, pm, pc))
            stats.setdefault((cfg, "decode"), []).append((x, dm, dc))
            stats.setdefault((cfg, "decode_n"), []).append(dt)
            rows.append({"config": cfg, "engine": label, "prompt_target": ctx, "prompt_tokens": round(x),
                         "passes_ok": n, "passes_failed": failed,
                         "prefill_tok_s": round(float(pm), 1), "prefill_ci95": round(float(pc), 1),
                         "decode_tok_s": round(float(dm), 1), "decode_ci95": round(float(dc), 1),
                         "decode_tokens_mean": int(round(float(dt))) if dt == dt else None})

    # speedups vs Strata, per context
    base = data.get(BASE, {})
    for cfg, label, *_ in SERIES[1:]:
        for ctx in sorted(data.get(cfg, {})):
            if ctx not in base:
                continue
            a = [r for r in base[ctx] if not r.get("failed")]
            c = [r for r in data[cfg][ctx] if not r.get("failed")]
            if not a or not c:
                continue
            x = np.mean([r["prefill_tokens"] for r in c])
            for metric in ("prefill", "decode"):
                k = metric + "_tok_s"
                stats.setdefault((cfg, metric + "_x"), []).append(
                    (x, *ratio_ci([r[k] for r in c], [r[k] for r in a], rng)))
            for row in rows:
                if row["config"] == cfg and row["prompt_target"] == ctx:
                    for metric in ("prefill", "decode"):
                        _, m, lo, hi = stats[(cfg, metric + "_x")][-1]
                        row[f"{metric}_speedup"] = round(float(m), 3)
                        row[f"{metric}_speedup_ci95"] = f"{lo:.3f}-{hi:.3f}"

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(SWEEP, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    ticks = [round(x, -2) for x, *_ in stats[(SERIES[0][0], "prefill")]]
    # baseline decode points whose answers stopped well short of the longest answer (the requested length): hollow
    longest = max((r["decode_tokens_mean"] or 0) for r in rows)
    short = [n < 0.95 * longest for n in stats.get((BASE, "decode_n"), [])]

    # Figure 1: speed vs context, two panels (separate y axes per panel, never a dual axis)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    for ax, metric, title in ((axes[0], "prefill", "Prefill speed"), (axes[1], "decode", "Decode speed")):
        style(ax, title, "tokens / second (mean, 95% CI)", ticks)
        for cfg, label, color, marker in SERIES:
            pts = stats.get((cfg, metric))
            if not pts:
                continue
            x, m, c = (np.array(v) for v in zip(*pts))
            ax.fill_between(x, m - c, m + c, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, m, color=color, linewidth=2, marker=marker, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label)
            if metric == "decode" and cfg == BASE and any(short):
                hx, hm = x[np.array(short)], m[np.array(short)]
                ax.plot(hx, hm, linestyle="none", marker=marker, markersize=7, markerfacecolor=SURFACE,
                        markeredgecolor=color, markeredgewidth=2)
                ax.annotate("hollow = Strata stopped\nearly (~300 tokens)", (hx[0], hm[0]), xytext=(0, -30),
                            textcoords="offset points", ha="center", va="top", fontsize=8, color=INK2)
            ax.annotate(f"{m[-1]:,.0f}", (x[-1], m[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.suptitle(ARGS.title, x=0.01, ha="left",
                 color=INK, fontsize=13, fontweight="bold")
    lo_n, hi_n = min(r["passes_ok"] for r in rows), max(r["passes_ok"] for r in rows)
    passes = f"{lo_n}" if lo_n == hi_n else f"{lo_n}-{hi_n}"
    fig.text(0.01, 0.005, f"Shaded band = 95% CI (t distribution) over {passes} passes; numbers = mean at the longest "
             f"prompt. Decode asks for {longest:,} tokens.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(DOCS, ARGS.prefix + "speed_vs_context.png"), facecolor=SURFACE)
    plt.close(fig)

    # Figure 2: speedup vs context
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150, sharey=True)
    for ax, metric, title in ((axes[0], "prefill", "Prefill speedup vs Strata"),
                              (axes[1], "decode", "Decode speedup vs Strata")):
        style(ax, title, "Xolotl / Strata (ratio of means)", ticks)
        ax.axhline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        for (cfg, label, color, marker), dodge in zip(SERIES[1:], (0.95, 1.05)):
            pts = stats.get((cfg, metric + "_x"))
            if not pts:
                continue
            x, m, lo, hi = (np.array(v) for v in zip(*pts))
            x = x * dodge   # side by side so the two series' error bars don't overlap
            ax.errorbar(x, m, yerr=[m - lo, hi - m], color=color, linewidth=2, marker=marker, markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=2, capsize=4, elinewidth=1.5, label=label)
            if metric == "decode" and any(short):
                ax.plot(x[np.array(short)], m[np.array(short)], linestyle="none", marker=marker, markersize=7,
                        markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=2, zorder=5)
            # the left-dodged series labels to the left so the two end labels never collide
            left = ARGS.split_labels and dodge < 1
            ax.annotate(f"{(m[-1] - 1) * 100:+.0f}%", (x[-1], m[-1]), xytext=(-10 if left else 10, 0),
                        textcoords="offset points", ha="right" if left else "left", va="center", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc=ARGS.speedup_legend)
    fig.text(0.01, 0.005, "Error bars = 95% bootstrap CI of the ratio of means (10,000 resamples). Dashed line = "
             "no change." + (" Hollow = Strata's answer stopped early, so lengths are not matched." if any(short) else ""),
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(DOCS, ARGS.prefix + "speedup_vs_context.png"), facecolor=SURFACE)
    plt.close(fig)

    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
