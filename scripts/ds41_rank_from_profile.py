#!/usr/bin/env python3
"""Turn a Strata DeepSeek-V4.1 pack's expert-profile.bin into a ds41_generate --rank file.

expert-profile.bin is one uint32 routing count per (layer, expert), layer-major (40 x 384). ds41_generate --rank wants
int32 (layer, expert) pairs, hottest first; it seeds the VRAM cache and the RAM arena in that order. Without --rank the
engine seeds in plain index order (layer 0's experts first). Using Strata's own profile means both engines start from
the same expert statistics.

    ds41_rank_from_profile.py <pack>/expert-profile.bin out.bin [--layers 40] [--experts 384]
"""
import argparse
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, default=384)
    a = ap.parse_args()
    raw = open(a.profile, "rb").read()
    n = a.layers * a.experts
    if len(raw) != 4 * n:
        raise SystemExit(f"{a.profile}: {len(raw)} bytes, expected {4 * n} ({a.layers} x {a.experts} uint32)")
    counts = struct.unpack(f"<{n}I", raw)
    order = sorted(range(n), key=lambda i: (-counts[i], i))
    with open(a.out, "wb") as f:
        for i in order:
            f.write(struct.pack("<ii", i // a.experts, i % a.experts))
    total = sum(counts)
    top = sum(counts[i] for i in order[: n // 4])
    print(f"{a.out}: {n} experts ranked; the top quarter takes {100 * top / max(total, 1):.1f}% of routed tokens")


if __name__ == "__main__":
    main()
