#!/usr/bin/env python3
"""Iso-FLOP sweep summary: collect (params, tokens, FLOPs, val BPB) from
logs/isoflop/*.jsonl and print the frontier table; optionally plot
BPB vs FLOPs per model size.

Usage: python3 analysis/isoflop.py [logs/isoflop] [--plot out.png]
"""

import json
import sys
from pathlib import Path


def load_run(path: Path):
    meta, last_done, last_step = None, None, None
    with open(path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "run":
                meta = ev
            elif ev.get("event") == "epoch_done":
                last_done = ev
            elif ev.get("event") == "step":
                last_step = ev
    if not meta or not last_done:
        return None
    params = meta["params"]
    # Tokens actually consumed: max-steps cap × batch × seq when capped,
    # else epochs × full stream.
    steps = last_step["step"] if last_step else 0
    max_steps = meta.get("max_steps", 0)
    if max_steps:
        steps = max_steps
    tokens = steps * meta["batch_size"] * meta["seq_len"]
    flops = 6.0 * params * tokens
    return {
        "run": path.stem,
        "params": params,
        "d_model": meta["d_model"],
        "n_layer": meta["n_layer"],
        "tokens": tokens,
        "flops": flops,
        "val_bpb": last_done["val_bpb"],
        "train_bpb": last_done["train_bpb"],
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0]) if args else Path("logs/isoflop")
    plot_out = None
    if "--plot" in sys.argv:
        i = sys.argv.index("--plot")
        plot_out = sys.argv[i + 1] if i + 1 < len(sys.argv) else "screenshots/isoflop_curve.png"

    runs = []
    for p in sorted(root.glob("*.jsonl")):
        r = load_run(p)
        if r:
            runs.append(r)
        else:
            print(f"  (skipping incomplete {p.name})", file=sys.stderr)
    if not runs:
        print("no completed runs found", file=sys.stderr)
        return 1

    runs.sort(key=lambda r: (r["flops"], r["params"]))
    print(f"{'run':<16} {'params':>8} {'tokens':>8} {'PFLOPs':>7} {'val_bpb':>8} {'train_bpb':>9}")
    for r in runs:
        print(
            f"{r['run']:<16} {r['params'] / 1e6:>7.2f}M {r['tokens'] / 1e6:>7.1f}M "
            f"{r['flops'] / 1e15:>7.2f} {r['val_bpb']:>8.3f} {r['train_bpb']:>9.3f}"
        )

    # Per-budget winners (group by FLOPs within 15 % — the sweep's rows).
    print("\nfrontier (best val_bpb per FLOP budget):")
    used = [False] * len(runs)
    for i, r in enumerate(runs):
        if used[i]:
            continue
        group = [r]
        used[i] = True
        for j in range(i + 1, len(runs)):
            if not used[j] and abs(runs[j]["flops"] - r["flops"]) / r["flops"] < 0.15:
                group.append(runs[j])
                used[j] = True
        best = min(group, key=lambda g: g["val_bpb"])
        print(
            f"  ~{best['flops'] / 1e15:>5.2f} PF → {best['run']} "
            f"({best['params'] / 1e6:.1f}M params, val {best['val_bpb']:.3f})"
        )

    if plot_out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        by_size = {}
        for r in runs:
            by_size.setdefault(r["d_model"], []).append(r)
        for d, rs in sorted(by_size.items()):
            rs.sort(key=lambda r: r["flops"])
            ax.plot(
                [r["flops"] for r in rs],
                [r["val_bpb"] for r in rs],
                "o-",
                label=f"d={d} ({rs[0]['params'] / 1e6:.1f}M)",
            )
        ax.set_xscale("log")
        ax.set_xlabel("training FLOPs (6·N·D)")
        ax.set_ylabel("val BPB (enwik8, sliding eval)")
        ax.set_title("Aurora iso-FLOP sweep — GTX 1650")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        Path(plot_out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_out, dpi=150)
        print(f"\nplot → {plot_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
