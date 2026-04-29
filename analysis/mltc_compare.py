#!/usr/bin/env python3
"""M-LTC vs CfC baseline comparison: overlay τ, α, |x|, val_bpb across runs.

Usage:
    python analysis/mltc_compare.py logs/mltc/baseline.jsonl logs/mltc/mltc25.jsonl ... --out screenshots/mltc.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.stderr.write("error: matplotlib required (pip install matplotlib)\n")
    sys.exit(1)


def load_run(path: Path):
    meta = None
    epochs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("event") == "run":
                meta = ev
            elif ev.get("event") == "epoch_done":
                epochs.append(ev)
    if meta is None:
        raise ValueError(f"{path}: no 'run' event")
    return meta, epochs


def label_for(meta):
    mf = meta.get("memory_frac", 0.0)
    if mf == 0.0:
        return "baseline (CfC)"
    return f"M-LTC mf={mf:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("screenshots/mltc_compare.png"))
    ap.add_argument("--title", default="M-LTC vs CfC baseline")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    runs = [load_run(p) for p in args.logs]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(args.title, fontsize=14)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    def plot_metric(ax, key_or_fn, title, ylabel, hline=None):
        for (meta, epochs), color in zip(runs, colors):
            if not epochs:
                continue
            xs = [e["epoch"] + 1 for e in epochs]
            if callable(key_or_fn):
                ys = [key_or_fn(e) for e in epochs]
            else:
                ys = [e[key_or_fn] for e in epochs]
            ax.plot(xs, ys, "o-", color=color, label=label_for(meta), lw=2)
        if hline is not None:
            ax.axhline(hline, color="red", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    plot_metric(axes[0, 0], "val_bpb", "val BPB (lower = better)", "bpb")
    plot_metric(axes[0, 1], "train_bpb", "train BPB", "bpb")
    plot_metric(
        axes[0, 2],
        lambda e: e["tau_means"][1] if len(e["tau_means"]) > 1 else e["tau_means"][0],
        "τ mean (top layer)",
        "τ",
        hline=0.35,
    )
    plot_metric(
        axes[1, 0],
        lambda e: e["activity"][1] if len(e["activity"]) > 1 else e["activity"][0],
        "|x| top layer (saturation > 0.95)",
        "mean |x|",
        hline=0.95,
    )
    plot_metric(
        axes[1, 1],
        lambda e: e["alpha_devs"][1] if len(e["alpha_devs"]) > 1 else e["alpha_devs"][0],
        "α deviation top layer",
        "max |α − 0.5|",
        hline=0.45,
    )
    plot_metric(
        axes[1, 2],
        lambda e: e["alpha_means"][1] if len(e["alpha_means"]) > 1 else e["alpha_means"][0],
        "α mean top layer",
        "mean α",
        hline=0.5,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
