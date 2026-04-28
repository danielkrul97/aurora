#!/usr/bin/env python3
"""Aurora training diagnostics — parse JSONL metrics + generate PNG plots.

Usage:
    python analysis/diagnostics.py logs/run.jsonl [--out plots/]

Generates four plots into the JSONL's directory (or --out):
  - bpb.png         train + val BPB over time + epoch boundaries
  - rates.png       per-layer firing rates over epochs
  - alphas.png      per-layer LSG α evolution over epochs
  - membrane.png    per-layer membrane mean/peak over epochs

Requires matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except ImportError:
    sys.stderr.write("error: matplotlib required (pip install matplotlib)\n")
    sys.exit(1)


def parse_jsonl(path: Path):
    runs = []
    chunks = []  # (epoch, chunk_idx, train_bpb, elapsed_s)
    epochs = []  # (epoch, train_bpb, val_bpb, rates, alphas, v_means, v_peaks, elapsed_s)
    checkpoints = []
    bad_lines = 0

    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError as e:
                bad_lines += 1
                sys.stderr.write(f"warn: line {line_no} not valid JSON ({e}); skipping\n")
                continue
            kind = ev.get("event")
            if kind == "run":
                runs.append(ev)
            elif kind == "chunk":
                chunks.append(
                    (ev["epoch"], ev["chunk"], ev["train_bpb"], ev["elapsed_s"])
                )
            elif kind == "epoch_done":
                epochs.append(ev)
            elif kind == "checkpoint":
                checkpoints.append(ev)
    if bad_lines:
        sys.stderr.write(
            f"warn: {bad_lines} malformed line(s) — usually NaN/Inf in metrics\n"
        )
    return runs, chunks, epochs, checkpoints


def plot_bpb(chunks, epochs, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Per-chunk train BPB (running mean within epoch).
    if chunks:
        # Cumulative time across all epochs (sum of elapsed_s within epoch
        # is OK approx for rolling x-axis; chunk's elapsed_s is from
        # epoch start).
        epoch_offsets = {}
        cum = 0.0
        for ep, _, _, _ in chunks:
            if ep not in epoch_offsets:
                epoch_offsets[ep] = cum
                cum += max(c[3] for c in chunks if c[0] == ep)
        xs, ys = [], []
        for ep, _, bpb, elapsed in chunks:
            xs.append((epoch_offsets[ep] + elapsed) / 60.0)
            ys.append(bpb)
        ax.plot(xs, ys, color="tab:blue", alpha=0.5, label="train (per-chunk)", lw=1)

    # Per-epoch train + val BPB.
    if epochs:
        ep_x = [ev["epoch"] + 1 for ev in epochs]
        train = [ev["train_bpb"] for ev in epochs]
        val = [ev["val_bpb"] for ev in epochs]
        # Use epoch index for these — different x scale, so put on twin axis.
        ax2 = ax.twiny()
        ax2.plot(ep_x, train, "o-", color="tab:blue", label="train (epoch end)", lw=2)
        ax2.plot(ep_x, val, "s-", color="tab:orange", label="val (epoch end)", lw=2)
        ax2.set_xlabel("epoch")
        ax2.legend(loc="upper right")

    ax.set_xlabel("training time (min)")
    ax.set_ylabel("BPB (bits per byte)")
    ax.set_title("BPB convergence")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_per_layer(epochs, key, ylabel, title, out_path: Path):
    if not epochs:
        return
    n_layers = len(epochs[0][key])
    fig, ax = plt.subplots(figsize=(10, 5))
    ep_x = [ev["epoch"] + 1 for ev in epochs]
    for l in range(n_layers):
        ys = [ev[key][l] for ev in epochs]
        ax.plot(ep_x, ys, "o-", label=f"layer {l}", lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  → {out_path}")


def plot_membrane(epochs, out_path: Path):
    if not epochs:
        return
    n_layers = len(epochs[0]["v_means"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep_x = [ev["epoch"] + 1 for ev in epochs]
    for l in range(n_layers):
        means = [ev["v_means"][l] for ev in epochs]
        peaks = [ev["v_peaks"][l] for ev in epochs]
        axes[0].plot(ep_x, means, "o-", label=f"layer {l}", lw=2)
        axes[1].plot(ep_x, peaks, "o-", label=f"layer {l}", lw=2)
    for ax, sub in zip(axes, ("mean v_pre", "peak v_pre")):
        ax.set_xlabel("epoch")
        ax.set_ylabel(sub)
        ax.set_title(sub)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.axhline(1.0, color="red", lw=0.5, alpha=0.5, label="θ")
    axes[1].set_yscale("symlog")
    fig.suptitle("Membrane potential per layer (peaks > 100 = runaway risk)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  → {out_path}")


def summarize(runs, chunks, epochs):
    print("\n=== Run summary ===")
    if runs:
        r = runs[0]
        print(f"  corpus    : {r['corpus']}  ({r['corpus_bytes']:,} bytes)")
        print(f"  vocab     : {r['vocab']}  ({r['n_tokens']:,} tokens)")
        print(f"  arch      : {r['hidden']} × 2 layers, chunk {r['chunk']}, lr {r['lr']}")
        print(f"  gpu       : {r['gpu']}")
        print(
            f"  epochs    : start {r['start_epoch']} → target {r['epochs']}"
        )
    print(f"  events    : {len(runs)} runs / {len(chunks)} chunks / {len(epochs)} epochs")
    if epochs:
        last = epochs[-1]
        best_val = min(ev["val_bpb"] for ev in epochs)
        print(f"  last val  : {last['val_bpb']:.3f} BPB (best {best_val:.3f})")
        print(f"  rates     : {last['rates']}")
        print(f"  alphas    : {last['alphas']}")
        print(f"  v peaks   : {last['v_peaks']}")
        if last["v_peaks"] and max(last["v_peaks"]) > 1000:
            print("  ⚠ membrane peaks > 1000 — runaway risk for longer runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="path to metrics JSONL file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory for PNGs (default: same dir as jsonl)",
    )
    args = parser.parse_args()

    if not args.jsonl.exists():
        sys.stderr.write(f"error: {args.jsonl} not found\n")
        sys.exit(1)

    out_dir = args.out or args.jsonl.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.jsonl.stem

    runs, chunks, epochs, checkpoints = parse_jsonl(args.jsonl)
    summarize(runs, chunks, epochs)

    print("\n=== Generating plots ===")
    plot_bpb(chunks, epochs, out_dir / f"{stem}_bpb.png")
    plot_per_layer(
        epochs,
        "rates",
        "firing rate",
        "Per-layer firing rate (target ≈ 0.10)",
        out_dir / f"{stem}_rates.png",
    )
    plot_per_layer(
        epochs,
        "alphas",
        "α (LSG width)",
        "Per-layer LSG surrogate width α",
        out_dir / f"{stem}_alphas.png",
    )
    plot_membrane(epochs, out_dir / f"{stem}_membrane.png")
    print()


if __name__ == "__main__":
    main()
