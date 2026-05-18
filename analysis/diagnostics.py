#!/usr/bin/env python3
"""Aurora training diagnostics — parse JSONL metrics + generate PNG plots.

Usage:
    python analysis/diagnostics.py logs/run.jsonl [--out plots/]

Generates PNG plots into the JSONL's directory (or --out):
  - bpb.png             train + val BPB over time + epoch boundaries
  - bundle_norm.png     mean L2 norm of context HV per epoch
  - cleanup_top1.png    argmax-cosine codebook recovery rate per epoch

Schema: aurora_train.rs writes one JSON object per event, one event per
line. Events: `run`, `chunk`, `epoch_done`, `checkpoint`. Fields specific
to the HDC stack: `bundle_norm`, `cleanup_top1`.

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
    epochs = []  # event dicts with bundle_norm / cleanup_top1
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

    if chunks:
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

    if epochs:
        ep_x = [ev["epoch"] + 1 for ev in epochs]
        train = [ev["train_bpb"] for ev in epochs]
        val = [ev["val_bpb"] for ev in epochs]
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


def plot_scalar(epochs, key, ylabel, title, out_path: Path, hline=None):
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ep_x = [ev["epoch"] + 1 for ev in epochs]
    ys = [ev[key] for ev in epochs if key in ev]
    if not ys:
        return
    ax.plot(ep_x[: len(ys)], ys, "o-", lw=2, color="tab:purple")
    if hline is not None:
        ax.axhline(hline, color="red", lw=0.5, alpha=0.5, label=f"target ≈ {hline}")
        ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
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
        print(
            f"  arch      : HDC dim={r['dim']}, chunk={r['chunk']}, lr={r['lr']}, "
            f"stateful={r.get('stateful', False)}"
        )
        print(f"  backend   : {r['backend']}")
        print(f"  epochs    : target {r['epochs']}")
    print(f"  events    : {len(runs)} runs / {len(chunks)} chunks / {len(epochs)} epochs")
    if epochs:
        last = epochs[-1]
        best_val = min(ev["val_bpb"] for ev in epochs)
        print(f"  last val  : {last['val_bpb']:.3f} BPB (best {best_val:.3f})")
        if "bundle_norm" in last:
            print(f"  bundle    : {last['bundle_norm']:.3f}  (mean L2 of context HV)")
        if "cleanup_top1" in last:
            top1 = last["cleanup_top1"]
            print(f"  cleanup   : {top1:.4f}  (codebook argmax recovery, pure-HDC)")
            if top1 < 0.05:
                print(
                    "  ⚠ cleanup_top1 < 5% — codebook capacity likely saturated; "
                    "consider larger D or trainable codebook (S56+)"
                )


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
    plot_scalar(
        epochs,
        "bundle_norm",
        "mean ||context_hv||",
        "Bundle norm (≈ √D for stable random ±1 codebook)",
        out_dir / f"{stem}_bundle_norm.png",
    )
    plot_scalar(
        epochs,
        "cleanup_top1",
        "top-1 cleanup accuracy",
        "Codebook argmax recovery (pure-HDC signal)",
        out_dir / f"{stem}_cleanup_top1.png",
    )
    print()


if __name__ == "__main__":
    main()
