#!/usr/bin/env bash
# Sprint 62 — multi-seed Muon baseline on enwik8_10m, nano config.
#
# Establishes a reproducible reference run for the current optimizer
# (Muon hybrid + AdamW on embed/LN). Use this as the "before" for any
# future ablation: same config, three seeds, compare mean ± std val_bpb.
#
# Sprint 62 reference numbers (Muon hybrid, nano, 4 ep, enwik8_10m):
#   mean val_bpb 1.8197, std 0.0080, mean train_bpb 1.8164 (eff. zero overfit)
#   AdamW prior baseline at same config: val_bpb 1.9276 ± 0.0002
#
# Wall-clock estimate (GTX 1650): ~20 min/run × 3 ≈ 1h.
#
# Usage:
#   bash scripts/screening_muon_overnight.sh
#
# Outputs: experiments/results/sprint62-muon-overnight/muon_s<seed>.{ckpt,jsonl}

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=experiments/results/sprint62-muon-overnight
mkdir -p "$OUT"

CORPUS=data/enwik8_10m.txt
[[ -f "$CORPUS" ]] || { echo "error: $CORPUS missing — bash scripts/get_corpus.sh enwik8-10m" >&2; exit 1; }

BIN=target/release/examples/aurora_train
if [[ ! -x "$BIN" ]] || find src/ examples/ -newer "$BIN" -name '*.rs' | grep -q .; then
    echo "build: aurora_train (release, gpu)…"
    cargo build --release --features gpu --example aurora_train
fi

# Nano config; warmup=2000 << total_steps ensures cosine actually engages.
COMMON="--corpus $CORPUS --vocab 4096 --d-model 384 --n-layer 6 --n-head 6 --d-ff 1536 \
  --seq-len 256 --batch-size 16 --epochs 4 --warmup-steps 2000 --grad-clip 1.0 --gpu \
  --lr 6e-4 --min-lr 6e-5 --weight-decay 0.1 \
  --muon-lr 0.02 --muon-momentum 0.95"

for seed in 42 1337 7; do
    stem="muon_s${seed}"
    log="$OUT/${stem}.log"
    echo "→ $stem"
    $BIN $COMMON --seed "$seed" \
        --checkpoint "$OUT/${stem}.ckpt" \
        --metrics "$OUT/${stem}.jsonl" \
        > "$log" 2>&1
    grep epoch_done "$OUT/${stem}.jsonl" | tail -1
done

echo
echo "=== summary ==="
for f in "$OUT"/*.jsonl; do
    name=$(basename "$f" .jsonl)
    bpb=$(grep epoch_done "$f" | tail -1 | sed 's/.*val_bpb":\([0-9.]*\).*/\1/')
    echo "  $name: best val_bpb=$bpb"
done
