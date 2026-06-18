#!/usr/bin/env bash
# 10-hyp sprint — convergence test (CPU, 3 epochs, seed 42).
#
# The 1-epoch screen is biased toward early-convergence accelerators. This
# re-runs the survivors at 3 epochs to see which Δ PERSIST (real win) vs
# shrink toward zero (faster-start illusion). Also re-tests the two likely
# proxy artifacts with corrected settings:
#   ema  0.999 → 0.99   (decay was ≫ run length: 0.999^700 ≈ 0.49 init)
#   swiglu iso → full d_ff 1024 (drop the ⅔ narrowing confound)
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=experiments/results/screen10_c3
mkdir -p "$OUT"
BIN=target/release/examples/aurora_train
CORPUS=data/enwik8_10m.txt
BPE=data/enwik8_10m_4096.bpe

COMMON="--corpus $CORPUS --bpe $BPE --vocab 4096 --d-model 256 --n-layer 4 --n-head 4 --d-ff 1024 \
  --seq-len 256 --batch-size 16 --epochs 3 --warmup-steps 200 --grad-clip 1.0 \
  --lr 6e-4 --min-lr 6e-5 --weight-decay 0.1 --muon-lr 0.02 --muon-momentum 0.95 \
  --eval-windows 256 --seed 42"

names=(baseline embedlr parallel layerscale canon headbias ema99 swiglu_full)
flags=(""        "--embed-lr-mult 3.0" "--parallel-block" "--layerscale" "--canon-kernel 4" \
       "--head-bias" "--ema 0.99" "--swiglu")

for i in "${!names[@]}"; do
    name="${names[$i]}"
    echo "→ $name ${flags[$i]}"
    # shellcheck disable=SC2086
    $BIN $COMMON ${flags[$i]} --checkpoint "$OUT/${name}.ckpt" --metrics "$OUT/${name}.jsonl" \
        > "$OUT/${name}.log" 2>&1 || echo "  RUN FAILED (see $OUT/${name}.log)"
    grep epoch_done "$OUT/${name}.jsonl" 2>/dev/null | tail -1 | sed 's/^/  /' || true
done

echo
echo "=== convergence test: val_bpb @ 3 epochs vs the 1-epoch screen ==="
base=$(grep epoch_done "$OUT/baseline.jsonl" 2>/dev/null | tail -1 | sed 's/.*val_bpb":\([0-9.]*\).*/\1/')
for name in "${names[@]}"; do
    bpb=$(grep epoch_done "$OUT/${name}.jsonl" 2>/dev/null | tail -1 | sed 's/.*val_bpb":\([0-9.]*\).*/\1/')
    if [[ -n "$bpb" && -n "$base" && "$name" != baseline ]]; then
        delta=$(awk -v a="$bpb" -v b="$base" 'BEGIN{printf "%+.4f", a-b}')
        printf "  %-12s %s  (Δ3ep %s)\n" "$name" "$bpb" "$delta"
    else
        printf "  %-12s %s\n" "$name" "${bpb:-FAILED}"
    fi
done
