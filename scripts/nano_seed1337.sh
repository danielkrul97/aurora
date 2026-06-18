#!/usr/bin/env bash
# 10-hyp sprint — 2nd-seed spot-check of the combo winner (canon+embedlr).
# seed 42 gave baseline 1.7971 / combo 1.6612 (Δ −0.136). This re-runs at
# seed 1337 to rule out a seed-42 fluke before enshrining --tier4.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=experiments/results/nano_confirm
BIN=target/release/examples/aurora_train
COMMON="--corpus data/enwik8_10m.txt --bpe data/enwik8_10m_4096.bpe --vocab 4096 \
  --d-model 384 --n-layer 6 --n-head 6 --d-ff 1536 --seq-len 256 --batch-size 16 \
  --epochs 4 --warmup-steps 2000 --grad-clip 1.0 --lr 6e-4 --min-lr 6e-5 \
  --weight-decay 0.1 --muon-lr 0.02 --muon-momentum 0.95 --eval-windows 512 --seed 1337"

for spec in "baseline_s1337:" "combo_s1337:--canon-kernel 4 --embed-lr-mult 3.0"; do
    name="${spec%%:*}"; extra="${spec#*:}"
    echo "→ $name $extra"
    # shellcheck disable=SC2086
    $BIN $COMMON $extra --checkpoint "$OUT/${name}.ckpt" --metrics "$OUT/${name}.jsonl" \
        > "$OUT/${name}.log" 2>&1 || echo "  FAILED"
    grep epoch_done "$OUT/${name}.jsonl" 2>/dev/null | tail -1 | sed 's/^/  /' || true
done
echo
bb=$(grep -h epoch_done "$OUT/baseline_s1337.jsonl" 2>/dev/null | sed 's/.*val_bpb":\([0-9.]*\).*/\1/' | sort -g | head -1)
cc=$(grep -h epoch_done "$OUT/combo_s1337.jsonl"    2>/dev/null | sed 's/.*val_bpb":\([0-9.]*\).*/\1/' | sort -g | head -1)
echo "=== seed 1337: baseline=$bb combo=$cc  Δ=$(awk -v a="$cc" -v b="$bb" 'BEGIN{if(a&&b)printf "%+.4f",a-b}') ==="
echo "    (seed 42 was: baseline=1.7971 combo=1.6612 Δ=-0.1359)"
