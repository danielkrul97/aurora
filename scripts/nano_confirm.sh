#!/usr/bin/env bash
# 10-hyp sprint — nano scale-confirm of the two screen winners.
#
# Tests whether canon (−0.24) and embed-lr (−0.10) from the d256/L4 proxy
# hold at the GOLD scale (nano d384/L6, 4 ep) and whether they STACK.
# Baseline should reproduce ~1.82 (the documented Muon-nano anchor), which
# validates the setup. Single seed first; add seeds if the winners hold.
# CPU only (features not WGSL-ported) — ~2.3 h/run × 4 ≈ 9 h.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=experiments/results/nano_confirm
mkdir -p "$OUT"
BIN=target/release/examples/aurora_train
CORPUS=data/enwik8_10m.txt
BPE=data/enwik8_10m_4096.bpe

# Matches scripts/screening_muon_overnight.sh (the 1.8197 ± 0.008 anchor).
COMMON="--corpus $CORPUS --bpe $BPE --vocab 4096 --d-model 384 --n-layer 6 --n-head 6 --d-ff 1536 \
  --seq-len 256 --batch-size 16 --epochs 4 --warmup-steps 2000 --grad-clip 1.0 \
  --lr 6e-4 --min-lr 6e-5 --weight-decay 0.1 --muon-lr 0.02 --muon-momentum 0.95 \
  --eval-windows 512 --seed 42"

names=(baseline canon embedlr combo)
flags=(""        "--canon-kernel 4" "--embed-lr-mult 3.0" "--canon-kernel 4 --embed-lr-mult 3.0")

for i in "${!names[@]}"; do
    name="${names[$i]}"
    echo "→ $name ${flags[$i]}"
    # shellcheck disable=SC2086
    $BIN $COMMON ${flags[$i]} --checkpoint "$OUT/${name}.ckpt" --metrics "$OUT/${name}.jsonl" \
        > "$OUT/${name}.log" 2>&1 || echo "  RUN FAILED (see $OUT/${name}.log)"
    grep epoch_done "$OUT/${name}.jsonl" 2>/dev/null | tail -1 | sed 's/^/  /' || true
done

echo
echo "=== nano confirm: best val_bpb (d384/L6, 4 ep, seed 42; anchor≈1.8197) ==="
base=$(grep -h epoch_done "$OUT/baseline.jsonl" 2>/dev/null | sed 's/.*val_bpb":\([0-9.]*\).*/\1/' | sort -g | head -1)
for name in "${names[@]}"; do
    best=$(grep -h epoch_done "$OUT/${name}.jsonl" 2>/dev/null | sed 's/.*val_bpb":\([0-9.]*\).*/\1/' | sort -g | head -1)
    if [[ -n "$best" && -n "$base" && "$name" != baseline ]]; then
        delta=$(awk -v a="$best" -v b="$base" 'BEGIN{printf "%+.4f", a-b}')
        printf "  %-10s %s  (Δ %s)\n" "$name" "$best" "$delta"
    else
        printf "  %-10s %s\n" "$name" "${best:-FAILED}"
    fi
done
