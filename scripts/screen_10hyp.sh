#!/usr/bin/env bash
# 10-hypothesis sprint — first directional screen (CPU, single seed, 1 epoch).
#
# Each run = plain Muon baseline + exactly ONE feature flag, so the delta
# isolates that feature's marginal effect vs the same baseline. This is a
# PROXY config (d256/L4 ≈ 4.2M, 1 epoch) for ranking only — survivors get
# re-confirmed at nano scale (d384/L6, 4 ep, 3 seeds) vs 1.8197 ± 0.008.
#
# CPU only: the features live in src/transformer (the --gpu path is separate
# WGSL and ignores them). ~20 min/run × 11 ≈ 3.5 h. Safe to Ctrl-C / kill.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=experiments/results/screen10
mkdir -p "$OUT"
BIN=target/release/examples/aurora_train
CORPUS=data/enwik8_10m.txt
BPE=data/enwik8_10m_4096.bpe

[[ -x "$BIN" ]] || { echo "build first: cargo build --release --features gpu --example aurora_train" >&2; exit 1; }
[[ -f "$CORPUS" ]] || { echo "missing $CORPUS" >&2; exit 1; }

COMMON="--corpus $CORPUS --bpe $BPE --vocab 4096 --d-model 256 --n-layer 4 --n-head 4 --d-ff 1024 \
  --seq-len 256 --batch-size 16 --epochs 1 --warmup-steps 200 --grad-clip 1.0 \
  --lr 6e-4 --min-lr 6e-5 --weight-decay 0.1 --muon-lr 0.02 --muon-momentum 0.95 \
  --eval-windows 512 --seed 42"

# name -> extra flag(s). swiglu shrinks d_ff to ≈2/3 to stay iso-param.
names=(baseline zloss embedlr headbias parallel layerscale ema swiglu canon diffattn sink)
flags=(""        "--z-loss 1e-4" "--embed-lr-mult 3.0" "--head-bias" "--parallel-block" \
       "--layerscale" "--ema 0.999" "--swiglu --d-ff 688" "--canon-kernel 4" "--diff-attn" "--sink 4")

for i in "${!names[@]}"; do
    name="${names[$i]}"
    echo "→ $name ${flags[$i]}"
    # shellcheck disable=SC2086
    $BIN $COMMON ${flags[$i]} --checkpoint "$OUT/${name}.ckpt" --metrics "$OUT/${name}.jsonl" \
        > "$OUT/${name}.log" 2>&1 || echo "  RUN FAILED (see $OUT/${name}.log)"
    grep epoch_done "$OUT/${name}.jsonl" 2>/dev/null | tail -1 | sed 's/^/  /' || true
done

echo
echo "=== screen10 summary: val_bpb (1 epoch, d256/L4, seed 42) ==="
base=$(grep epoch_done "$OUT/baseline.jsonl" 2>/dev/null | tail -1 | sed 's/.*val_bpb":\([0-9.]*\).*/\1/')
for name in "${names[@]}"; do
    bpb=$(grep epoch_done "$OUT/${name}.jsonl" 2>/dev/null | tail -1 | sed 's/.*val_bpb":\([0-9.]*\).*/\1/')
    if [[ -n "$bpb" && -n "$base" && "$name" != baseline ]]; then
        delta=$(awk -v a="$bpb" -v b="$base" 'BEGIN{printf "%+.4f", a-b}')
        printf "  %-12s %s  (Δ %s)\n" "$name" "$bpb" "$delta"
    else
        printf "  %-12s %s\n" "$name" "${bpb:-FAILED}"
    fi
done
