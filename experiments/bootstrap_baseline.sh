#!/usr/bin/env bash
# Establish the frozen baseline metric used by the autonomous research loop.
# Run ONCE per machine before invoking /loop.
#
# Usage:
#   bash experiments/bootstrap_baseline.sh        # honors existing baseline
#   bash experiments/bootstrap_baseline.sh --force  # re-establish

set -euo pipefail

cd "$(dirname "$0")/.."

CFG="experiments/baseline_config.json"
OUT="experiments/baseline.jsonl"
CKPT="experiments/baseline.ckpt"
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

if [[ ! -f "$CFG" ]]; then
    echo "error: $CFG missing" >&2
    exit 1
fi

if [[ -f "$OUT" && $FORCE -eq 0 ]]; then
    EXISTING=$(grep '"event":"epoch_done"' "$OUT" | tail -1 | jq -r '.val_bpb' 2>/dev/null || echo "?")
    echo "baseline exists: $OUT (val_bpb=$EXISTING)"
    echo "pass --force to re-run"
    exit 0
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq not found. install: sudo apt install jq" >&2
    exit 1
fi

CORPUS=$(jq -r '.cli_args | index("--corpus") as $i | .[$i + 1]' "$CFG")
if [[ ! -f "$CORPUS" ]]; then
    echo "error: corpus $CORPUS missing. fetch via: bash scripts/get_corpus.sh enwik8-10m" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1 && ! nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi failed. check GPU driver." >&2
    exit 1
fi

echo "build: aurora_train (release, gpu)"
cargo build --release --features gpu --example aurora_train

mapfile -t ARGS < <(jq -r '.cli_args[]' "$CFG")

echo "run: baseline (~$(jq -r '.expected_runtime_min' "$CFG") min expected)"
echo "args: ${ARGS[*]}"

./target/release/examples/aurora_train \
    "${ARGS[@]}" \
    --metrics "$OUT" \
    --checkpoint "$CKPT"

VAL_BPB=$(grep '"event":"epoch_done"' "$OUT" | tail -1 | jq -r '.val_bpb')
echo
echo "baseline established: val_bpb=$VAL_BPB"
echo "metrics: $OUT"
echo "next step: invoke /loop with experiments/RESEARCHER.md"
