#!/usr/bin/env bash
# Spustí aurora_train overnight benchmark na enwik8_10m s GPU.
# Výstup do logs/overnight_<timestamp>.log; checkpoint do
# data/enwik8_10m_h1024_v4096.ckpt (auto-resume při dalším běhu).
#
# Usage:
#   bash scripts/overnight.sh           # default 25 epoch
#   bash scripts/overnight.sh 8         # custom epoch count
#
# Po spuštění proces běží na pozadí (nohup). Sleduj přes:
#   tail -f logs/overnight_<timestamp>.log
# Stop:
#   kill $(pgrep -f aurora_train)

set -euo pipefail

cd "$(dirname "$0")/.."

EPOCHS="${1:-25}"
CORPUS="data/enwik8_10m.txt"
CHECKPOINT="data/enwik8_10m_h1024_v4096.ckpt"

# --- Pre-flight checks ---

if [[ ! -f "$CORPUS" ]]; then
    echo "error: $CORPUS chybí. Stáhni přes: bash scripts/get_corpus.sh enwik8-10m" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "warning: nvidia-smi nenalezen — GPU dostupnost neověřena"
else
    if ! nvidia-smi >/dev/null 2>&1; then
        echo "error: nvidia-smi selhal. Zkontroluj driver." >&2
        exit 1
    fi
fi

# Build pokud binary chybí nebo je starší než zdrojáky
BIN="target/release/examples/aurora_train"
if [[ ! -x "$BIN" ]] || find src/ examples/ -newer "$BIN" -name '*.rs' | grep -q .; then
    echo "build: aurora_train (release, gpu)…"
    cargo build --release --features gpu --example aurora_train
fi

# --- Run ---

mkdir -p logs
LOG="logs/overnight_$(date +%Y%m%d_%H%M).log"

nohup "$BIN" \
    --corpus "$CORPUS" \
    --vocab 4096 \
    --hidden 1024 \
    --epochs "$EPOCHS" \
    --chunk 64 \
    --lr 0.002 \
    --checkpoint "$CHECKPOINT" \
    > "$LOG" 2>&1 &

PID=$!

# Krátké čekání + ověření že proces opravdu běží (chytí early crash)
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "error: aurora_train nestartoval — viz $LOG"
    tail -20 "$LOG" >&2
    exit 1
fi

echo "started: PID $PID"
echo "log:     $LOG"
echo "ckpt:    $CHECKPOINT"
echo "epochs:  $EPOCHS  (~$(( EPOCHS * 17 )) min odhad pro 10 MB / hidden 1024)"
echo
echo "monitor:"
echo "  tail -f $LOG"
echo "  watch -n 30 nvidia-smi"
echo "  grep -E 'epoch done|val bpb|rates|α \\[' $LOG"
echo
echo "stop:"
echo "  kill $PID"
