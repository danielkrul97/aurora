# Aurora

Custom LLM postavený na **closed-form continuous-time (CfC) Liquid Neural Network**
cell — neurobiologicky inspirovaná architektura odvozená z dynamiky C. elegans
(Hasani et al., Nature Machine Intelligence 2022, [arXiv:2106.13898](https://arxiv.org/abs/2106.13898)).
Implementace v Rustu, GPU přes [wgpu](https://wgpu.rs) (Vulkan / Metal / DX12).

> **Status:** Sprint 50+ — celý projekt zúžen na LNN-CfC po empirické validaci
> (Sprint 50 byla 2. statisticky významná architektura nad SSM baseline). Stará
> SNN/SSM cesta je odstraněna; viz `docs/11_implementacni_plan_v2.md` pro
> historii experimentů 26–50, a `docs/12_implementacni_plan_v3.md` pro aktivní plán.

## Architektura

Per timestep, per neuron `i`:

```
z_t       = [I_t, x_{t-1}]                  (concat input + recurrent state)
f[i]      = σ(b_f[i] + W_f[i,:] · z_t)       (gating logit)
g[i]      = tanh(b_g[i] + W_g[i,:] · z_t)    (fast sub-state)
h[i]      = tanh(b_h[i] + W_h[i,:] · z_t)    (slow sub-state)
τ[i]      = exp(log_τ[i])                    (per-neuron timescale, clamped)
α[i]      = σ(-f[i] · τ[i])                  (interpolation weight ∈ (0,1))
x[t][i]   = α[i] · g[i] + (1 - α[i]) · h[i]  (convex combo → x ∈ [-1, 1])
```

Smooth aktivace bez surrogate gradientu (na rozdíl od SNN), stav je z konstrukce
omezený na `[-1, 1]^H`, decay je per-neuron a **závisí na vstupu** v každém kroku
(nemá Hillis-Steele asociativitu — sekvenční T-loop, žádný parallel scan).

## Co je hotové

### Cell + tréninková infrastruktura
- **CPU implementace** (`src/lnn.rs`) — multi-layer stateful forward + reverse-time BPTT,
  Adam, gradient clip, batched parallel via rayon, finite-diff verifikované všechny grad cesty
  (`tests/lnn_smoke.rs`).
- **GPU implementace** (`src/gpu/lnn_gpu.rs`, `src/gpu/kernels_lnn.rs`) — WGSL kernely:
  forward (per-layer, batch-fused), softmax+CE, top-projection backward, per-layer reverse-time
  backward, per-batch grad reduction, Adam, log_τ clamp, GPU-side L2 norm + clip-rescale.
- **CPU↔GPU parita < 1e-3** přes forward, single-step backward+Adam, multi-step (10 kroků)
  drift bounded < 5e-3. Asserted v `tests/lnn_gpu_parity.rs`.
- **Tokenizace BPE** (`src/bpe.rs`) — linked-list trainer + min-heap encoder, O(N log N).
- **Checkpoint format `AURORA-LNN01`** (`src/checkpoint.rs`) — váhy + log_τ, auto-resume přes
  GPU↔CPU round-trip. Žádný backward compat se starými SNN formáty.

### Trénink
- **`aurora_train`** (`examples/aurora_train.rs`) — single entry point. Default GPU,
  `--cpu` fallback. BPE + chunked TBPTT + Adam + gradient clip + JSONL metriky +
  per-epoch sample generation + checkpoint save/load.
- **`scripts/overnight.sh`** — pre-flight checks + nohup + log path/PID.
- **JSONL structured metrics** (`--metrics path.jsonl`) — `run` / `chunk` / `epoch_done` /
  `checkpoint` events; per-event flush.
- **Python diagnostics** (`analysis/diagnostics.py`) — 4 PNG grafy: BPB curve, per-layer
  state magnitude `|x|`, per-layer gating α (mean + max-deviation), per-layer mean τ.

## Spuštění

```bash
# Build + testy
cargo test --release                                       # CPU testy (lnn_smoke + checkpoint)
cargo test --release --features gpu --test lnn_gpu_parity  # GPU parity (4 testy)

# Stažení korpusů (pro plnohodnotný trénink)
bash scripts/get_corpus.sh tiny             # 1.1 MB tiny-shakespeare
bash scripts/get_corpus.sh enwik8-10m       # 10 MB enwik8 subset
bash scripts/get_corpus.sh enwik8           # 100 MB enwik8 (Hutter Prize)

# Hlavní LM trénink (default GPU, BPE auto-train, checkpoint auto-resume)
cargo run --release --features gpu --example aurora_train -- \
  --corpus data/enwik8_10m.txt --vocab 4096 --hidden 512 \
  --layers 2 --epochs 25 --chunk 64 --lr 0.005 \
  --checkpoint data/enwik8_10m_h512_v4096.ckpt

# Vynucený CPU režim
cargo run --release --example aurora_train -- --cpu --hidden 64 --layers 2 --epochs 1

# Overnight wrapper (nohup + pre-flight + JSONL metrics)
HIDDEN=512 VOCAB=4096 bash scripts/overnight.sh 25

# Live monitoring
tail -f logs/<run>.log                    # human-readable
tail -f logs/<run>.jsonl | jq             # structured

# Post-mortem diagnostics → 4 PNG do logs/
python3 analysis/diagnostics.py logs/<run>.jsonl

# Standalone BPE training (jen tokenizer)
cargo run --release --example train_bpe -- data/tiny.txt 1024 data/tiny_1024.bpe
```

## Constraints

- **GPU:** wgpu 23 přes Vulkan/Metal/DX12 (no CUDA toolchain). Cílový hardware
  GTX 1650 / 4 GB VRAM. `MAX_HIDDEN = 1024` v WGSL workgroup memory.
- **Per-batch grad memory:** backward alokuje per-batch grad buffer pro váhy
  (B × n_h × n_z floatů per layer). Pro h=512, B=16 to je ~150 MB; pro h=1024 ~600 MB.
  Bounded jen `n_hidden`.
- **Žádný parallel scan:** CfC má input-dependent decay → T-loop sekvenční.
  Pro typické T=64 to není bottleneck (workgroup parallelism přes neurony).

## Dokumentace

- [`docs/12_implementacni_plan_v3.md`](./docs/12_implementacni_plan_v3.md) — **aktivní plán**
- [`docs/11_implementacni_plan_v2.md`](./docs/11_implementacni_plan_v2.md) — historie SNN+SSM era (Sprint 26–50)
- [`docs/10_implementacni_plan.md`](./docs/10_implementacni_plan.md) — hluboká historie (Sprint 0–25)
- [`docs/`](./docs/README.md) — průzkum SNN/LLM (8 tematických dokumentů, referenční)
