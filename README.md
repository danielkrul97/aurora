# Aurora

Custom LLM postavený na neurobiologicky inspirované architektuře spiking
neural network. Implementace v Rustu, kombinuje surrogate-gradient
trénink (BPTT, Adam) s biologicky-motivovanými prvky (gated recurrent
LIF, sparse spike compaction, learnable surrogate width).

> **Status:** primitivy a tréninková infrastruktura hotové, hlavní
> overnight benchmark na enwik8 v běhu. Aktuální BPB plato: **2.68**
> (tiny_200k, 1 epoch, hidden=1024, vocab=4096) — viz
> [`docs/11_implementacni_plan_v2.md`](./docs/11_implementacni_plan_v2.md)
> pro aktuální stav a další směřování. Reálný cíl po SDSA integraci:
> 1.30–1.40 BPB na enwik8 (SpikeGPT reference 1.26).

## Co je hotové

### Neuron a plasticita
- LIF neuron + dendritické kompartmenty + homeostatická adaptace prahu
- STDP (pair-based) a three-factor R-STDP s eligibility trace + neuromodulátorem
- Discrete-time `Network` i event-driven `EventNetwork` (~24× rychlejší při řídké aktivitě)
- Lazy eligibility commit, rayon paralelizace per-step, per-neuron modulátory pro per-output reward

### Surrogate gradient stack
- LIF + recurrent surrogate SNN s W_rec hidden→hidden, gradient clipping, Adam
- **Gated recurrent SNN** (LSTM-style per-neuron forget gate β = σ(W_β·x + W_β_rec·s + b))
- Subtract-on-spike membrane reset (ne hard reset)
- Mini-batched parallel BPTT přes rayon, externalizovaný grad buffer
- Stateful TBPTT s detach na chunk hranicích (carry across chunks)
- **Multi-layer gated SNN** (CPU + GPU) s cross-layer credit assignment
- **Per-layer learnable surrogate width α** (LSG, Lian et al. 2023) — CPU + GPU gradient
- **Per-neuron spike-rate l2 regularization** směrem k target firing rate (CPU + GPU)
- Positive `b_h` init (0.3) — fix pro "hidden vrstvy never fire" problém

### Tokenizace + LM
- **BPE tokenizer** (linked-list trainer + min-heap encoder, O(N log N) obě strany)
- 1.1 MB / 1024 vocab trénuje a enkóduje pod 1 s
- Token-level training APIs napříč CPU + GPU (`forward_backward_batch_tokens_stateful`)
- Vocab-agnostic `softmax_ce` GPU kernel (stride loops, ≥ 256 vocab)
- Byte-level char-LM cesta zachována (autoregressive generation, BPB metric)

### GPU stack (wgpu / WGSL, NVIDIA Vulkan / Metal / DX12)
- Forward megakernel (single dispatch covers entire chunk, workgroup-shared state)
- Sparse spike compaction (workgroup-shared atomic counter buildne `active_indices`)
- 2-layer fused megakernel (chains layers in-shader, batch-fused B workgroups)
- Workgroup_size 256, MAX_HIDDEN 1024, dispatch limit fix pro hidden=1024 × vocab=4096
- 14× backward kernels + Adam + gradient clipping na GPU
- Per-layer α gradient reduction kernel (LSG na GPU side)
- Spike-rate reduction + bias correction kernels (rate reg na GPU)
- CPU↔GPU parity test < 5e-2 přes všech 14 grad bufferů

### Tréninková infrastruktura
- **`aurora_train` example** — hlavní training entry point (GPU 2-layer + BPE + checkpointing + auto-resume + sample generation)
- **Checkpoint format v2** (binary `AURORA02`) — váhy + per-layer α, auto-resume, v1 backward compat
- **`scripts/overnight.sh`** — pre-flight checks + nohup + log path/PID
- **JSONL structured metrics** (`--metrics path.jsonl` flag) — run/chunk/epoch_done/checkpoint events, sub-second flush pro `tail -f`
- **Python diagnostics** (`analysis/diagnostics.py`) — 4 PNG plots: BPB curve, per-layer rates, α evolution, membrane mean/peak
- Anti-regrese: `tests/lm_quality_regression.rs` (CPU byte + BPE + checkpoint round-trip), `tests/gpu_parity.rs`

### Stand-alone moduly (zatím nezaintegrované)
- **Spiking attention building blocks** (`src/attention.rs`) — LayerNorm, LinearProj, SDPA, spike-gated SDPA. Ověřené central-difference numerickou kontrolou.
- **SDSA modul** (`src/sdsa.rs`) — Spike-Driven Self-Attention (Yao 2023): linear attention nad binary Q/K/V s carry mezi chunky. Plný forward/backward, 4 smoke testy. **Wiring do `MultiLayerGatedSnn` = Sprint 36+.**

## Spuštění

```bash
cargo test --release                                      # 49 testů (~1.5 s)
cargo test --release --features gpu --test gpu_parity     # 6 GPU parity testů

# Stažení korpusů
bash scripts/get_mnist.sh                  # MNIST do data/mnist/
bash scripts/get_corpus.sh tiny            # 1.1 MB tiny-shakespeare
bash scripts/get_corpus.sh enwik8-10m      # 10 MB enwik8 subset
bash scripts/get_corpus.sh enwik8          # 100 MB enwik8 (Hutter Prize)

# Hlavní LM trénink (GPU + BPE + checkpoint auto-resume)
cargo run --release --features gpu --example aurora_train -- \
  --corpus data/enwik8_10m.txt --vocab 4096 --hidden 1024 \
  --epochs 25 --chunk 64 --lr 0.002

# Overnight wrapper (nohup + pre-flight + JSONL metrics)
bash scripts/overnight.sh                  # default 25 epoch
bash scripts/overnight.sh 8                # custom

# Live monitoring běžícího tréninku
tail -f logs/<run>.log                     # human-readable
tail -f logs/<run>.jsonl | jq              # structured

# Post-mortem diagnostics → 4 PNG do logs/
python3 analysis/diagnostics.py logs/<run>.jsonl

# Standalone BPE training (jen tokenizer)
cargo run --release --example train_bpe -- data/tiny.txt 1024 data/tiny_1024.bpe

# MNIST varianty (R-STDP / surrogate / sequential)
cargo run --release --example mnist_train
cargo run --release --example mnist_eval
cargo run --release --example mnist_surrogate
cargo run --release --example seq_mnist

# Czech char-LM (legacy CPU example, UTF-8 corpus)
cargo run --release --example char_lm_gated_cz -- data/cesky.txt
```

## Dokumentace

- [`docs/11_implementacni_plan_v2.md`](./docs/11_implementacni_plan_v2.md) — **aktivní plán a stav** (Sprint 26+)
- [`docs/10_implementacni_plan.md`](./docs/10_implementacni_plan.md) — historický záznam (Sprint 0–25, zamrzlý kvůli velikosti 92 KB)
- [`docs/`](./docs/README.md) — průzkum SNN/LLM (8 tematických dokumentů)
- [`docs/09_aurora_neuron.md`](./docs/09_aurora_neuron.md) — design jednoho neuronu
