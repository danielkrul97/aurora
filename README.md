# Aurora

Custom LLM postavený na **bipolar Hyperdimensional Computing** (HDC / VSA — Kanerva
2009 MAP). Token = ±1 hypervektor v `R^D`; sekvence = position-binding bundle
(`accum_t = sum_{s≤t} P^s(token_hv_s)`); klasifikace = trainable lineární readout
nad context HV. Implementace v Rustu, GPU přes [wgpu](https://wgpu.rs)
(Vulkan / Metal / DX12).

> **Status:** Sprint 52+ — projekt se otočil z LNN-CfC na HDC po refactoru. Stará
> LNN cesta je odstraněna. Historie LNN é v `docs/12_implementacni_plan_v3.md`
> (Sprint 51 LNN-only era), HDC pivot navazuje sekcí Sprint 52+.

## Architektura

Jeden chunk délky `T`, počáteční stav `(accum_0, age_0)`:

```
for t in 0..T:
    pos     = age_0 + t
    bound_t = P^pos(codebook[input_t])     (cyclic shift permute)
    accum_t = accum_{t-1} + bound_t        (bundle accumulator)
    ctx_t   = accum_t / sqrt(pos + 1)      (norm-stable context HV)
    logits_t = b + W · ctx_t               (linear readout)
state' = (accum_T, age_0 + T)
```

`bind` a `bundle` jsou differentiable, takže readout grad teče zpět do `ctx_t`
přirozeně; pro frozen codebook (S52–S55) se ale zastaví na úrovni readoutu.
Trainable codebook je naplánovaný jako S56+ ablace.

## Co je hotové

### Cell + tréninková infrastruktura
- **CPU implementace** (`src/hdc/`) — codebook (item memory), bind/bundle/permute
  primitivy, trainable readout s Adam, batched parallel via rayon, finite-diff
  verifikované grad cesty (`tests/hdc_smoke.rs`).
- **GPU implementace** (`src/gpu/hdc_gpu.rs`, `src/gpu/kernels_hdc.rs`) — WGSL kernely:
  encoder forward (per-batch workgroup, T-loop fused), linear forward + backward,
  softmax+CE, per-batch grad reduction, Adam, GPU-side L2 norm + clip-rescale.
  Sdílené utility kernely v `src/gpu/kernels_common.rs`.
- **CPU↔GPU parita < 1e-3** přes forward (logits + contexts), single-step
  backward+Adam, multi-step (10 kroků) drift bounded < 5e-3, state carry parity
  napříč chunks. Asserted v `tests/hdc_gpu_parity.rs`.
- **Tokenizace BPE** (`src/bpe.rs`) — linked-list trainer + min-heap encoder, O(N log N).
- **Checkpoint format `AURORA-HDC01`** (`src/checkpoint.rs`) — codebook + readout,
  auto-resume přes GPU↔CPU round-trip. Žádný backward compat se staršími formáty.

### Trénink
- **`aurora_train`** (`examples/aurora_train.rs`) — single entry point. Default GPU,
  `--cpu` fallback. BPE + chunked encoder + Adam + gradient clip + JSONL metriky +
  per-epoch sample generation + checkpoint save/load. `--stateful` pro carry
  bundle accumulator napříč chunks v rámci epochy.
- **`scripts/overnight.sh`** — pre-flight checks + nohup + log path/PID.
- **JSONL structured metrics** (`--metrics path.jsonl`) — `run` / `chunk` / `epoch_done` /
  `checkpoint` events; per-event flush. Per-epoch HDC fields: `bundle_norm`,
  `cleanup_top1` (argmax cosine codebook recovery rate).
- **Python diagnostics** (`analysis/diagnostics.py`) — PNG grafy: BPB curve,
  bundle norm history, cleanup top-1 accuracy.

## Spuštění

```bash
# Build + testy
cargo test --release                                       # CPU testy (hdc_smoke + checkpoint)
cargo test --release --features gpu --test hdc_gpu_parity  # GPU parity (5 testů)

# Stažení korpusů (pro plnohodnotný trénink)
bash scripts/get_corpus.sh tiny             # 1.1 MB tiny-shakespeare
bash scripts/get_corpus.sh enwik8-10m       # 10 MB enwik8 subset
bash scripts/get_corpus.sh enwik8           # 100 MB enwik8 (Hutter Prize)

# Hlavní LM trénink (default GPU, BPE auto-train, checkpoint auto-resume)
cargo run --release --features gpu --example aurora_train -- \
  --corpus data/enwik8_10m.txt --vocab 4096 --dim 4096 \
  --epochs 25 --chunk 64 --lr 0.002 \
  --checkpoint data/enwik8_10m_d4096_v4096.ckpt

# Vynucený CPU režim
cargo run --release --example aurora_train -- --cpu --dim 256 --epochs 1

# Overnight wrapper (nohup + pre-flight + JSONL metrics)
bash scripts/overnight.sh 25

# Live monitoring
tail -f logs/<run>.log                    # human-readable
tail -f logs/<run>.jsonl | jq             # structured

# Post-mortem diagnostics → PNG do logs/
python3 analysis/diagnostics.py logs/<run>.jsonl

# Standalone BPE training (jen tokenizer)
cargo run --release --example train_bpe -- data/tiny.txt 1024 data/tiny_1024.bpe
```

## Constraints

- **GPU:** wgpu 23 přes Vulkan/Metal/DX12 (no CUDA toolchain). Cílový hardware
  GTX 1650 / 4 GB VRAM. `MAX_DIM = 4096` v WGSL workgroup memory (16 KB shared).
- **Per-batch grad memory:** backward alokuje per-batch grad buffer pro readout
  (B × V × D floatů). Pro D=4096, V=4096, B=16 to je ~1 GB; pro D=2048, V=512
  ~64 MB.
- **Frozen codebook:** S52–S55 codebook je ±1 random init, jen readout se učí.
  Trainable codebook (gradient přes bind/bundle/permute) plánováno jako S56 ablace.

## Dokumentace

- [`docs/12_implementacni_plan_v3.md`](./docs/12_implementacni_plan_v3.md) — **aktivní plán** (S51 LNN-only + S52+ HDC pivot)
- [`docs/11_implementacni_plan_v2.md`](./docs/11_implementacni_plan_v2.md) — historie SNN+SSM era (Sprint 26–50)
- [`docs/10_implementacni_plan.md`](./docs/10_implementacni_plan.md) — hluboká historie (Sprint 0–25)
- [`docs/`](./docs/README.md) — průzkum SNN/LLM (8 tematických dokumentů, referenční)
