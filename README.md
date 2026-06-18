# Aurora

Decoder-only transformer language model, from-scratch v Rustu. Pre-norm
GPT-style stack (token + positional embedding, N transformer bloků s
causal multi-head attention a GELU FFN, final LayerNorm, tied LM head),
trénovaný přes AdamW s cosine LR schedule. CPU i GPU backend (wgpu /
WGSL, Vulkan/Metal/DX12).

> **Status:** Po refaktoringu Sprint 51–60. Projekt byl původně
> biologicky inspirovaná SNN (Sprint 0–50, BPB plato 2.365 na tiny_600k);
> pivot na klasický transformer otevřel cestu na realistické BPB < 1.5
> na enwik8. SNN linie je archivována v `docs/01_*` až `docs/11_*`. CPU
> + GPU stack hotový, GPU end-to-end přes 13 parity testů. Empirický
> overnight benchmark na enwik8_10m je další krok — viz
> [`docs/12_refactoring_plan_transformer.md`](./docs/12_refactoring_plan_transformer.md).

## Co je hotové

### CPU transformer
- **Causal multi-head attention** — wrapper okolo single-head SDPA,
  per-head split/concat, causal mask
- **GELU FFN** (tanh approximation, Karpathy/nanoGPT konvence)
- **Token + positional embedding** + tied LM head
- **Pre-norm decoder block** (LN → MHA → +res → LN → FFN → +res)
- **`DecoderTransformer`** stack — full forward/backward přes všechny vrstvy
- **AdamW** s decoupled weight decay (WD vypnut na LN γ/β a positional embed)
- **Cosine LR schedule** s linear warmup
- **Gradient clipping** (global L2 norm)
- NanoGPT residual init scaling (`1/√(2·n_layer)` na W_o a W_proj)

### GPU stack (wgpu / WGSL, NVIDIA Vulkan / Metal / DX12)
- **GEMM** (`A @ B`, `A @ Bᵀ`) — 16×16 workgroup tiles
- **Element-wise**: GELU forward/backward, LayerNorm forward/backward,
  vector_add, scale, zero
- **Causal MHA forward**: workgroup per query position, causal mask inline
- **Causal MHA backward (Sprint 60B)**: 3-pass tiled flash variant
  (`d_scores` scratch → `dq` per i → `dk`/`dv` per j); žádné f32 atomics, žádné race
- **Token + positional embed** lookup, scatter-add backward
- **AdamW kernel** s decoupled WD (per-tensor `apply_wd` flag)
- **Pure-GPU gradient accumulation (Sprint 60A)**:
  `OuterProductAccumKernel` (weight grads), `RowReduceAddKernel` (LN γ/β
  reduce), `GradNormSqReduceKernel` (clip), `EmbedScatterReduceKernel`
- **`DecoderTransformerGpu`** orchestruje full forward + backward na GPU
- CPU↔GPU parity testy < 5e-2 přes 13 nezávislých kontrol

### Inference
- **KV cache (Sprint 60C)** — `KvCache::prefill(model, tokens)` +
  `generate_step(model, &mut cache, new_token) -> logits`. O(N) per token
  místo O(N²). Parity test: bit-exact s full-forward greedy generation.
- Temperature sampling přes `text_io::sample_token`.

### Tokenizace + checkpoint
- **BPE tokenizer** (linked-list trainer + min-heap encoder, O(N log N))
- **Checkpoint v3** (binary `AURORA03`) — váhy embed + N blocks + final LN,
  fresh AdamW state at resume

### Tréninková infrastruktura
- **`aurora_train` example** — hlavní entry point. CLI: `--d-model`,
  `--n-layer`, `--n-head`, `--d-ff`, `--seq-len`, `--batch-size`, `--lr`,
  `--warmup-steps`, `--weight-decay`, `--grad-clip`, `--gpu` (opt-in),
  `--viz` (opt-in), `--governor` (opt-in, CPU + GPU).
- **JSONL structured metrics** (`--metrics path.jsonl`) — `run` / `step` /
  `epoch_done` / `checkpoint` events, per-event flush
- **`scripts/overnight.sh`** — nohup wrapper s pre-flight checks
- **Anti-regrese**: `transformer_primitives` (6 numerických gradient
  testů), `transformer_smoke` (3 convergence testů), `lm_quality_regression`
  (3 testů), `stability` (7 testů), `gpu_parity` (13 testů s `--features gpu`),
  `kv_cache` (3 testů), `governor` (12 testů — PHAGE).

### Live vizualizace (Bevy, opt-in)
- **`src/viz/`** + cargo feature `viz` — real-time Bevy 0.18 okno
  s pipeline-flow vizualizací: embed → N×{attn/ffn bar} → LM head,
  loss curve, decoded input/pred preview s OK/MISS indikátorem
- Trénink běží ve worker threadu, viz na main threadu, sdílí
  `Arc<Mutex<TrainingSnapshot>>` — žádný impact na throughput
- Window close = trénink pokračuje headless

### Runtime efficiency governor — PHAGE (opt-in, CPU + GPU)
- **`src/governor/`** + flag `--governor` — za běhu hlídá efektivitu
  tréninku (`η = −Δloss/Δwall`), lokalizuje původce stagnace a mění **živou
  matematiku** (LR / Muon-LR / momentum / weight-decay / grad-clip přes
  `MathState` vektor) bez rekompilace — Aurora ty skaláry staví znovu každý
  krok, takže governor je jen čte odjinud.
- **Bezpečnost přes verifier** (jádro): žádná změna se neadoptuje na
  korelaci. Každý kandidát projde **párovým probe-rollout** (snapshot →
  krátký rollout pod kontrolou vs. léčbou nad sdílenými vahami/daty) a
  **anytime-valid sekvenčním testem** (normal-mixture confidence sequence),
  který adoptuje jen efekt **nad noise floor**; jinak revert / below-floor.
  Většina kandidátů je pod floor a governor je správně odmítne.
- **Backend-agnostický core** (`TrialBackend` trait). CPU snapshotuje přes
  `Clone`; **GPU přes on-device buffer copy** (`snapshot_state`/
  `restore_state`, žádný host readback), takže reflex rollback i párové
  trialy běží na GPU plnou rychlostí.
- **Vrstvy**: vitals (multi-timescale EWMA) → diagnosis (posterior nad
  taxonomií poruch: lr-too-hot/cold, grad-noise, throughput…) → pharmacy
  (Gaussian-Thompson bandit nad fault-licencovanými rameny) → verifier →
  paměť (`<ckpt>.phage` TSV, cross-run učení). Divergence řeší reflex
  (snapshot rollback + LR cut, bez trialu).
- `governor` JSONL eventy (`--metrics`), 12 CPU + 2 GPU unit testů
  (`tests/governor.rs`, `tests/governor_gpu.rs`).

## Spuštění

```bash
cargo test --release                                      # 22 CPU testů (~2 s)
cargo test --release --features gpu                       # 35 testů (~5 s)
cargo test --release --features gpu --test gpu_parity     # 13 GPU parity testů

# Stažení korpusů
bash scripts/get_corpus.sh tiny            # 1.1 MB tiny-shakespeare
bash scripts/get_corpus.sh enwik8-10m      # 10 MB enwik8 subset
bash scripts/get_corpus.sh enwik8          # 100 MB enwik8 (Hutter Prize)

# Hlavní LM trénink, CPU
cargo run --release --example aurora_train -- \
  --corpus data/enwik8_10m.txt --vocab 4096 \
  --d-model 384 --n-layer 6 --n-head 6 --d-ff 1536 \
  --seq-len 256 --batch-size 16 --epochs 20 \
  --lr 6e-4 --warmup-steps 2000 --weight-decay 0.1

# GPU backend (přidej --gpu, vyžaduje --features gpu)
cargo run --release --features gpu --example aurora_train -- \
  --gpu --corpus data/enwik8_10m.txt --vocab 4096 \
  --d-model 384 --n-layer 6 --n-head 6 --d-ff 1536 \
  --seq-len 256 --batch-size 16 --epochs 20 \
  --lr 6e-4 --warmup-steps 2000

# Runtime efficiency governor — PHAGE (hlídá efektivitu + hot-swap živé
# matematiky). Eventy v JSONL, paměť .phage. Přidej --features gpu --gpu pro
# on-device trialy (snapshot/restore přímo na GPU, žádný host readback).
cargo run --release --example aurora_train -- \
  --corpus data/enwik8_10m.txt --vocab 4096 \
  --d-model 384 --n-layer 6 --n-head 6 --d-ff 1536 \
  --seq-len 256 --batch-size 16 --epochs 20 \
  --lr 6e-4 --warmup-steps 2000 --metrics logs/gov.jsonl --governor

# Live visualization (Bevy okno + CLI souběžně, --features viz,gpu)
cargo run --release --features viz,gpu --example aurora_train -- \
  --viz --gpu --corpus data/tiny_600k.txt --bpe data/tiny_600k_2048.bpe \
  --vocab 2048 --d-model 384 --n-layer 6 --n-head 6 --d-ff 1536 \
  --seq-len 256 --batch-size 16 --epochs 8 \
  --checkpoint data/tiny_600k_d384_l6_v2048.ckpt

# Overnight wrapper (nohup + pre-flight + JSONL metrics)
bash scripts/overnight.sh                  # default 25 epoch
bash scripts/overnight.sh 8                # custom

# Live monitoring
tail -f logs/<run>.log
tail -f logs/<run>.jsonl | jq

# Post-mortem diagnostics
python3 analysis/diagnostics.py logs/<run>.jsonl

# Standalone BPE training
cargo run --release --example train_bpe -- data/tiny.txt 1024 data/tiny_1024.bpe
```

## Dokumentace

- [`docs/12_refactoring_plan_transformer.md`](./docs/12_refactoring_plan_transformer.md) — **aktivní plán** (Sprint 51+)
- [`docs/11_implementacni_plan_v2.md`](./docs/11_implementacni_plan_v2.md) — SNN historie Sprint 26–50 (zamrzlé)
- [`docs/10_implementacni_plan.md`](./docs/10_implementacni_plan.md) — SNN historie Sprint 0–25 (zamrzlé)
- [`docs/`](./docs/README.md) — průzkum SNN/LLM (8 tematických dokumentů, kontext pro SNN linii)
- [`docs/02_transformer_architektura.md`](./docs/02_transformer_architektura.md) — teoretický základ pro současné jádro
