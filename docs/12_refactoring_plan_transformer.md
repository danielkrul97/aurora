# Refaktoring plán Aurora → Transformer

> Pokračování `11_implementacni_plan_v2.md`. v2 plán dosáhl ~60 KB a popisuje Sprint 26–50 (BPE → SSM cell, lokální optimum 2.365 BPB na tiny_600k). Tento dokument zachycuje strategický pivot: nahradit gated SNN / SSM stack klasickým decoder-only transformerem (nanoGPT-class) a otevřít cestu na BPB < 1.5 na enwik8.
>
> **Princip:** stejné jako v v2 — žádné odhady v týdnech, postup po exit kritériích. Refaktoring je breaking change v rámci research fáze; backward compat se neudržuje.

---

## Motivace

Sprint 26–50 vytrénovaly Auroru do lokálního optima **2.365 BPB** na tiny_600k (SSM cell, h=512, 6 epoch, lr=0.005). Sedm po sobě jdoucích negativních architektonických intervencí (SDSA, AdLIF, deeper stack, late/early/plain trophic apoptosis, continuous SSM) ukázalo že:

1. Aurora's gated/SSM cell + spike-binarizace mezi vrstvami má **self-consistent regularizační rovnováhu**, kterou každá inkrementální úprava (extra dampening, extra branch, depth) rozbije.
2. **Implicit regularizace přes 1-bit spike quantization** (Sprint 49) brání memorization na malém korpusu — ale zároveň limituje absolutní quality plateau.
3. Reálný cíl po SDSA (1.30–1.40 BPB, SpikeGPT reference 1.26) zůstává neprokázaný — žádná SNN-class architektura v repo neprošla pod 2.3.

**Pivot:** zachovat projektové infrastruktury (BPE, korpus pipeline, checkpoint, JSONL metrics, anti-regrese), nahradit jádro modelu klasickým transformerem. GPT-2-small reference na enwik8 dosahuje **~1.0 BPB**; nanoGPT-scale model (~10M params) by měl realisticky padnout pod **1.5 BPB** v overnight runu.

Aurora ztratí "biologically-inspired" charakter na úrovni jádra. Označení projektu v `Cargo.toml` (`description = "Biologically-inspired spiking neural network"`) se upraví. SNN/biologická linie zůstane zachycena v `docs/01_*.md` až `09_*.md` jako research log.

---

## Architektonický cíl

**Decoder-only GPT-style** autoregresivní LM (nanoGPT konvencí):

```
tokens (B × T)
  ↓ token_embed (V → D), pos_embed (T → D), sum
  ↓ block_0  (LN → MHA-causal → +residual → LN → FFN → +residual)
  ↓ block_1
  ⋮
  ↓ block_{N-1}
  ↓ final_LN
  ↓ lm_head (tied with token_embed, D → V)
logits (B × T × V) → softmax_cross_entropy
```

**Pre-norm**, tied input/output embeddings, causal mask, multi-head attention, GELU FFN. Žádný dropout v první iteraci (přidat až jako tuning lever).

**Cílová konfigurace** (nano-GPT class):

| param | hodnota | poznámka |
|---|---|---|
| `n_layer` | 6 | analog stávajícího 2-layer SNN (3× víc kompenzuje absenci recurrence) |
| `n_head` | 6 | d_head = 64 |
| `d_model` | 384 | ≈ 75 % stávajícího hidden=512 SSM |
| `d_ff` | 1536 | 4× d_model (klasické nanoGPT) |
| `seq_len` | 128–256 | nahrazuje TBPTT chunk; bez carry mezi sekvencemi |
| `vocab` | 4096 | beze změny (BPE shared) |
| `params` | ~10M | comfortably fit na GTX 1650 (4 GB VRAM) |

---

## Co se zachová (load-bearing infrastructure)

- **`src/bpe.rs`** — BPE tokenizer (O(N log N) trainer + encoder). Vocab-agnostic.
- **`src/text_io.rs`** — `load_corpus`, `bits_per_byte`, `sample_token`. Drop `one_hot*` (transformer používá integer token IDs, ne one-hot vektory).
- **`src/optimizer.rs`** — Adam stepper. Rozšíří se o `adamw_apply` (decoupled weight decay).
- **`src/linalg.rs`** — drop `active_indices`/sparse helpers, expand o dense GEMM helpers (CPU naïve f32 matmul, gelu).
- **`src/attention.rs`** — recyklovat `LayerNorm`, `LinearProj`, `scaled_dot_product_attention` (forward + backward, vše s ověřenými gradient testy). Drop `spike_gated_attention`.
- **`src/gpu/{mod,context,buffers}.rs`** — wgpu boilerplate, GpuBuffer abstrakce.
- **`SoftmaxCrossEntropyKernel`** z `src/gpu/kernels.rs` — vocab-agnostic (stride loops), znovupoužitelný.
- **`examples/train_bpe.rs`** — offline BPE training (tokenizer-only, žádné modely).
- **`scripts/get_corpus.sh`, `scripts/overnight.sh`** — pipeline beze změny; jen targets v overnight.sh se aktualizují.
- **`analysis/diagnostics.py`** — JSONL parser + plotting; spike-rate / α plot dropne, attention-entropy plot přijde místo nich.
- **JSONL schema** — `run` / `chunk` / `epoch_done` / `checkpoint` eventy zachovat; pole `rates` / `alphas` / `v_peaks` se vyřadí, přibudou `lr` / `grad_norm` / `attn_entropy_mean`.

---

## Co se smaže (clean cut)

Per uživatelské volbě "smazat všechny SNN moduly". Žádný `legacy/` adresář, žádný feature flag, žádný backward compat shim.

**Plné smazání zdroje:**

```
src/apoptosis.rs        src/snn_dense.rs        src/lif.rs
src/dendrite.rs         src/snn_gated.rs        src/lnn.rs
src/eligibility.rs      src/snn_gated_multi.rs  src/mnist.rs
src/encoding.rs         src/snn_loss.rs         src/network.rs
src/engine.rs           src/snn_recurrent.rs    src/neuromodulator.rs
src/homeostasis.rs      src/ssm.rs              src/neuron.rs
src/readout.rs          src/sdsa.rs             src/scheduler.rs
src/stdp.rs             src/surrogate.rs        src/synapse.rs
src/topology.rs         src/types.rs            src/hdc/  (jen-rozjeté)
```

**GPU stack — smazat:**

```
src/gpu/forward_mega.rs        (SNN-only megakernel)
src/gpu/forward_mega_2layer.rs (2-layer SNN megakernel)
src/gpu/snn_gated_gpu.rs       (single-layer GPU SNN)
src/gpu/snn_gated_gpu_multi.rs (multi-layer GPU SNN)
src/gpu/kernels.rs             (ALL SNN kernels except SoftmaxCrossEntropyKernel — extract first)
src/gpu/hdc_gpu.rs             (untracked, prerogue)
src/gpu/kernels_hdc.rs         (untracked)
src/gpu/kernels_common.rs      (untracked)
```

**Examples — smazat:**

```
examples/aurora_lnn_train.rs   (LNN-CfC; bude přepsán na transformer aurora_train)
examples/char_lm_gated_cz.rs   (legacy CPU SNN char-LM)
examples/mnist_eval.rs
examples/mnist_surrogate.rs
examples/mnist_train.rs
examples/seq_mnist.rs
```

**Tests — smazat:**

```
tests/apoptosis_smoke.rs       tests/attention_smoke.rs (jen spike_gated_*)
tests/bpe_smoke.rs (keep)      tests/char_lm_smoke.rs
tests/classify_synthetic.rs    tests/dendritic_xor.rs
tests/eligibility_trace.rs     tests/event_engine_plastic.rs
tests/gated_batch_parity.rs    tests/gated_lm_smoke.rs
tests/gpu_parity.rs (rewrite)  tests/homeostasis_balance.rs
tests/lm_quality_regression.rs (rewrite)
tests/lnn_smoke.rs             tests/multi_layer_gated_smoke.rs
tests/parity.rs                tests/per_output_credit.rs
tests/recurrent_remember.rs    tests/reward_modulated_xor.rs
tests/reward_potentiation.rs   tests/sdsa_integration.rs
tests/sdsa_smoke.rs            tests/single_neuron_lif.rs
tests/sparse_lm_parity.rs      tests/ssm_smoke.rs
tests/stateful_bptt.rs         tests/stdp_pairing.rs
tests/surrogate_classify.rs    tests/hdc_gpu_parity.rs
tests/hdc_smoke.rs
```

**Benches — smazat:**

```
benches/sparsity_scaling.rs    (SNN sparse activity)
benches/lm_sparse_speedup.rs   (SNN sparse CPU vs dense)
benches/lm_gpu_speedup.rs      (SNN GPU; rewrite as transformer perf bench)
benches/multi_layer_perf.rs    (SNN multi-layer)
```

**Cargo.toml změny:**

- `description` → `"Decoder-only transformer LM, from-scratch Rust"`.
- Drop benches (přepsat až ve Sprintu 59+).
- Drop `[[example]] name = "aurora_lnn_train"`.
- Default features beze změny (`gpu` zůstává opt-in).

**Cílová velikost po demoliční fázi:**  
~17 400 řádků → ~3000 řádků (zachová se `bpe`, `text_io`, `optimizer`, `linalg`, `checkpoint`, `attention` core primitivy + GPU context/buffers).

---

## Sprint 51 — Demolice SNN stacku

**Cíl:** vymazat všechen SNN kód, zachovat infrastrukturu. Po sprintu repo buildí ale neimplementuje LM — to je záměrné, čistá tabule pro Sprint 52.

**Deliverables:**

1. Smazat 27 source modulů (viz seznam výše).
2. Smazat 22 testů + 6 examples + 4 benches.
3. Smazat 3 SNN-specifické GPU moduly + `kernels.rs` (z něj nejprve extrahovat `SoftmaxCrossEntropyKernel` + jeho WGSL do `src/gpu/softmax_ce.rs`).
4. `src/lib.rs` redukovat na 7 modulů: `bpe`, `text_io`, `optimizer`, `linalg`, `checkpoint` (stub), `attention` (subset), `gpu` (subset).
5. `src/attention.rs` — odstranit `spike_gated_attention`, zachovat `LayerNorm`, `LinearProj`, `scaled_dot_product_attention` (forward + backward) + jejich gradient buffery.
6. `src/text_io.rs` — odstranit `one_hot_sequence`, `one_hot`, `sample_byte` (zachovat `sample_token`, `load_corpus`, `bits_per_byte`).
7. `src/linalg.rs` — odstranit `active_indices` (SNN-only). Ponechat `matvec`, `matvec_transpose`.
8. `src/checkpoint.rs` — vymazat AURORA01/02 implementaci, ponechat low-level helpery (`read_u32`, `write_f32_slice`, …). Modul je dočasně bez vlastního `save`/`load` (přijdou ve Sprintu 54).
9. `examples/aurora_train.rs` — smazat (ve Sprintu 54 se napíše nový).
10. `Cargo.toml` — viz výše.
11. `README.md` — krátký disclaimer "v rebuild na transformer; viz `docs/12_*.md`".
12. `CLAUDE.md` — žádné změny v této fázi (Sprint 60+ udělá clean-up po stabilizaci).

**Exit kritéria:**

- `cargo build` zelený (bez featu i s `--features gpu`).
- `cargo test --release` zelený s **0 testů** (po smazání všech SNN testů a před přidáním transformer testů to je očekávané).
- Zachované moduly nemají dead code warnings.
- Žádné `pub use` z `src/lib.rs` neukazuje na smazaný symbol.

**Záměrně neřešeno:**
- Žádný transformer kód ještě. Sprint 51 = pure cleanup.
- Žádný GPU kód kromě extrakce `SoftmaxCrossEntropyKernel`. Nový GPU stack v Sprintech 56–58.
- `docs/09_aurora_neuron.md` zůstává jako historický záznam (neutrální vůči refaktoringu).

---

## Sprint 52 — CPU transformer primitivy

**Cíl:** doplnit CPU primitivy potřebné pro decoder blok. Forward + backward s numerickou parita kontrolou (central-difference < 5e-2, stejný pattern jako `tests/attention_smoke.rs` v Sprintu 27).

**Deliverables:**

1. **Causal multi-head attention** v `src/transformer/attention.rs`:
   - `causal_mha_forward(x, w_q, w_k, w_v, w_o, n_head, d_head) -> (y, cache)` — interně split heads, per-head SDPA s causal maskou, concat, output projection.
   - Causal maska se neaplikuje materializací T×T tenzoru; místo toho se ve forward smyčce při i > j nastaví `s = NEG_INF` (stejný idiom jako `spike_gated_attention` v Sprintu 27).
   - Backward: gradient se propaguje přes `linear_backward` (output proj) → per-head SDPA backward → split heads → backward přes Q/K/V projekce.
2. **GELU + backward** v `src/linalg.rs` (extension):
   - `gelu(x)` = `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))` (approximation jako nanoGPT).
   - `gelu_backward(x, dy)` analytický.
3. **FFN** v `src/transformer/ffn.rs`:
   - `ffn_forward(x, w_fc, w_proj) -> (y, cache)` — `gelu(x · w_fc) · w_proj`.
   - Backward přes `linear_backward` na obou matvec.
4. **Embedding** v `src/transformer/embedding.rs`:
   - Token embedding (V × D matrix) + positional embedding (T_max × D matrix, learned).
   - Forward: integer lookup do obou tabulek + element-wise sum.
   - Backward: scatter-add gradientu do correspondingních řádků.
5. **Tied LM head**: helper `tied_lm_head_forward(h, w_tok_embed)` — výpočet logits = `h · w_tok_embedᵀ`. Backward akumuluje gradient do **stejné** váhové matice jako token embedding (test ověří že obě cesty přispívají).
6. **Soubor `src/transformer/mod.rs`** — re-export primitiv.
7. **Test `tests/transformer_primitives.rs`** — 5 numerických gradient tests (central diff < 5e-2):
   - `mha_causal_backward_matches_numerical`
   - `gelu_backward_matches_numerical`
   - `ffn_backward_matches_numerical`
   - `token_pos_embed_backward_scatters_correctly`
   - `tied_lm_head_gradient_accumulates_through_both_paths`

**Exit kritéria:**

- 5/5 nových testů zelených.
- Test čas < 5 s.
- Žádný symbol z `transformer/*` neimportuje SNN (snn_*, lif, surrogate, …) — sanity check že refaktoring je čistý.

**Záměrně neřešeno:**
- Stacked blok / full model — Sprint 53.
- Optimizer integrace — Sprint 53.
- GPU — Sprint 56+.

---

## Sprint 53 — CPU decoder blok + stack + Adam

**Cíl:** end-to-end CPU training step. Smoke convergence na hračkovém problému.

**Deliverables:**

1. **`TransformerBlock`** (`src/transformer/block.rs`):
   - Pre-norm: `y = x + MHA(LN(x))`, pak `y' = y + FFN(LN(y))`.
   - Forward vrací cache pro backward (LN cache × 2, MHA cache, FFN cache).
   - Backward s residual passthrough.
2. **`DecoderTransformer`** (`src/transformer/model.rs`):
   - Pole `blocks: Vec<TransformerBlock>`, `embed: TokenPosEmbed`, `final_ln: LayerNorm`.
   - `forward(tokens: &[u32; B*T]) -> (logits: Vec<f32>, cache: ForwardCache)`.
   - `backward(cache, targets) -> grads: GradBundle, loss: f32`.
   - `softmax_cross_entropy` s built-in gradient (existing `src/snn_loss.rs::cross_entropy_with_grad` — extrahovat do `src/transformer/loss.rs`, drop `snn_loss.rs` název).
3. **`GradBundle`** — bundle všech gradientů (per-layer × všechny matice + final LN + embed). Externalizovaný (jako stávající `GatedGrads`) pro budoucí parallel batch trainery.
4. **AdamW** v `src/optimizer.rs`:
   - `adamw_apply(params, grads, group, config, bc1, bc2, weight_decay)` — decoupled weight decay (`params -= lr * weight_decay * params` mimo gradient path).
   - `AdamWConfig { lr, beta1, beta2, eps, weight_decay }`.
5. **LR schedule** v `src/transformer/schedule.rs`:
   - `cosine_warmup(step, warmup_steps, max_steps, base_lr, min_lr) -> f32`.
6. **`DecoderTransformer::step_adam(&mut self, grads, &config, step)`** — Adam aplikace, gradient clipping (norm-based, existing pattern).
7. **Test `tests/transformer_smoke.rs`**:
   - `cpu_transformer_converges_on_copy_task` — 50 tokenů, model n_layer=2, d_model=64, 200 steps, finalní loss < 0.1 (random baseline ~ln(vocab)).
   - `cpu_transformer_step_does_not_nan` — 1 step na realistic config, žádný NaN v parametrech.

**Exit kritéria:**

- Oba testy zelené.
- Test čas < 30 s (delší toleruje copy task).
- Žádný regress v `transformer_primitives.rs` z Sprintu 52.

**Záměrně neřešeno:**
- Batch / TBPTT — Sprint 54 (full training loop).
- Generation / sampling — Sprint 54.
- GPU — Sprint 56+.

---

## Sprint 54 — Training loop + nová `aurora_train` + checkpoint v3

**Cíl:** plný CPU training pipeline na realistickém korpusu. Náhrada za starý `aurora_train`.

**Deliverables:**

1. **Checkpoint v3** v `src/checkpoint.rs`:
   - Magic `AURORA03` (clean break, žádný v1/v2 backward compat — branch je transformer).
   - Header: `vocab`, `d_model`, `n_layer`, `n_head`, `seq_len_max`, `epoch`, `step`.
   - Payload: per-block 6 vah (W_qkv combined nebo Q/K/V separátně — rozhodnutí v implementaci), W_o, W_fc, W_proj, 2× LN (γ, β). Plus embed (tok × pos), final LN, žádný separátní LM head (tied).
   - Žádný Adam state v ckpt; resume = fresh moments (stejně jako v2).
   - `save(path, &model, epoch, step) -> io::Result<()>` a `load(path) -> io::Result<Checkpoint>`.
2. **`examples/aurora_train.rs`** přepsat:
   - CLI: `--corpus`, `--vocab`, `--d-model`, `--n-layer`, `--n-head`, `--d-ff`, `--seq-len`, `--epochs`, `--batch-size`, `--lr`, `--warmup-steps`, `--weight-decay`, `--grad-clip`, `--checkpoint`, `--bpe`, `--metrics`, `--seed`, `--cpu` (default = GPU, fallback po Sprintu 58).
   - Pro Sprint 54: jen CPU cesta. `--cpu` defaultne na true; po Sprintu 58 se default přehodí.
   - Batch sample: shuffled non-overlapping windows of `seq_len + 1` tokenů z train splitu. Target = posunutý input o 1.
   - Per-step: forward, backward, AdamW, LR schedule update. Žádný TBPTT carry (transformer je stateless v rámci sekvence).
   - Per-epoch: full validation pass, sample generation (autoregressive, temperature 0.8, 200 tokenů).
   - JSONL metrics: `run`, `step` (místo `chunk` — krok je natural unit u transformeru), `epoch_done`, `checkpoint` eventy. `step` event obsahuje `step`, `train_loss_nats`, `lr`, `grad_norm`, `elapsed_s`.
3. **`tests/lm_quality_regression.rs`** přepsat:
   - `cpu_transformer_byte_lm_converges_under_one_epoch` — byte vocab=256, d_model=64, n_layer=2, korpus 50 KB z tiny_200k. Threshold val BPB < 5.0 (random = 8.0).
   - `cpu_transformer_bpe_lm_converges_under_one_epoch` — vocab=384, d_model=64, n_layer=2. Threshold val BPB < 5.0.
   - `checkpoint_v3_round_trip_preserves_loss` — save, load, ověř identický forward output.
4. **Anti-regrese:** udržet `cargo test --release` pod 60 s.
5. **`README.md`** update: nové CLI, nový hyperparam příklad, nahradit "Spiking Neural Network" za "Decoder-only Transformer", drop status sekci s SNN BPB čísly.

**Exit kritéria:**

- 3/3 regression testy zelené.
- CPU smoke run `cargo run --release --example aurora_train -- --corpus data/tiny_200k.txt --vocab 1024 --d-model 128 --n-layer 4 --n-head 4 --seq-len 64 --epochs 2 --batch-size 16` dokončí pod 5 minut.
- JSONL log validní (Python `json.loads` na každém řádku projde).
- `analysis/diagnostics.py` umí parsovat nové schema (i kdyby zobrazil placeholder — plný rework přijde ve Sprintu 59).

**Záměrně neřešeno:**
- GPU — Sprint 56+ (až po validaci CPU cesty).
- Mixed precision / flash attention / KV cache — Sprint 60+.

---

## Sprint 55 — Stabilizace + extended CPU validace

**Cíl:** vychytat numerické / convergence bugy, které unit smoke testy nezachytí. Připravit půdu pro GPU port.

**Deliverables:**

1. **Hyperparam sweep** na tiny_200k (CPU):
   - LR ∈ {3e-4, 6e-4, 1e-3, 3e-3} při d_model=128, n_layer=4, batch=16, seq_len=64.
   - n_layer ∈ {2, 4, 6} při d_model=128, lr=6e-4.
   - n_head ∈ {2, 4, 8} při d_model=128, n_layer=4.
   - Cíl: identifikovat best-of na malém modelu jako baseline. Logy v `logs/sprint55/`.
2. **Gradient health check** přidat do training loopu — pokud `grad_norm > 100` (před clippingem) déle než 100 steps po warmupu, varování do logu.
3. **Sampling diversity check** — generation s temperature ∈ {0.5, 0.8, 1.0, 1.5}, sample do JSONL. Validace že sampling není degenerátní (top-1 entropy > 0.5 z log(vocab)).
4. **Documentace v `docs/12_*.md` rozšířena** o pozorované delty + best-of config pro tiny_200k.

**Exit kritéria:**

- 1 paragraph poznámka v plánu, který hyperparam config byl best a očekávaný floor na tiny_200k.
- Žádná regrese existujících testů.

**Záměrně neřešeno:**
- Larger corpus — to počká na GPU (CPU by trvalo dny).
- Tied embed alternativy (non-tied, scaled by √D) — Sprint 60+.

---

## Sprint 56 — GPU kernels: matmul + GELU + LayerNorm

**Cíl:** první vlna WGSL kernelů. Bottom-up: jednotlivé building blocks ověřené parita testem proti CPU.

**Deliverables:**

1. **`src/gpu/matmul.rs`** — `GemmKernel`:
   - WGSL compute kernel `out = A @ B` (general B × M × K @ K × N → B × M × N). Workgroup-tiled (16×16 tiles).
   - Two variants: `gemm` a `gemm_transpose_b` (pro backward, `out = A @ Bᵀ`).
   - CPU helper `cpu_gemm(a, b, m, k, n)` pro parita test.
2. **`src/gpu/gelu.rs`** — `GeluKernel`, `GeluBackwardKernel`:
   - Element-wise WGSL, workgroup_size=256, vector load if possible.
3. **`src/gpu/layer_norm.rs`** — `LayerNormForwardKernel`, `LayerNormBackwardKernel`:
   - Per-row mean/var reduction (workgroup_size=D, sdílená paměť).
   - Backward: 3-pass nebo single-pass s atomic-add (single-pass preferováno; nuance jako v stávajícím `gated_cell_step_backward` pattern).
4. **`tests/gpu_parity.rs`** přepsat (zavřít předchozí inkarnaci):
   - `gemm_matches_cpu` (rel err < 5e-3).
   - `gemm_transpose_b_matches_cpu` (rel err < 5e-3).
   - `gelu_forward_backward_matches_cpu` (abs < 1e-4).
   - `layer_norm_forward_backward_matches_cpu` (rel err < 5e-2 — same threshold jako stávající attention testy).

**Exit kritéria:**

- 4/4 GPU parity testy zelené.
- Kernely buildí čistě na Vulkan / DX12 / Metal (stávající wgpu cross-platform).
- Žádný `dawn`-specific feature použít (Naga toolchain by se mohl měnit; držet conservative WGSL subset, žádný `f16`).

**Záměrně neřešeno:**
- Softmax / attention kernely — Sprint 57.
- AdamW kernel — Sprint 58 (zatím download grads, CPU AdamW, upload params; jen pro funkčnost).
- Mixed precision — Sprint 60+.

---

## Sprint 57 — GPU kernels: causal attention + embedding lookup

**Cíl:** zbytek transformer forward kernelů. Plný stack na GPU.

**Deliverables:**

1. **`src/gpu/attention.rs`** — `CausalMhaForwardKernel`, `CausalMhaBackwardKernel`:
   - Forward: workgroup-tiled QKᵀ scaled softmax × V. Causal maska aplikovaná inline (nikdy nematerializovat plnou T×T tabulku).
   - Backward: dQ, dK, dV přes 3 dispatch (standard softmax-Jacobian-vector product). Per-head paralelismus.
   - Pro Sprint 57 stačí "naive" verze — žádný flash-attention rework, ten až po profilingu (Sprint 60+).
2. **`src/gpu/embedding.rs`** — `TokenPosEmbedKernel`, `EmbedScatterAddKernel`:
   - Forward: gather z W_tok[token_id] + W_pos[t], element-wise add do output.
   - Backward: scatter-add do gradient bufferu. Atomic-add pro paralelní bezpečnost při kolizi (více pozic stejný token v batch).
3. **Tied LM head**: žádný separátní kernel — `gemm_transpose_b(hidden, w_tok_embed)`. Backward gradient přičten do `g_w_tok_embed` přes druhý `gemm` (`d_logits.T @ hidden`).
4. **`src/gpu/softmax_ce.rs`** — finalizovat extrakci z původního `kernels.rs` (Sprint 51 to udělal jen mechanicky; teď ověřit že fungue s nový workflow). Vocab-agnostic stride loops, žádná změna.
5. **`tests/gpu_parity.rs`** rozšířit:
   - `causal_mha_forward_matches_cpu` (rel err < 5e-2, sliding window n_head ∈ {2, 6}, d_head ∈ {16, 64}).
   - `causal_mha_backward_matches_cpu` (rel err < 5e-2, central-difference free — porovnává proti CPU analytic backward).
   - `token_pos_embed_forward_matches_cpu` (bit-exact pokud lookup je čistý gather).
   - `tied_lm_head_gradient_matches_cpu_through_both_paths` (rel err < 5e-2).

**Exit kritéria:**

- 4/4 + předchozí 4/4 = 8/8 GPU parity testů zelené.
- Causal mask na GPU se chová identicky s CPU (žádný off-by-one, žádný attention leak do budoucnosti — explicitní unit test na předposlední vs poslední token).

**Záměrně neřešeno:**
- Flash-attention memory layout — Sprint 60+ optimization.
- KV cache pro inference — Sprint 60+.

---

## Sprint 58 — GPU `DecoderTransformerGpu` + AdamW na GPU + návrat aurora_train

**Cíl:** poskládat všechny kernely do training pipeline. `aurora_train` má zase GPU backend.

**Deliverables:**

1. **`src/gpu/transformer.rs`** — `DecoderTransformerGpu`:
   - `from_cpu(&DecoderTransformer, GpuContext) -> Self` — upload všech vah.
   - `to_cpu(&self) -> DecoderTransformer` — download (pro checkpointing).
   - `forward(tokens) -> logits_gpu` (interní cache na GPU pro backward).
   - `backward(targets) -> loss: f32` (CPU-side loss reduction po download).
   - `step_adamw(&config, step)`.
   - `zero_grad()`, `clip_grads(norm)`.
2. **AdamW kernel** v `src/gpu/optimizer.rs`:
   - `AdamWKernel` — fused per-param AdamW step.
   - `AdamGradClipKernel` — global norm computation (workgroup-shared reduction přes všechny gradient buffery) + rescale.
3. **`examples/aurora_train.rs`** rozšířit:
   - `Backend` enum (analog stávajícího ve v2): `Gpu(DecoderTransformerGpu, name)`, `Cpu(DecoderTransformer)`.
   - `--cpu` flag default přepnutý na `false` (GPU default).
   - GPU init pre-flight (graceful fallback na CPU s warningem pokud GPU init selže).
4. **`tests/gpu_parity.rs`** finále:
   - `full_transformer_forward_matches_cpu` — full model 2-layer d_model=64, rel err < 5e-2 po 1 forward.
   - `full_transformer_training_step_matches_cpu` — 5 Adam kroků, loss trajektorie matches CPU (rel err < 5e-2 na každém kroku).

**Exit kritéria:**

- 10/10 GPU parity testů.
- `cargo run --release --features gpu --example aurora_train -- --corpus data/tiny_200k.txt --d-model 128 --n-layer 4 --n-head 4 --epochs 1 --batch-size 16` doběhne pod 60 s na GTX 1650.
- GPU val BPB ≤ CPU val BPB + 0.02 (parita; ne kvalita).

**Záměrně neřešeno:**
- Mixed precision — Sprint 60+.
- Pipeline parallelism — out of scope (1× GPU constraint).

---

## Sprint 59 — První overnight benchmark na enwik8_10m

**Cíl:** první "real" data point. Nano-GPT scale, enwik8_10m, overnight běh. Žádný kód, jen training run + analysis.

**Hyperparam config (initial guess, ladit podle Sprintu 55 nálezů):**

```
--corpus data/enwik8_10m.txt
--vocab 4096
--d-model 384
--n-layer 6
--n-head 6
--d-ff 1536
--seq-len 256
--batch-size 16
--epochs 20
--lr 6e-4
--warmup-steps 2000
--weight-decay 0.1
--grad-clip 1.0
```

Očekávaný model size: ~10M params. 16 × 256 = 4096 tokenů per batch, ~30K steps per epoch.

**Deliverables:**

1. **Pre-flight ověření** — 1 epoch dry run, kontrola NaN/Inf v gradientech, attention entropy ≠ degenerátní.
2. **Overnight běh** (`bash scripts/overnight.sh`).
3. **Post-mortem analysis** (`python3 analysis/diagnostics.py logs/<run>.jsonl`):
   - BPB convergence křivka.
   - Per-layer attention entropy (proxy za jak moc attention rozkládá vs collapses).
   - Grad norm trajektorie.
   - LR schedule.
4. **`docs/12_*.md` Sprint 59 záznam**: best val BPB, train-val gap, divergence diagnosis pokud nastane, srovnání s Aurora SNN 2.365 BPB (na jiném korpusu, ale ilustrativní).

**Exit kritéria:**

- Run doběhl bez crashe.
- Val BPB **< 2.0** na enwik8_10m (loose; ale jasně lepší než SNN floor 2.36 na tiny_600k).
- Žádný runaway (peak grad norm po warmupu < 50).
- Sample generation produkuje syntakticky věrohodný text (alespoň anglické slovní hranice respektovány na konci tréninku).

**Stretch target:** val BPB < 1.5 (porovnatelné s LSTM baselines na enwik8).

---

## Sprint 61 — GPU loss v `train_step` + GPU port `evaluate`

**Cíl:** vyhodit dva duplicitní host roundtripy z hot path. Per-step train už nestahuje T·V logits (CPU loss → upload d_logits → backward), a per-epoch validace už nečeká na `gpu.to_cpu()` + neběží na CPU.

**Co se změnilo:**

1. **`src/gpu/softmax_ce.rs`** — `SoftmaxCrossEntropyKernel` má nový push-constant `loss_scale: f32`. Shader násobí d_logits tímto skalárem v jediném shader stride loop. `dispatch_into` má nový argument; volání pro train používá `1.0 / t`, eval `1.0` (d_logits se zahodí).
2. **`src/gpu/transformer.rs`**:
   - `DecoderTransformerGpu` má nový field `softmax_ce: SoftmaxCrossEntropyKernel`.
   - `Activations` má dvě nové buffery sized `seq_len_max`: `loss_seq_buf` (per-token NLL) a `targets_buf` (u32 targets uploadovány per call).
   - Forward dispatchy (embed → bloky → final LN → tied LM head) jsou vytažené do privátního `fn encode_forward(&self, enc, t)` — train_step a eval_step volají identický kód.
   - `train_step` napojený na GPU softmax_ce: jeden command encoder pro fwd + softmax_ce + bwd, jeden submit, download jen T floatů na mean reporting. Odpadla CPU loss smyčka i `upload_prefix(&d_logits)`.
   - Nová metoda `pub fn eval_step(&self, tokens, targets) -> f32` — forward + softmax_ce, vrací sumu NLL přes T (caller agreguje napříč okny).
3. **`examples/aurora_train.rs`** — nová `evaluate_gpu(gpu, val_toks, seq_len, comp)` zrcadlí CPU `evaluate` (stride = seq_len). Call-site v `aurora_train` větví: pokud `gpu_backend` je `Some`, volá `evaluate_gpu`, jinak CPU `evaluate`. `gpu.to_cpu()` na začátku epochy zůstává — pořád ho potřebuje `checkpoint::save` a `generate(&ctx.model, …)`.
4. **`tests/gpu_parity.rs`** +3 testy:
   - `softmax_ce_grad_matches_cpu` — micro-gate, V=257 / T=5, rel err < 1e-4 na d_logits, < 1e-5 na per-token NLL.
   - `train_step_loss_matches_cpu_baseline` — GPU `train_step` loss vs CPU `forward+backward` loss na stejných váhách, |Δ| < 5e-3.
   - `evaluate_gpu_matches_cpu_evaluate` — d=32, n_layer=2, V=64, val_len=320, |Δ BPB| < 1e-3.

**Exit kritéria — splněno:**

- 16/16 testů v `tests/gpu_parity.rs` zelených (13 stávajících + 3 nové).
- `cargo check --features gpu --all-targets` clean; `cargo check` (bez GPU) clean.
- Eliminován per-step download `t * v` floatů + upload `t * v` floatů; nahrazeno downloadem `t` floatů. Pro nano config (T=128, V=4096): ~4 MB roundtrip per step uspořeno.

**Záměrně neřešeno (deferred):**

- **True B>1 vektorizace.** Všechny forward i backward kernely mají buffery sized `T·D` (B=1). Skutečné B>1 by vyžadovalo resize aktivačního arena na `B·T·D`, B-aware dispatch ve všech kernelech a přestavbu `fwd_ops` / `bwd_ops`. Out of scope Sprint 61 → otevřeno pro Sprint 62.
- **Reorder `to_cpu()` za eval.** Marginální wall-time výhra, větší churn — neuděláno.
- **GPU sample generation.** `generate()` v `aurora_train` pořád běží na CPU modelu po `to_cpu()`. KV cache pro inference je v `src/gpu/kv_cache.rs` připravená, ale není napojená.

---

## Sprint 63 — Sliding-window eval + Muon WD + RoPE

**Cíl:** odblokovat tři páky pro cestu k BPB < 1.5 identifikované v analýze
(scale-dominated režim): správné měření val BPB, decoupled weight decay na
Muon maticích a rotary position embedding místo learned pos table.

**Co se změnilo:**

1. **Sliding-window eval** (`examples/aurora_train.rs`): `evaluate` i
   `evaluate_gpu` přešly z non-overlapping oken (stride = seq_len) na
   překryvná (stride = seq_len/2) — skóruje se jen suffix okna za
   překryvem, takže každý token má ≥ seq_len/2 kontextu. Non-overlapping
   skórovalo začátky oken téměř bez kontextu a nafukovalo BPB o ~0.05–0.15.
   GPU strana: nová `eval_step_per_token` vrací per-position NLL
   (`eval_step` je nad ní tenký wrapper, parity test beze změny zelený).
   Eval cena ~2× (dvojnásobek oken) — pořád zanedbatelné vs trénink.
2. **Muon weight decay**: nový CLI flag `--muon-wd` (default 0.0 = beze
   změny chování). CPU stepper WD už uměl; na GPU byl `apply_wd` u Muon
   ops hardcoded `false` — teď `true` (s wd=0 je shader add no-op, flag
   plně řídí trainer). Hodnota jde do header printu i JSONL `run` eventu.
3. **RoPE** (`--rope`, default off — A/B-ovatelné):
   - `TransformerConfig.rope: bool`; `MhaWeights.rope`; `EmbedWeights.use_pos`.
   - CPU: `rope_rotate` v `src/attention.rs` (interleaved páry, base 10000,
     `inverse` = transponovaná rotace pro backward). MHA forward rotuje
     q/k po projekci (cache nese rotované hodnoty), backward aplikuje
     inverzní rotaci na d_q/d_k před projection backward. Learned w_pos
     se při rope nepřičítá a nedostává gradient (zůstává alokovaný —
     checkpoint layout beze změny).
   - GPU: nový `RopeKernel` (`src/gpu/rope.rs`, thread per pár, in-place).
     Forward pass po QKV gemmech, backward pass po `mha_bwd` na sdílených
     dq/dk. Embed kernely (`TokenPosEmbedKernel`, `EmbedScatterAddKernel`,
     `SingleTokenEmbedKernel`) mají `use_pos` push constant. KV-cached
     decoding (`generate_step_kv`, `forward_kv_only`) rotuje na absolutní
     pozici.
   - KV cache CPU (`block_step`) rotuje q_new/k_new na `t_new_pos`;
     historie nese rotované K, prefill je konzistentní.
   - **Checkpoint AURORA04**: nové pole `flags: u32` (bit 0 = rope) za
     `seq_len_max`. Žádný backward compat s AURORA03 (per konvence).
     Resume check v traineru porovnává i rope flag.

**Testy (+6):**

- `mha_rope_backward_matches_numerical` — central-difference check dx,
  dW_q, dW_k s rope on.
- `rope_relative_shift_invariance` — q·k závisí jen na relativním offsetu.
- `kv_cache_generate_step_matches_full_forward_rope` (CPU) a
  `gpu_kv_cache_parallel_kernel_handles_large_t_total_rope` (GPU).
- `full_transformer_rope_forward_and_grads_match_cpu` — loss + g_w_q/g_w_k/
  g_w_tok parity, g_w_pos == 0 na obou backendech.
- `checkpoint_round_trip_preserves_rope_flag`.

**Exit kritéria — splněno:** GPU sada 39/39, CPU sada 25+/25, end-to-end
GPU run s `--rope --muon-wd 0.05` konverguje (tiny.txt, loss 4.6 → 2.0
za ~1700 kroků). `cargo check --all-targets --features gpu` clean.

**Pozn. pro overnight run:** doporučený příkaz pro plný enwik8 je
`--gpu --rope --muon-wd 0.05..0.1` + plný korpus přes
`scripts/get_corpus.sh enwik8`. První čísla pod novým evalem nejsou
srovnatelná se starými (sliding-window měří ~0.05–0.15 níž — správně).

---

## Sprint 64 — Perf sweep (CPU hot paths + eval)

**Cíl:** projít celý projekt a odstranit měřitelné performance issues.
Inventura potvrdila, že GPU kernely (GEMM 64×64 tile + 4×4 register tile,
tiled MHA) i trénovací smyčka (1 submit/batch, gated norm readback) jsou
už v pořádku — zbývaly CPU hot paths a eval.

**Co se změnilo (vše bit-exact nebo v rámci f32 šumu):**

1. **CPU `evaluate` paralelizace** (`aurora_train.rs`): okna se skórují
   rayon-paralelně (forward je `&self`) a per-row skóre počítá logsumexp
   přímo — bez per-pozice softmax alokace. Eval byl jediná single-thread
   část CPU pipeline. Měřeno: nano val ~21 s → 5.2 s; d=128 run 25 s → 3.6 s.
2. **GPU eval batching**: nová `eval_step_batch` zrcadlí `train_step_batch`
   (1 encoder/submit/readback na `max_batch` oken místo per okno);
   `DecoderTransformerGpu` nově nese `pub max_batch`. Měřeno ~0.3 ms
   ušetřeného sync overheadu na okno — na malém val ~15 % evalu, na full
   enwik8 (~78k oken) odhad ~25 s/epochu.
3. **CPU linear/tied-head blocking** (`attention.rs`, `embedding.rs`):
   smyčky blokované po 8 pozicích — váhová matice (w_fc 2.3 MB, w_tok
   6 MB ≫ L2) se streamuje 1× per blok místo per pozici. Pořadí akumulace
   zachováno → bit-exact. Měřeno: nano CPU step 6.61 → 6.34 s (−4 %);
   na d=128 (váhy L2-resident) neutrální — dle očekávání.
4. **`rope_rotate`**: jeden `powf` na volání (multiplikativní rekurence
   freq) a sin/cos hoistnuté ze smyčky přes heady (bylo `t·n_head·half`
   powf+sincos, je `half` powf + `t·half` sincos).
5. **`model.backward`**: softmax+CE gradient počítaný in-place do
   `d_logits` (žádné 2 alokace per pozice), 1/T scale fúzovaný do
   normalizační passy.
6. **MHA fwd/bwd**: per-head scratch buffery hoistnuté před smyčku přes
   heady (4·n_head → 4 alokace na volání).
7. **Instrumentace**: epoch summary tiskne `(eval X.Xs)` a JSONL
   `epoch_done` event má nové pole `eval_s`.
8. Drobnosti: `gpu_perf_bench` má `required-features = ["gpu"]` (plain
   `cargo test` bez GPU featury se předtím nezkompiloval).

**Měření (GTX 1650 + host CPU, sekvenčně, baseline = HEAD worktree):**

| Metrika | Před | Po |
|---|---|---|
| CPU d=128 epocha (400 KB tiny) | 197.4 s | 171.9 s (eval 3.6 s) |
| CPU nano step (d=384 L=6 V=4096) | 6 612 ms | 6 336 ms |
| CPU nano epocha (150 KB enwik) | 484.2 s | 447.5 s (eval 5.2 s) |
| GPU eval (310 oken, d=128) | 0.7 s | 0.6 s |
| GPU nano train step (bench) | 28.4 ms | 27.9 ms (no regression) |

**Konvergence:** train/val bpb identické v rámci f32 šumu
(3.031/2.981 vs 3.031/2.980 na nano). Testy 61/61 GPU + 35/35 CPU zelené.

**Záměrně neřešeno:**

- `generate_step_kv` GPU alokuje ~15 bufferů per token — používá se jen
  v testech (sampling v traineru běží na CPU po `to_cpu()`), nestojí za
  arenu.
- BPE `encode` drží 3 vektory indexů (pro 100 MB korpus ~2 GB RAM peak);
  rychlostně je O(N log N) v pořádku. Případný u32-index refactor až
  kdyby RAM reálně chyběla.
- CPU GEMM SIMD/threading uvnitř jednoho okna — rayon přes batch už
  saturuje jádra; vnitřní paralelismus by jen přerozdělil tentýž čas.

---

## Sprint 65 — Tier-1 modernizační balík (speedrun kánon)

**Cíl:** zabudovat šest prověřených modernizací z modded-nanogpt
literatury jako opt-in balík pro pack-vs-baseline A/B na plném enwik8.
Všechno CPU + GPU, za CLI flagy (default chování beze změny):

1. **RMSNorm** (`--rmsnorm`) — LN bez mean-subtrakce a bez β. β zůstává
   alokovaná (checkpoint layout nezměněn), nedostává gradient. GPU: mode
   bit v push constants LN fwd/bwd kernelů.
2. **ReLU²** (`--relu2`) — squared-ReLU místo GELU ve FFN (Primer). GPU:
   mode bit v gelu kernelech.
3. **QK-norm** (`--qk-norm`) — parameter-free RMS norm per-head Q/K řádků
   po projekci, před RoPE. Nový kernel `qk_norm.rs` (fwd + in-place bwd);
   `MhaCache` nese `q_raw`/`k_raw` pro backward.
4. **Untied head** (`--untied-head`) — samostatná zero-init `w_head`
   matice (AdamW + WD, ne Muon). Zero-init + softcap ⇒ přesně uniformní
   logity v kroku 0 (loss = ln V, ověřeno).
5. **Softcap** (`--softcap 30`) — `cap·tanh(logits/cap)` (Gemma 2).
   Backward bez extra cache: tanh se rekonstruuje z capped logits.
   Nový kernel `softcap.rs`.
6. **WSD schedule** (`--lr-schedule wsd`, `--decay-frac 0.4`) — warmup →
   stable → lineární decay; run je prodloužitelný bez přeplánování.

Plus `--zero-ffn-proj` (zero-init FFN down-projekce; w_o už zero byla)
a **`--tier1`** = celý balík jedním flagem (softcap 30, wsd).

**Checkpoint AURORA05:** flags bity 1–4 (rmsnorm/relu2/qk_norm/untied),
nové f32 pole `softcap` v hlavičce, podmíněný tensor `w_head` za
`w_pos`. Žádný backward compat s AURORA04.

**Testy (+9):** numerické gradient checky (rmsnorm, relu2 FFN, qk-norm
MHA, softcap+untied full-model), zero-proj init, WSD shape, KV-cache
parity tier1 (CPU), checkpoint round-trip tier1, GPU full-model tier1
parity (loss + g_w_q/k/fc/proj/ln1 + g_w_head + per-token NLL + nulový
β grad). Celkem 70/70 zelených (GPU sada).

**Smoke (tiny_200k, GTX 1650):** `--gpu --tier1` 1 epocha → val 3.245,
resume z AURORA05 pokračuje (2. epocha → 3.085). CPU backend i
`--features gpu,viz` build ověřeny.

**Doporučený overnight A/B:** baseline `--gpu --rope --muon-wd 0.05`
vs pack `--gpu --tier1 --muon-wd 0.05`, plný enwik8, stejný seed a
wall-clock budget. Pack se přijímá/odmítá jako celek; ablace jen pokud
prohraje (noise floor ~0.035 BPB).

### Výsledek overnight A/B (2026-06-10/11)

Plný enwik8 (95 MB, 29.9M tokenů, BPE 4096 z 10MB prefixu), nano
d384/L6, seq 256, batch 16, **2 epochy/arm**, warmup 1000, seed
0xC0FFEE, sekvenčně na GTX 1650 (~4h12m/arm, wall-clock parita 0.3 %):

| arm | val e0 | val e1 (final) | train e1 |
|---|---|---|---|
| baseline (rope + muon-wd 0.05) | 1.591 | **1.375** | 1.478 |
| tier1 pack | 1.640 | **1.388** | 1.536 |

**Závěry:**

1. **Cíl BPB < 1.5 SPLNĚN** — baseline 1.375 po 2 epochách. Scale
   (plný korpus) byl podle predikce dominantní páka (~−0.5 vs 10MB runy).
   Rekord = 1.375.
2. **Pack na 2-epochovém budgetu nevyhrál:** Δ = +0.013 proti baseline,
   hluboko pod noise floorem ~0.035 ⇒ statistická remíza, žádný důkaz
   zisku. Per protokol se pack nepřijímá; rekordní konfigurace zůstává
   baseline.
3. **Trajektorie packu je strmější** (e0→e1: −0.252 vs −0.216; zero-init
   untied head + WSD stable fáze startují pomaleji, ale dohánějí).
   Hypotéza: na 4+ epochách se pořadí otočí. Než ablace jednotlivých
   členů (drahá, 6 nocí), správný další krok je **4-epochový re-run
   obou arms** — levnější a testuje budget-limited vysvětlení přímo.

### Dlouhokontextový experiment T=1024 (2026-06-11/12)

Stejný protokol jako rekord, jediná změna kontext: **B=4 × T=1024 =
4096 tok/krok** (= tokenový batch rekordu ⇒ identických 6569
kroků/epochu, stejný schedule/warmup/seed/binárka). 2 epochy, 7.9 h
(1.79× pomalejší na token), cached-probs attention (flash je na T≤1024
pomalejší, viz Sprint 67). Trainer val: e0 1.561, e1 **1.337**.

**Dekompozice evalem `eval_ckpt`** (harness reprodukuje rekord 1.3752):

| eval okna | ctx1024 ckpt | baseline ckpt |
|---|---|---|
| 256 | 1.3733 | 1.3752 |
| 512 | 1.3490 | — |
| 1024 | **1.3374** | — (arena 256) |

**Závěry:**

1. **Nový rekord 1.337** — ale celý zisk je **deployment-time kontext**,
   ne lepší natrénování: při identickém 256-okenním evalu jsou oba
   modely statisticky shodné (Δ 0.002). Trénink na dlouhém kontextu
   per-token kvalitu nezlepšil; **umožnil** ale dlouhý kontext při
   inferenci, a ten je hodný −0.036 BPB (a škáluje: 256→512→1024 dává
   −0.024 → −0.012, ještě nesaturováno).
2. **Praktický důsledek: tier3 `--seq-curriculum` je přesně správný
   nástroj** — trénovat převážně na T=256 (1.79× levnější token) a
   prodloužit na 1024 jen ke konci; deployment zisk zůstane, trénovací
   cena klesne téměř na úroveň T=256. Kandidát na příští overnight.
3. Wall-clock srovnání zůstává otevřené: 2 ep × T=1024 (7.9 h) vs
   4-epochový baseline (≈8.4 h) nikdo neměřil — spojit s plánovaným
   4-epochovým re-runem.

---

## Sprint 60+ — otevřená pole

| Větev | Co odemyká | Effort |
|---|---|---|
| **Flash-attention WGSL** | Větší seq_len bez OOM, ~2× throughput | large |
| **KV cache + inference path** | Faster sampling, opens chatbot demo | medium |
| **RMSNorm** (RoPE hotové ve Sprint 63) | Modern norm, Llama-style | small |
| **Mixed precision (fp16 storage, fp32 compute)** | 2× VRAM headroom, blocked dokud Naga nepodpoří `f16` | large (waiting) |
| **enwik8 full (100 MB) overnight** | Hutter Prize comparable BPB | medium (waiting) |
| **GPT-2-small reprodukce** | n_layer=12, d_model=768 — VRAM tight, mixed precision dependency | medium-large |
| **Hyperparam search** (proper, ne hand-tuning) | Optuna / random search wrapper kolem `aurora_train` | medium |
| **SwiGLU FFN** | +2-3 % na quality dle literatury | small |
| **True B>1 batched fwd/bwd na GPU** | Vektorizace všech aktivačních bufferů na `B·T·D` + B-aware dispatch | large |

---

## Co tento plán **záměrně neřeší**

- **Backward compat se SNN modelem.** AURORA02 checkpointy se nečtou. v2 plán je historie.
- **Biologická inspirace v jádře.** Aurora se stává čistě transformerovým projektem. Historický záznam (SNN linie, sprint 0–50) zůstává v `docs/01_*` až `docs/11_*` plus `docs/09_aurora_neuron.md`.
- **Distribuovaný trénink, multi-GPU.** Single GTX 1650 stále constraint.
- **Inference / chat UI.** Sampling v `aurora_train` stačí pro post-mortem inspekci.
- **Quantization, distillation, MoE.** Out of scope.
- **Tokenizer redesign.** BPE z Sprintu 26 je dostačující.

---

## Open questions / risk areas

1. **Tied embedding gradient deadlock na GPU.** Atomická scatter-add do gradient bufferu při paralelním backward přes embed + tied lm_head může mít contention. Sprint 57 to musí ověřit (možná split = separátní gradient buffery + final reduce).
2. **Cosine LR schedule + AdamW interakce.** Transformer-default config (lr=6e-4, warmup 2000, weight_decay 0.1) na nano-scale na enwik8_10m **není testován v literatuře**. Sprint 55 hyperparam sweep musí dát baseline.
3. **Workgroup memory budget na causal MHA.** Stávající `forward_mega_2layer` byl konstrávaný kolem hidden ≤ 1024. MHA forward s n_head=6, d_head=64, T=256 má jiný memory pattern — workgroup-tiled QKᵀ není trivial port. Sprint 57 risk.
4. **Embedding scatter-add atomic kontence na GTX 1650** (Turing TU116, ne tensor cores). Bude se měřit, ale není apriorní red flag.
5. **JSONL schema breaking change.** `analysis/diagnostics.py` musí umět detekovat verzi (SNN-era `rates`/`alphas` vs transformer-era `lr`/`grad_norm`). Buď dvojí parser nebo prefer-new + fallback. Sprint 54 deliverable.

---

# Sprint 67 — 10-hypotéz balík: canon + embed-lr vyhrály

Cíl: 10 naprosto unikátních hypotéz na zvednutí kvality, ověřit a aplikovat.
Všech 10 implementováno CPU-first + **grad-check** (`tests/transformer_primitives.rs`,
21 testů, central-difference tol 5e-2 — projekt nemá autograd, takže každý
nový backward je ručně ověřený). Infra navíc: `--eval-windows N` (cap val-eval,
eval byl ~79 s/epochu = skutečný per-run bottleneck).

## Flagy (všechny default-off, kompozují)
`--z-loss <w>` (PaLM logit z-loss), `--embed-lr-mult <m>` (LR× pro embed/head
Adam grupy), `--head-bias` (unigram-init LM-head bias), `--parallel-block`
(GPT-J topologie), `--layerscale` (per-kanál CaiT gaty), `--ema <d>` (weight-EMA
eval, CPU), `--swiglu` (gated FFN + `w_gate` Muon; reduce `--d-ff` pro iso-param),
`--canon-kernel <k>` (causal depthwise conv v residual streamu), `--diff-attn`
(DIFF Transformer přes SDPA-reuse: `SDPA(Q1,K1,V)−λ·SDPA(Q2,K2,V)`), `--sink <K>`
(učené register-tokeny). Composite: **`--tier4` = canon-kernel 4 + embed-lr-mult 3.**

## Metodika: 3-stupňový funnel
Screening na enwik8_10m (proxy d256/L4). **Klíčové ponaučení:** 1-epochové proxy
je *zaujaté ve prospěch early-convergence akcelerátorů* — convergence test (3 ep)
**obrátil pořadí**.

| lever | Δ 1ep (d256/L4) | Δ 3ep | verdikt |
|-------|-----------------|-------|---------|
| canon | −0.059 | **−0.236** | roste s tréninkem → reálný |
| embedlr | −0.181 | **−0.103** | smrsklo, ale drží → reálný |
| parallel | −0.136 | −0.021 | miráž (early-conv) |
| layerscale | −0.106 | −0.019 | miráž (early-conv) |
| headbias | −0.023 | −0.011 | pod prahem |
| ema | +0.574 | −0.002 | **artefakt** (decay 0.999 ≫ 708 kroků) → neutrál @0.99 |
| swiglu | +0.162 | +0.003 | **artefakt** (iso-param ⅔ zúžení) → neutrál @full d_ff |
| sink / zloss / diffattn | ~0 | — | neutrál na téhle škále |

## Nano confirm (gold škála, d384/L6, 4 ep, enwik8_10m, seed 42)
| run | best val_bpb | Δ |
|-----|--------------|---|
| baseline | 1.7971 (✅ reprodukuje anchor ~1.82) | — |
| canon | 1.6978 | **−0.099** |
| embedlr | 1.7214 | **−0.076** |
| **combo (--tier4)** | **1.6612** | **−0.136** |

Stacking sub-aditivní (~78 %): canon a embed-lr jsou skoro ortogonální
(architektura × optimizer), overlap ~22 %. Druhý seed (1337) běží.

## Aplikováno trvale
- `--tier4` composite (canon + embed-lr).
- **Checkpoint AURORA06**: `canon_kernel: u32` v hlavičce + per-block `canon_w`
  tenzor. AURORA05 se **stále čte** (dual-magic, canon off) — staré checkpointy
  se neztratí. embed-lr-mult je training-time (neukládá se; CLI vyhrává i na resume).
- Regrese: `checkpoint_v6_round_trip_preserves_canon` (52 testů zelených).

## Další
- Druhý/třetí seed confirm combo (běží seed 1337).
- **Rekordní recept**: canon+embed-lr na full enwik8 + T=1024 + tier1 (současný
  rekord 1.337) — projekce nový rekord ~1.27–1.31, netestováno.
- Scale-hungry levery (diff-attn, z-loss s vyšší vahou) možná přidají při T=1024.

## GPU port canonu — HOTOVO
`src/gpu/canon.rs` — 3 WGSL kernely (fwd; bwd-dx gather; bwd-dw per-tap reduce,
oba **bez f32 atomik / bez race**, jedna sekvence `[T·d]` na běh). Zapojeno do
`transformer.rs`: per-block `canon_w` buffer + AdamW momenty, `canon_out` v
LayerCache (ln1 + post-attn residual ho čtou), `canon_din` backward scratch,
dispatch ve `encode_forward` (před ln1) a `block_backward` (po vadd_dx). Parity
test `full_transformer_canon_forward_and_grads_match_cpu` (loss + g_canon_w +
g_w_q + g_w_tok < 5e-2), 80 testů zelených. GPU smoke trénuje stabilně.
**Pozn.:** embed-lr-mult je optimizer-only a na GPU step se zatím nečte — pro
plný `--tier4` na GPU chybí 1 malá úprava (scale LR pro embed grad-grupy).

## Sprint 68 — PHAGE: runtime governor efektivity (`--governor`)

Samostatný subsystém `src/governor/`, který za běhu hlídá, jestli trénink
jde efektivně, lokalizuje původce zpomalení/stagnace a mění živou matematiku
bez rekompilace. Default off, zapíná `--governor` (CPU i GPU, viz níže).

**Klíčový vhled, na kterém to stojí.** Aktuace v Auroře je *zadarmo*:
`AdamWConfig`/`MuonConfig` se staví znovu každý krok, operátorové booly
(`rms`, `relu2`, `rope`, `qk_norm`) a `softcap`/`z_loss` se čtou každý forward
při neměnných tvarech tenzorů. Celá matematika je tedy funkce malého
`MathState` (MSV) vektoru — žádný dlopen/JIT. Skutečný protivník není
aktuace, ale **kauzální atribuce pod noise floor** (screening noise ~0.035
BPB ≫ typický efekt). Proto těžiště algoritmu je verifier, ne knoby.

**Architektura (7 vrstev, vše čisté + testovatelné):**
- `state.rs` MSV — `lr_scale`/`muon_lr_scale`/`momentum`/`weight_decay`/
  `grad_clip_scale`/`softcap`, `apply_optim` škáluje per-step configy in-place
  (všechny backendy ho konzumují → LR/momentum/wd governance je automatická).
- `vitals.rs` — multi-timescale EWMA, `Fingerprint`, `η = −Δloss/Δwall`.
- `reflex.rs` — divergence guard (NaN/exploze) → `clone_from` rollback + LR cut,
  jediná vrstva co jedná bez trialu (jistota).
- `diagnosis.rs` — diferenciální diagnostika: posterior nad taxonomií poruch.
- `pharmacy.rs` — fault→licencovaná ramena + Gaussian-Thompson bandit (TSV
  sidecar `*.phage`, cross-run učení).
- `verifier.rs` — **jádro novosti**: anytime-valid normal-mixture confidence
  sequence nad *párovými* probe rozdíly. Imunní vůči peekingu → governor smí
  zastavit hned, jak interval překročí floor, bez inflace false-adopt.
- `trial.rs` / `backend.rs` — párový rollout přes `TrialBackend` trait
  (snapshot/restore/paired_diff); core nesahá na model. Sdílené `W`+data+probe
  ⇒ nuisance „kde na křivce jsme" se v rozdílu vyruší (model-level A/A test
  dává **přesně 0**). `CpuTrialBackend` přes `Clone`, `GpuTrialBackend` přes
  on-device buffer copy (viz CPU+GPU níže).

**Verdikt = Adopt / Revert / BelowFloor.** Většina kandidátů je pod floor a
governor je správně *odmítne* (disciplína, co chyběla ručnímu screeningu).
Bandit se učí jen na potvrzených ziscích, ne na korelaci.

**Verifikace.** `tests/governor.rs` 12 testů zelených — load-bearing je
`cs_aa_false_positive_control` (null šum nesmí pálit ≤8 %) a
`paired_rollout_identity_cancels_to_zero` (párování ruší nuisance přesně).
41 anti-regrese testů zelených, build default/gpu/viz čistý. Smoke
(tiny_200k, cold LR 2e-5): governor mlčí dokud trénink jede, na plató
diagnostikuje `lr_too_cold`, otevře trial, 40 párových vzorků → `BelowFloor`
(1.3× LR bump na nano modelu je pod 0.003 BPB floor), `.phage` perzistuje.

**CPU + GPU (backend-agnostický core).** `end_step(.., &mut dyn TrialBackend)`.
`GpuTrialBackend` snapshotuje **on-device** — přidány
`DecoderTransformerGpu::{snapshot_state, refresh_state, restore_state}` +
`GpuStateSnapshot` (kopíruje váhy + AdamW momenty + Muon momentum přes
`copy_buffer_to_buffer`, žádný host readback), rollout přes `train_step_batch`/
`step_muon_hybrid`, probe BPB přes `eval_step_batch`. Trialy i reflex rollback
běží na GPU plnou rychlostí (jen 2 perzistentní snapshot buffer sety navíc).
GPU verifikace `tests/governor_gpu.rs`: bit-exact snapshot→step→restore vč.
Muon w_q — **pozor na zero-init w_o**: 1. krok má nulové matrix-grady, test
musí warmupnout. **GPU gotcha:** divergence se klíčuje JEN na loss —
`grad_norm` je na non-report stepech NaN (skipnutý readback), NE divergence
(jinak reflex mlátí každý druhý krok). GPU smoke (cold LR): stejné chování i
BPB jako CPU, trialy on-device.

**Hranice (poctivě).** (a) Strukturální změny (tvary tenzorů, swap identity
optimizeru AdamW↔Muon) nejdou hot-swapnout — governor je „medikace" živými
skaláry, ne „chirurgie". (b) Krátký trial nevidí pomalé/odložené efekty
(weight decay, tvar schedule) — ty jsou nízko-confidence. (c) Trial overhead
(2×horizon shadow kroky per trialing step) je reálný a viditelný (GPU smoke
~20 → ~60 ms/step během trialů); gated cooldownem + posterior prahem. (d)
Probe je z val splitu — mírný Goodhart, hlavní BPB stále z plného valu.

**Follow-up po prvním reálném GPU běhu (enwik8_10m, 13.9M, tier1+canon).** Běh
ukázal 3 věci k opravě (governer NEškodil — committed zůstal identita, val
2.69→2.05/3 epochy — ale byl *němý*: každý trial belowfloor):
1. **Runtime A/A σ kalibrace** (`Calibrating` state) — margin byl zaseknutý na
   0.0096 (n=40), protože `cfg.sigma=0.02` floor přebíjel mnohem menší
   empirickou párovou σ. Před prvním trialem teď governer pustí
   `calib_samples` (16) párových vzorků reálně-velkým arm a nastaví
   `cs.sigma = max(1.5·std, 0.001)`. Smoke: změřená std **0.000187** → σ
   **0.001**, margin **0.0096 → 0.00048** (20×). Floor (0.003) je teď binding
   constraint, ne statistická nejistota — reálný efekt >0.003 *projde*.
2. **Schedule-aware diagnóza** — `lr_too_hot` hard-gated když LR roste
   (warmup grady jsou velké z principu; reálnou explozi chytá reflex),
   `lr_too_cold` gated když LR klesá (WSD tail — zvyšovat LR je špatně). LR
   směr z `lr_now` vs vlastní slow EWMA (`Fingerprint::lr_rising/lr_falling`).
3. **Regime cap** — warmup regime capnut na `min(warmup_steps, total/10)`
   (běh měl warmup 2000 z 2564 ⇒ 78 % „warmup", degenerovaný bandit kontext).

Verifikace: `calibration_sigma_is_what_unblocks_adoption` (σ=0.02→BelowFloor,
σ=0.006→Adopt na stejném 0.01 efektu), `diagnosis_respects_schedule`,
`regime_warmup_is_capped`, `calibration_enables_adoption_below_default_floor`
(MockBackend). 16 governor + 2 GPU testů zelených.

---

# Sprint 69 — Performance: měřicí základ + GPU occupancy

Cíl: vyždímat GTX 1650 (TU116, GDDR6, ~192 GB/s, ridge ~14.8) + i5-12400F (12T)
maximum. Roofline-grounded, měřením gated. **Baselines (nano d384/L6/B16/T256):**
CPU **20.6 s/step**, GPU **1.17 s/step** (17.6× GPU).

## Phase 0 — Měřicí základ (HOTOVO)

- `src/gpu/profiler.rs` — `GpuProfiler` přes wgpu `TIMESTAMP_QUERY` +
  `TIMESTAMP_QUERY_INSIDE_ENCODERS` (`encoder.write_timestamp` mezi dispatchi).
  `context.rs` enabluje feature když ji adapter má. `probe_copy_ms` validuje
  celou cestu; `tests/gpu_profiler.rs` zelený na GTX 1650 (de-riskuje Phase 4).
- `examples/aurora_train.rs` — `ms_per_step` + `tok_per_s` v JSONL `step`
  (windowed) i `epoch_done` (train-only = `(elapsed−eval)/steps`) + stdout.
- `tests/perf_smoke.rs` — katastrofický-regrese gate (tiny d64/L2, strop 250ms,
  ne flaky micro-bench).

## Phase 1 — CPU codegen (HOTOVO, ale NEUTRÁLNÍ — důležité zjištění)

- `.cargo/config.toml` `target-cpu=native` + `linalg::dot` (8 akumulátorů +
  `mul_add` → AVX2 FMA) v `linear_forward` + tied head.
- **MĚŘENÍM NEUTRÁLNÍ: 20.6 → 19.9 s/step (~3 %).** Dopředný dot je jen ~3 %
  stepu → **CPU NENÍ compute-bound na matmulu** (`vfmadd` se bez `mul_add`
  vůbec neemitoval — LLVM nekontrahuje f32). Bottleneck = Muon NS (par_iter přes
  6 bloků = půl-idle na 12T), per-step `Vec` alloc/zeroing, memory traffic.
  Velký CPU zisk = hluboký refactor (Muon-over-matrices reparallel, buffer
  reuse) → mizerná ROI vs GPU. Vektorizovaný dot ponechán (korektní, ~3 % plus).
  Pozor: native baked na tento CPU; každý gpu rebuild rekompiluje wgpu.

## Phase 2 — GPU batch→M occupancy (NAVRŽENO, turnkey edit-list)

**Proč:** `gpu_perf_bench` měří nano B16 batch speedup **1.00×** — současné
batchování je falešná paralelizace (B sekvencí sdílí `self.acts` arénu → wgpu je
serializuje WAW bariérami). GEMMy jsou occupancy-starved při M=T.

**Design (grid.z, čistší než host b-loop):**
- **Flatten M=B·T** (position-independent / per-row): embed-write, LN, QK-norm,
  všechny GEMMy (q/k/v/o/fc/proj + head), gelu, vadd, softcap, softmax_ce. Bind
  groups bindují CELÉ buffery (`as_entire_binding`) → stačí resize + dispatch
  dostane M=`(b*t) as u32` / count B·*. Žádný rebind.
- **MHA přes `wgid.z=b`** (grid `(T,n_head,B)`): kernel `qkv_base=bi*t*d_model`,
  `probs_base=bi*n_head*t*t`; per-b probs (`probs_cap *= mb`). Žádný extra
  push-constant. OOM až T=1024 vysoké B (B=1 record nezměněn).
- **embed + rope = host b-loop** (NELZE flatnout): `d_w_pos[ti%T] +=` koliduje
  přes B sekvencí (race); b-loop serializuje přes WAW → korektní. Potřebují
  `out_offset`/`dy_offset` push-constant.
- **Weight grady zdarma korektní:** outer_accum GEMM redukuje přes M=B·T = suma
  přes batch.

**Verifikace (klíč):** při B=1 je KAŽDÁ změna no-op (B·T=T, z=1, bi=0, offsety=0)
→ `gpu_parity` (13 testů) MUSÍ zůstat zelený. Pak batch-consistency:
`train_step_batch` grady == Σ B× `train_step`.

**Edit-list (vše to-do; resize + MHA-fwd byly prototypovány a revertnuty, ať
strom kompiluje — v git reflogu):**
1. `Activations::new`: `td=mb*t*d`, `td_ff=mb*t*d_ff`, `probs_cap=mb*n_head*t*t`;
   `ln*_inv/*_mean/f_*` → `bt`; logits/d_logits → `bt*v`. (`mb=max_batch.max(1)`)
2. MHA fwd kernel `bi=wgid.z` + qkv_base/probs_base; `dispatch_with_bind(+b)`,
   grid `(t,n_head,b)`.
3. MHA bwd (dscores/dq/dkv) kernely: `bi=wgid.z` + offsety; dispatch z=b.
4. embed fwd/bwd: `out_offset`/`dy_offset` push-constant + b-loop v encode.
5. rope fwd/bwd: `qk_offset` push-constant + b-loop.
6. `encode_forward(enc,t,b)`: flatten dispatche na M=B·T; embed/rope/mha b-aware.
7. `block_backward`/backward část `encode_train_element`: flatten + z=b.
8. `train_step_batch`: jeden batched encode místo B× `encode_train_element`.

**Pozor — kaskáda:** signatura `encode_forward`/dispatchů protáhne přes všechny
call-sites (train_step, eval_step, train_step_batch). ~40 vzájemně závislých
editů + 2–3 `gpu_parity` debug cykly. All-or-nothing (fwd+bwd layouty musí
sedět). → dedikovaný běh s plným budgetem. Očekávaný zisk ~1.3–1.4× @B8 T256
(roofline; empiricky 1.07–1.4× strop).
