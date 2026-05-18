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

## Sprint 60+ — otevřená pole

| Větev | Co odemyká | Effort |
|---|---|---|
| **Flash-attention WGSL** | Větší seq_len bez OOM, ~2× throughput | large |
| **KV cache + inference path** | Faster sampling, opens chatbot demo | medium |
| **RoPE + RMSNorm** | Modern positional + norm, Llama-style | medium |
| **Mixed precision (fp16 storage, fp32 compute)** | 2× VRAM headroom, blocked dokud Naga nepodpoří `f16` | large (waiting) |
| **enwik8 full (100 MB) overnight** | Hutter Prize comparable BPB | medium (waiting) |
| **GPT-2-small reprodukce** | n_layer=12, d_model=768 — VRAM tight, mixed precision dependency | medium-large |
| **Hyperparam search** (proper, ne hand-tuning) | Optuna / random search wrapper kolem `aurora_train` | medium |
| **SwiGLU FFN** | +2-3 % na quality dle literatury | small |

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
