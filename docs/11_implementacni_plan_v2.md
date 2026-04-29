# Implementační plán Aurora — v2

> Pokračování `10_implementacni_plan.md`. Ten dokument zachycuje sprint 0–25 a je teď zamrzlý jako historický záznam (92 KB). Tady pokračujeme aktivně sprint 26+.
>
> **Princip:** žádné odhady v týdnech, postup po exit kritériích, breaking changes uvnitř výzkumné fáze jsou OK.

---

## Stav po Sprintu 29 (snapshot)

Co je hotové a v produkci (`main` branch):

- **Sprint 26 — BPE.** Linked-list trainer + min-heap encoder, O(N log N) obě strany. 1.1 MB / 1024 vocab pod 1 s. Token-level CPU API (`forward_backward_batch_tokens_stateful`, …). GPU `softmax_ce` přes stride loops podporuje libovolný vocab.
- **Sprint 27 — Spiking attention building blocks.** `src/attention.rs`: `LayerNorm`, `LinearProj`, `scaled_dot_product_attention`, `spike_gated_attention`. Forward + backward, gradienty ověřené central-difference. Stand-alone, **NEZAINTEGROVANÉ** do SNN buňky.
- **Sprint 28 — `aurora_train` + checkpointing.** Jediný hlavní example: GPU 2-layer megakernel + BPE + checkpoint save/load + auto-resume. Nahradil 6 starých `char_lm_*.rs`. `tests/lm_quality_regression.rs` jako anti-regrese (CPU+byte, CPU+BPE, checkpoint round-trip — pod 1 s).
- **Sprint 29 — GPU throughput fixy.** Workgroup_size 64→256 + MAX_HIDDEN 512→1024 v 2-layer megakernelu (4× paralelismu). Workgroup_size 64→256 v 4 elementwise kernelech (Adam, vector_add, scale, zero) — nutné pro `vocab=4096 × hidden=1024` matice (>65535 dispatch limit). **f16 storage cesta zablokovaná na úrovni Naga toolchainu** (wgpu issue #4384 — `enable f16;` zatím neimplementováno). Cesta zpět otevřená až přijde.

Smoke benchmark (před Sprint 30): tiny.txt 200 K subset, hidden=1024, vocab=4096 → val BPB **2.77** za 21 s.

Reálný cíl po Sprintu 30+ (z literárního review, viz dále): **1.30–1.40 BPB na enwik8** v overnight běhu na GTX 1650.

---

## Literature anchors pro další směřování

Plný research report v interní paměti agenta. Klíčové papíry, na které se v dalších sprintech odkazujeme:

- **Spike-driven Transformer V1** (Yao et al., NeurIPS 2023, arXiv:2307.01694) — SDSA: softmax + matmul → binary mask + Hadamard. Aurora má všechny stavební bloky hotové.
- **PLIF** (Fang et al., 2020, arXiv:2007.05785) — learnable τ.
- **AdLIF** (Bittar et al., 2024, arXiv:2408.07517) — adaptation current. +5–8 % na speech vs LIF.
- **Variance-conserving init pro SNN** (Klos & Memmesheimer, 2024, arXiv:2410.00580) — Kaiming-like vzorec respektující heaviside.
- **Learnable Surrogate Gradient (LSG)** (Lian et al., IJCAI 2023) — per-layer learnable α. +5–6 % na CIFAR-DVS.
- **TET loss** (Deng et al., ICLR 2022, arXiv:2202.11946) — per-timestep CE reweighting.
- **MPBN** (Guo et al., ICCV 2023, arXiv:2308.08359) — BN před heaviside, fold do θ při inference.
- **P-SpikeSSM / SiLIF** (arXiv:2406.02923, arXiv:2506.06374) — diagonal SSM reformulace, parallel scan místo sériového BPTT. Největší throughput unlock.
- **SpikeGPT** (Zhu et al., 2023, arXiv:2302.13939) — referenční baseline pro byte-level SNN-LM. 46M params → BPC 1.26 na enwik8.

---

## Sprint 30 — Init + LSG + spike-rate monitoring

**Cíl:** Aplikovat low-effort improvements identifikované v literatuře a měřit BPB delta.

**Záměrný omezený scope:** TET loss a GPU LSG α-gradient se odkládají do Sprintu 31. Důvod: TET vyžaduje úpravu `softmax_ce` kernelu (CPU + GPU), GPU α-gradient potřebuje reduction kernel — obojí přesahuje 2-h sprint budget.

**Deliverables:**

1. **Variance-conserving init** v `GatedLayer::new`. Vzorec respektuje target firing rate `ν₀` a heaviside kvantizaci:
   `σ_W = √(1 / (n_in × ν₀ × (1 - ν₀)))`
   pro `ν₀ = 0.1` (10% firing rate cíl). Aplikuje se na `w_in` i `w_rec`. Gating váhy zůstávají při 0.1× scale (slabší init pro stabilitu).

2. **Per-layer learnable α** (LSG) na CPU.
   - Nový field `alpha: f32` v `GatedLayer` (default 4.0 pro backward compat).
   - CPU backward používá `layer.alpha` místo globálního `self.spike_fn.alpha`.
   - Gradient: `∂L/∂α_l = Σ_t Σ_i d_h_spike[t,i] × ∂surrogate(v_t,i − θ)/∂α`,
     kde `∂surrogate/∂α = -2|v−θ| / (1 + α|v−θ|)³`.
   - Adam state pro α (1 element per layer).
   - GPU: `gated_cell_back` dispatcher čte `layer.alpha` při každém volání. **GPU NEROČÍTÁ α gradient v tomto sprintu** — α na GPU-only běhu zůstává konstantní, ale respektuje cokoliv co CPU vytrénovalo (přes `from_cpu`).
   - Checkpoint format bumpnut na `AURORA02`. Loader podporuje obě verze (v1 → α default 4.0, v2 → alpha v payloadu).

3. **Spike-rate monitoring** v `aurora_train`. Per-layer firing rate (mean spike count / (B × T × N)) tisknut v progress reportu vedle BPB. Žádný gradient zatím — jen instrumentace pro tuning.

4. **CPU regression test** — `tests/lm_quality_regression.rs` rozšířen o assert na rate hodnoty (po 1 epoše per-layer rate je v rozmezí 0.02–0.4, tj. nedošlo k saturaci nebo úplnému umrtvení).

**Exit kritéria:**
- `cargo test --release` zelený (45 → 46+ testů).
- `cargo test --features gpu --test gpu_parity` zelený (CPU vs GPU stále match — α default je shodná).
- Smoke run aurora_train na tiny_200k.txt s hidden=1024 ukáže val BPB **lepší nebo stejný** jako 2.77 baseline.
- Plný literature plan committed.

**Záměrně neřešíme v této iteraci:**
- TET loss (CPU+GPU `softmax_ce` kernel surgery).
- GPU side α-gradient (vyžaduje reduction kernel).
- Spike-rate gradient přes bias (CPU+GPU bias backward modifikace).
- MPBN (Sprint 32+).

---

## Sprint 30 — kritický nález (post-hoc)

Implementace Sprintu 30 odhalila zásadní fakt: **hidden vrstvy nikdy nefires**. Při post-hoc měření per-layer firing rate (přidaného právě v tomto sprintu) se ukazuje že po 1 epoše tréninku na tiny_200k @ hidden=1024, vocab=4096:

- `rates [0.0000, 0.0000]` (oba layery)
- val BPB 2.77 (≈ konstantní b_o prediction baseline)
- α stays at 4.00 (žádný gradient signal teče přes spiking path)

**Důsledek:** Aurora se učí čistě přes `b_o` (output bias), ne přes spiking dynamiku. Bez firing v hidden vrstvách `d_h_spike = W_out · d_logits · 0 = 0`, takže `d_w_in`, `d_w_rec` jsou nuly. Trénují se pouze biasy.

**Příčina:** Init je příliš slabý, membrane potential nikdy nepřekročí θ=1.0 z cold-start. Forget-gate-bias-1 trick udržuje β≈1, ale nezvyšuje vstupní amplitude.

**Pokus o oprava ve Sprintu 30:** variance-conserving init (Klos & Memmesheimer 2024). Reverted — formule nepasuje na one-hot input layer 0 (effective rate 1/n_in, ne ν=0.1), výsledek byl ještě tišší network.

**Otevřené možnosti pro fix v Sprintu 31:**

| Přístup | Effort | Riziko |
|---|---|---|
| Snížit `θ` z 1.0 na 0.5 (lehčí firing práh) | tiny | low — možná destabilizuje gate |
| `b_h` init kladná (push membrane up) | tiny | low |
| Layer-aware init: layer 0 σ_W=θ pro one-hot, deeper layers var-conserving | small | low |
| Spike rate l2 reg pushuje rate → ν₀=0.1 přes bias gradient | medium | medium |
| Curriculum learning (krátké chunky → dlouhé) | medium | low |
| Reset trick: explicitně přidat membrane noise při init | tiny | low |

**Doporučení:** **Sprint 31 začne tímto problémem.** Až tohle bude vyřešené, teprve pak má smysl honit TET / GPU LSG α-grad. Bez firing nemá α gradient nikdy hodnotu (matematicky je 0).

---

## Sprint 31 — Probudit hidden vrstvy ✓ DONE

**Výsledek:** První skutečný BPB krok dolů od baseline. Layers fire, gradients flow.

**Co se udělalo:**

1. **Positive `b_h` init** (`DEFAULT_B_H_INIT = 0.3`) — kritická změna. Přidá konstantní bias do membrane potential, takže neuron sedí na ~37 % cesty k thresholdu. Slabé inputy už pak postačí na firing, surrogate gradient je non-zero, gradient signal teče do `W_in` / `W_rec`.

   Mikro-teorie: s β = sigmoid(b_gate=1) ≈ 0.73 a b_h = 0.3, equilibrium membrane v_eq = 0.3 / (1 − β) = 1.11, těsně nad θ=1. Naivní steady-state firing rate ≈ 1/3.

2. **Membrane stats** v `MultiLayerCache::membrane_stats()` (mean, peak per layer) — diagnostika, která by pomohla najít runaway dříve. Aurora_train teď tiskne `v mean/peak [...]` v `epoch_diagnostics`.

3. **Regression test** rozšířen: `cpu_byte_lm_converges_under_one_epoch` teď asserts `rates[l] > 0.001` per layer. Catches future regressions kde init/dynamic změna zase vrstvy umlčí.

**Měřené delty (proti Sprint 30 baseline):**

| Test | Před | Po | Delta |
|---|---|---|---|
| GPU aurora_train tiny_200k h=1024 vocab=4096, 1 ep | val BPB 2.768, rates [0.0, 0.0] | val BPB **2.684**, rates [0.35, 0.22] | **−3.0 %** + firing |
| CPU regression byte 50K h=64, 1 ep | val BPB 4.83, rates not asserted | val BPB **4.68**, rates [0.50, 0.54] | **−3.1 %** + firing |
| CPU regression BPE 50K h=64 vocab=384 | val BPB 4.05 | val BPB 4.07 | žádný regression |

**Otevřené v Sprintu 31 (záměrně):**

- **Membrane runaway** — peaks 7k–30k pozorovány. Nestala se NaN, ale numericky riziková pro delší tréninky / hlubší modely. Varianty fixu: rate l2 reg pull rate dolů, soft membrane clamp, snížit `b_gate` init pro silnější leak. Ponecháno jako Sprint 32 priority.
- **GPU side α-gradient** — α na GPU stále jen čte CPU-evolved hodnotu. Sprint 32+.
- **TET loss** — softmax_ce kernel surgery. Sprint 32+.
- **Spike rate l2 reg** přes bias gradient. Sprint 32+.

---

## Sprint 32 — Spike-rate l2 reg ✓ DONE (částečně)

**Hotovo:**

1. **Spike-rate l2 reg na bias** (CPU + GPU). `RateReg { target: 0.1, lambda: 0.01 }` na `MultiLayerGatedSnn`. Dva nové GPU kernely:
   - `SpikeRateReduceKernel` — workgroup-shared reduction `h_spike_packed[layer_slice]` → scalar count per layer (1 dispatch per layer per chunk)
   - `RateRegBiasKernel` — adds `2λ × (count/total − target)` ke každé `b_h_grad[i]` v layer
   - Total overhead: ~50 µs per chunk per layer, zanedbatelné

   CPU varianta dělá totéž v `backward_lm_tokens_into` před návratem.

   `MultiLayerGatedSnnGpu` přebírá `rate_reg` z CPU netu při `from_cpu`. `aurora_train` pracuje s defaultními hodnotami; user může přepsat na CPU netu před `from_cpu`.

**Měřené delty proti Sprint 31 baseline:**

| Test | Před | Po | Delta |
|---|---|---|---|
| GPU aurora_train tiny_200k h=1024 vocab=4096, 1 ep | val BPB 2.684, peaks 7k/30k | val BPB 2.684, peaks 9.6k/32k | žádná, runaway nezmenšen |
| GPU parity tests | 6/6 ✓ | 6/6 ✓ | žádná regrese |
| CPU regression byte 50K | 4.68, rates [0.50, 0.54] | 4.68, rates [0.50, 0.54] | žádná regrese |

**Co jsme se dozvěděli (Adam absorbuje signal):**

Per-layer rate reg correction je konstantní napříč vrstvou (`2λ × diff` přidáno ke všem `b_h_grad[i]`). Adam moment-normalizace tuto konstantní pertubaci v podstatě absorbuje — efektivní step na bias je drcen tím že ostatní parametry mají typically srovnatelné magnitude gradientů. Vyšší λ (0.05–0.1) sice viditelně shifne rate, ale na CPU regression suite (50K corpus, 80 chunks) zabije firing entirely (BPB 7.77 = random).

**Důsledek:** mechanismus je v pipelině, ale potřebuje *smarter formulation* aby měl reálný účinek bez destabilizace. Kandidáti pro Sprint 33:

| Přístup | Effort | Riziko |
|---|---|---|
| **Per-neuron rate reg** — každý neuron má vlastní rate, vlastní correction; jemnější kontrola | medium | low |
| **AdamW-style decoupled rate reg** — bias decay aplikován po Adam stepu, mimo gradient cestu | small | low |
| **Surrogate-aware rate reg** — `correction_per_neuron = 2λ × (rate − target) × Σ surrogate(v_t,i)`, projde naturalním gradient flow | small | medium |

**Záměrně neřešeno v Sprintu 32 (přesouváme do 33+):**
- TET loss (`softmax_ce` kernel surgery — větší než 2-h sprint).
- GPU side α-gradient (reduction kernel, malý ale samostatný).

---

## Sprint 33 — Per-neuron rate reg ✓ DONE

**Cíl:** Vyřešit Sprint 32's "Adam absorbuje signal" problém přechodem na per-neuron correction místo per-layer scalar.

**Hotovo:**

1. **Per-neuron rate reg** (CPU + GPU). Každý neuron počítá vlastní rate a dostává vlastní correction:
   - CPU: `rate_i = (1/T) × Σ_t cache.layers[l].h_spike[t][i]` per neuron, `b_h_grad[l][i] += 2λ × (rate_i − target)`.
   - GPU `SpikeRateReduceKernel` přepsán: output shape `[n_layers × n_hidden]` (per-neuron sum across batch+time), thread-per-neuron parallelism.
   - GPU `RateRegBiasKernel` přepsán: čte `count[count_offset + i]`, `inv_total = 1/(B×T)`.

2. **Default λ ladění:** zůstává 0.01 — nedrobí firing na malé regression suite, dává viditelný (nedramaticky) efekt na aurora_train. Vyšší λ (0.05) zlepší tuning rate ale zabije firing na 50K corpus regression (BPB 7.39 vs threshold 6.0). Production callery můžou λ přepsat: `net.rate_reg.lambda = X`.

**Měřené delty proti Sprint 32 baseline:**

| Test | Sprint 32 | Sprint 33 | Delta |
|---|---|---|---|
| GPU aurora_train tiny_200k 1 ep | val BPB 2.684, peaks 9.6k/32k | val BPB **2.682**, peaks 7.7k/30k | −0.001 BPB, peaks −20 % |
| CPU regression byte 50K | val BPB 4.68, rates [0.50, 0.54] | val BPB 5.04, rates [0.45, 0.49] | rate reg už má signal (rates kompresí), drobná BPB regrese |
| GPU parity 6/6 | ✓ | ✓ | žádná regrese |

**Klíčová matematická poznámka:**

Per-layer scalar correction se v Adamu chová jako *layer-wide bias decay* (všechny neurony stejně), což moment normalization v podstatě zruší. Per-neuron correction je *per-parameter signal* který Adam normalizuje na ~unit-magnitude per parameter — stejně jako jakákoliv jiná složka gradientu. **Mathematically the same scale, but per-neuron actually flows.**

**Otevřené (Sprint 34+):**
- TET loss (`softmax_ce` kernel surgery — CPU + GPU).
- ~~GPU side α-gradient (reduction kernel + Adam pro α buffer)~~ — Sprint 34.
- λ scheduling — adaptivní λ podle current rate distribution. Mohlo by automatizovat trade-off.

---

## Sprint 34 — GPU α-gradient ✓ DONE

**Cíl:** dokončit LSG smyčku tak, aby α evolvovala i při GPU-only tréninku (do Sprintu 33 byla učenlivá jen na CPU, GPU jen četla současnou hodnotu).

**Hotovo:**

1. **`gated_cell_back` kernel:** přidaná binding 7 — `d_h_spike_packed` buffer (B × L × T × N, stejný indexing jako `h_v_pre_packed`). Kernel persistuje per-(b, t, h) hodnotu `d_spike` na 1 dodatečný write per thread.

2. **Nový kernel `AlphaGradReduceKernel`:** post-backward 256-thread workgroup-shared reduction přes (B × T × N) entries pro každou vrstvu. Spočítá `Σ d_spike · ∂surrogate/∂α = Σ d_spike · (-2|v−θ|) / (1 + α|v−θ|)³` → 1 scalar per layer do `fb.alpha_grad[2]`.

3. **`step_adam_gpu` rozšířen:** download `alpha_grad` (jen 2 floaty), apply Adam na host straně (per-layer 1-element optimizer), update `alpha_per_layer` který přečte cell_back další chunk.

4. **Single-layer kompatibilita:** `snn_gated_gpu.rs` má dummy `d_h_spike_packed` buffer, signature parita zachována, gpu_parity testy zelené.

**Měřené delty proti Sprint 33 baseline:**

| Test | Sprint 33 | Sprint 34 | Delta |
|---|---|---|---|
| GPU aurora_train tiny_200k 1 ep | val BPB 2.682, α [4.00, 4.00] | val BPB 2.686, α **[4.01, 4.05]** | α se konečně hýbe |
| GPU parity 6/6 | ✓ | ✓ | žádná regrese |
| Per-chunk overhead (α reduce) | 0 | ~50 µs | pod 0.1 % chunk času |

**Drobná BPB regrese** (2.682 → 2.686) je očekávaná — α se v 1 epoše posunula jen o 0.01–0.05, což je perturbace bez dostatečného času na compensaci v ostatních váhách. Při delším tréninku (overnight enwik8) by α mělo dostat čas zkonvergovat na lepší hodnotu.

**Pozn. k formě integrace:** α-Adam state na GPU by byl 8 floatů, neefektivní (overhead launch > výkon). Ponecháno na hostu, download α-grad → CPU Adam → write back. Žádná měřitelná latence.

**Otevřené (Sprint 35+):**
- TET loss (CPU + GPU softmax_ce surgery).
- λ scheduling — adaptivní λ podle current rate.
- Overnight enwik8 benchmark — teď máme všechny nástroje, čas to změřit.

---

## Sprint 35 — SDSA modul (CPU) ✓ DONE (modul, integrace = Sprint 36)

**Cíl:** Implementovat Spike-Driven Self-Attention (Yao et al. 2023, arXiv:2307.01694) jako stand-alone modul s ověřenou matematikou. Replace softmax+matmul over Q/K/V se linear attention nad binary spike Q/K/V — fits Aurora's stateful-between-chunks design (attn accumulator carry like `h_v_after`).

**Hotovo:**

1. **`src/sdsa.rs`** — `SdsaCell { layer_norm, w_q, w_k, w_v, spike_fn }`. Forward + backward:
   ```text
   h_norm     = LayerNorm(h_spike)
   Q          = step(W_q · h_norm − θ)
   K          = step(W_k · h_norm − θ)
   V          = step(W_v · h_norm − θ)
   attn_t     = attn_{t−1} + K_t ⊙ V_t           ← linear, not quadratic
   sdsa_out_t = Q_t ⊙ attn_t
   y_t        = h_spike_t + sdsa_out_t           ← residual
   ```
   - `SdsaCache` ukládá `attn_states` per timestep + `final_attn_state` pro carry mezi chunky.
   - `SdsaGrads` + `SdsaAdam` připraveny pro Adam step.

2. **Reuse Sprintu 27 building blocks:** `LayerNorm`, `LinearProj` z `src/attention.rs`. Žádná duplikace kódu.

3. **`tests/sdsa_smoke.rs`** — 4 testy:
   - forward dává finite output, attn_state monotonně neklesá (K⊙V ≥ 0)
   - backward dává finite gradients
   - attn_state správně carry-on mezi sekvenčními voláními
   - když `Q == 0` (forced via `w_q.fill(0)`), `y == h_spike` exactly (residual fallback)

**Záměrně neudělané v Sprintu 35** (Sprint 36 priorita):
- Wiring do `MultiLayerGatedSnn::forward_lm_tokens_stateful` + `backward_lm_tokens_into` (vyžaduje úpravu `GatedState` o `attn_state` field, hoisting logits computation out of t-loop, dense W_out matvec když SDSA on).
- Adam coupling pro SDSA params (gradient propagace už hotová, jen nepřipojeno k optimizeru).
- Checkpoint format v3 s SDSA params.
- GPU port (Sprint 37+ — vyžaduje 4 nové kernely: linearize attn, residual add, sdsa backward, plus existing layer_norm/linear pre-existing forward kernels).

**Pozn. ke "spike forward + surrogate backward":** SDSA forward používá heaviside (step) pro binarizaci Q/K/V. Backward používá fast-sigmoid surrogate. Z principu surrogate-gradient SNN tréninku **analytical gradient ≠ true derivative** — finite-difference test by selhal. Testy proto verifikují konzistenci (finite outputs, monotone accumulator, residual fallback) místo strict numerical match.

**Reálný cíl po integraci:**

SpikGPT (RWKV-style spiking, 46M params) hits BPC 1.26 na enwik8. Aurora s SDSA + recurrent gated cell + 1024 hidden by realistic mohlo dosáhnout 1.30–1.40 BPB v overnight běhu. To je jediná architektura která může citlivě posunout BPB plato; všechny ostatní fixy (bod 30–34) byly inkrementální.

---

## Sprint 36 — MetricsWriter + Python diagnostics ✓ DONE

**Cíl:** Strukturovaný metrický log z aurora_train do JSON Lines + Python post-mortem skript pro diagnostické grafy. Tier 1 z analýzy (žádné nové Rust deps).

**Hotovo:**

1. **`MetricsWriter`** v `examples/aurora_train.rs` — hand-rolled JSONL writer (avoid serde_json dep, schema je fully controlled). Eventy:
   - `{"event":"run", ...}` — 1× při startu (corpus, arch, hyperparams, GPU)
   - `{"event":"chunk", ...}` — per progress report (epoch, chunk_idx, train_bpb, elapsed_s)
   - `{"event":"epoch_done", ...}` — 1× per epoch (train_bpb, val_bpb, rates, alphas, v_means, v_peaks, elapsed_s)
   - `{"event":"checkpoint", ...}` — po save
   - Flush per event → `tail -f` má sub-second lag, SIGTERM-killed runs zachovávají use­ful trail
   - NaN/Inf serializované jako string `"NaN"` (deliberately invalid JSON token aby parser hlasitě selhal)

2. **CLI flag `--metrics <path>`** s default `logs/<ckpt_stem>_<unix_ts>.jsonl`. Opt-out: `--metrics /dev/null`.

3. **`analysis/diagnostics.py`** — pure Python (jen std lib + matplotlib):
   - Parsuje JSONL, summary v terminálu (run config, BPB nejlepší, last rates/alphas/peaks, runaway warning)
   - 4 PNG plots: BPB convergence (per-chunk + per-epoch), per-layer firing rates, α evolution, membrane mean/peak (s log-scale + θ=1.0 referenční čára)
   - Usage: `python analysis/diagnostics.py logs/run.jsonl [--out plots/]`

**Co tohle umožňuje (diagnostika):**
- "Proč BPB stagnuje" → křivka loss + per-chunk variance ukáže zda je gradient nulový nebo skutečné plato
- "Kdy začalo runaway" → membrane peaks per epoch; když najednou skok = identifikujeme chunk
- "Konverguje α?" → α evolution graf
- "Aurora-train vs SDSA-aurora" → 2 JSONL souboary + side-by-side plots = empirický A/B

**Neudělané (Tier 2+3 z analýzy):**
- Snapshot dumps (spike rastery, weight histogramy) per epoch — vyžaduje binary `.npy` formát + bigger file management. Sprint 37+, ne kritické.
- Live UI (rerun.io) — overhead vs. value tradeoff. `tail -f logs/<run>.jsonl | jq` v terminálu pokrývá 80 %.

---

## Sprint 37 — Stop membrane runaway ✓ DONE

**Cíl:** Overnight enwik8_10m run odhalil že membrane potential při tréninku exploduje na 10⁷, čímž zabíjí f32 precizi a trénink stagnuje. Tři koordinované změny aby se to víc nestávalo.

**Pozorování z overnight runu** (`logs/overnight_20260429_0112.log`):
- Best val BPB **3.21** v epoše 1, pak 12 epoch plato
- Membrane peaks **10–27 milionů** (θ=1.0)
- Rate distribution bimodal: L0 **0.5–0.66** (over-firing), L1 **0.03–0.06** (silenced)
- Sample text = Wikipedia syntax noise bez sémantiky

**Tři změny:**

1. **Hard membrane clamp** `h_v_pre.clamp(-10, 10)` v CPU `forward_lm_tokens_stateful` + WGSL `forward_mega.rs` + `forward_mega_2layer.rs`. f32 precision zachována, dynamika nad |v|=10 ztracena (acceptable — θ=1.0 je 10× pod cap).

2. **`b_gate` init 1.0 → 0.0** (β = sigmoid(0) = 0.5, ne 0.73). Half-life recurrent carry: 1 timestep místo ~3 → exponenciální runaway potlačen u zdroje.

3. **`W_rec` scale 0.9/√H → 0.3/√H** (3× menší). Working spectral radius klesá z ~0.52 na ~0.17 → strongly contractive recurrent dynamics.

4. **`b_h` init 0.3 → 0.5** (kompenzace za nižší β: equilibrium `b_h/(1−β) = 1.0` zachován blízko θ).

**Měřené delty na tiny_200k smoke:**

| Test | Sprint 36 | Sprint 37 | Delta |
|---|---|---|---|
| GPU aurora_train tiny_200k 1 ep | val BPB 2.685, peaks 13k/34k | val BPB **2.613**, peaks **+10.00/+10.00** | **−0.07 BPB**, 1.3M× lower peaks |
| Rates | [0.37, 0.22] | [0.40, 0.09] | L1 close to target 0.1 |
| α | [4.01, 4.05] | [4.03, 4.05] | beze změny (běh příliš krátký) |
| GPU parity 6/6 | ✓ | ✓ | žádná regrese |
| CPU regression byte | 4.68 BPB, rates [0.50, 0.54] | 4.74 BPB, rates [0.45, 0.60] | drobná regrese kvalita (-0.06) ale stable |

**Best Aurora result ever**: 2.613 BPB. Sprint 30 baseline byla 2.768 — celková zlepšování za 7 sprintů: **−0.16 BPB**.

**Pozn. ke checkpointu:** Existující `data/enwik8_10m_h1024_v4096.ckpt` má staré (b_gate=1, runaway-trained) váhy. **Smaž ho** před fresh overnight runem; nebo bude pokračovat z broken stavu.

**Otevřené (Sprint 38+):**
- Nový overnight enwik8_10m benchmark se Sprint 37 dynamics — to je teď první priorita.
- Adaptive λ pro rate reg podle membrane magnitude — méně urgent po clampu.
- TET loss, SDSA wiring — stále čekají.

---

## Sprint 38 — SDSA wiring do MultiLayerGatedSnn ✓ DONE (CPU)

**Cíl:** Sprint 35 SDSA modul integrovat do CPU trénovací cesty `MultiLayerGatedSnn`. Forward, backward, Adam, end-to-end test.

**Hotovo:**

1. **`GatedState` rozšířen** o `attn_state: Vec<f32>` (default empty). `fresh_states()` alokuje attn_state jen pro top vrstvu když je SDSA on.

2. **`MultiLayerGatedSnn`** dostal `pub sdsa: Option<SdsaCell>` + privátní `sdsa_adam`. **`MultiLayerGatedGrads`** rozšířen o `sdsa: Option<SdsaGrads>` (umožňuje parallel batch trainers s vlastním SDSA grad).

3. **`enable_sdsa(&mut rng)`** API — turn on SDSA na existujícím netu. Default off zachovává backward compat se všemi existujícími testy.

4. **Forward**: po SNN cell t-loopu, pokud SDSA on, sdsa_forward přes top-layer h_spike, residual `y = h_spike + sdsa_out`, dense W_out matvec proti `y[t]`. Pokud SDSA off, existing sparse W_out cesta beze změny.

5. **Backward**: nový helper `backward_top_through_sdsa()` — pre-počítá d_y a W_out + b_o grads (dense outer product), volá sdsa_backward, vrací per-t d_h_spike pro top vrstvu. Hlavní per-t reverse loop pak používá tuto pre-computed hodnotu místo sparse `matvec_transpose(W_out, d_logits)`.

6. **Adam**: `step_adam()` volá `sdsa_step_adam()` na konci pokud SDSA on. **`MultiLayerGatedGrads::{zero, add, l2_norm, scale}`** všechny pokrývají SDSA pole.

7. **`tests/sdsa_integration.rs`** — 2 testy:
   - `sdsa_enabled_forward_backward_train_step`: SDSA on, ověří finite loss, Adam updateuje SDSA váhy (W_q[0] se hýbe), attn_state je non-zero po několika chunks
   - `sdsa_disabled_path_unchanged`: SDSA off, fresh_states nezalokuje attn_state, training nedotčen

**Test results:**
- 51/51 unit testů (49 + 2 SDSA integration)
- 6/6 GPU parity (žádná regrese)
- Clippy clean v novém kódu

**Záměrně neudělané** (Sprint 39):
- **GPU port** — aurora_train je GPU-only, takže SDSA zatím není používaná v hlavní training cestě. Vyžaduje nové WGSL kernely: linear-attention forward (per-t accumulator update + Hadamard q×state), linear-attention backward (reverse cumulative attn-grad), spike binarize Q/K/V via existing surrogate.
- **Aurora_train `--sdsa` flag** — bude přidán až SDSA poteče přes GPU.
- **Checkpoint v3** s SDSA params — momentálně se SDSA váhy ztratí přes resume.

**Důsledek pro empirické testování:**

SDSA matematika je verified, integrace do CPU pipeline je hotová a otestovaná. Ale **Aurora_train neschopen SDSA**, takže overnight enwik8_10m benchmark se SDSA on **vyžaduje Sprint 39** (GPU port). Do té doby SDSA experiment dostupný jen přes CPU regression-style smoke testy.

---

## Sprint 39 — Empirická validace SDSA ✓ DONE (negative result)

**Cíl:** Před investicí do GPU portu SDSA (~5 h focused work) ověřit empiricky, jestli SDSA v současné implementaci skutečně pomáhá. Tier 1 z analytického přístupu — měřit, ne kódovat.

**Implementace:**

1. **`aurora_train --cpu` flag** — přepne na čistý CPU `MultiLayerGatedSnn::forward_backward_batch_tokens_stateful`. Nefunkční pro hidden=1024 (CPU pomalý), ale vhodné pro malé experimenty.

2. **`aurora_train --sdsa` flag** — zapíná `enable_sdsa(rng)`. Validace: `--sdsa` bez `--cpu` vrátí error (GPU SDSA ještě není).

3. **Backend enum** v aurora_train pro abstraction GPU/CPU. Per-backend state allocation, dispatch logic. Připraveno pro Sprint 40+ GPU port.

4. **Header** + JSONL run event nyní obsahují backend name (např. "NVIDIA GeForce GTX 1650" nebo "CPU (rayon)"), title je "Aurora · 2-layer gated SNN · BPE · SDSA" když SDSA on.

5. **Regression check**: GPU mode dává val BPB 2.610 na tiny_200k (vs Sprint 37 baseline 2.613) — žádná regrese.

**Empirický A/B test** — tiny_200k, vocab=1024, hidden=128, chunk=32, lr=0.003, 5 epoch:

| Epoch | bez SDSA | s SDSA |
|---|---|---|
| 1 | 3.270 | 3.265 |
| 2 | 3.235 | 3.278 |
| 3 | 2.982 | 3.161 |
| 4 | 2.756 | 2.912 |
| 5 | **2.628** | **2.765** |

**SDSA je v současné implementaci o 0.137 BPB HORŠÍ.** Convergence pomalejší, finální výsledek slabší.

**Hypotézy proč:**

1. **Q/K/V spikes téměř nikdy nefires** — LayerNorm centruje hodnoty kolem 0, threshold θ=1.0 → spike rate ~0. SDSA contribution `Q ⊙ attn_state` je víceméně nulová. Residual `y = h_spike + 0` redukuje SDSA na no-op forward, ale gradient noise tečou skrz dead path → škodí.

2. **Init scale W_q/W_k/W_v** (Xavier `√(6/(d_in + d_out))`) je pro non-spike outputy. Pro pre-activation která má jít do heaviside nad θ=1.0 by mělo být **mnohem větší**.

3. **Architektura nesedí pro tento setting** — Yao 2023 dělal vision; LM dynamics jsou jiné.

**Co s tím (volby):**

| Přístup | Effort | Riziko |
|---|---|---|
| **A. Skip SDSA, focus na PLIF/AdLIF/MPBN** | medium | low — má jasný precedens v literatuře |
| **B. Tune SDSA θ + init pro vision→LM transfer** | medium | medium — možná neopravitelné |
| **C. Akceptovat negative result, dokumentovat, ukončit SDSA branch** | low | žádné — ale opouštíme Aurora's potenciálně novel contribution |
| **D. Bigger A/B na velké data (vocab=4096, hidden=512+)** | medium (CPU pomalý) nebo wait pro GPU port | low — empirika přes víc bodů |

**Cumulative pokrok od začátku v2 plánu:**
- Sprint 30 baseline: 2.768 BPB
- Sprint 37 (membrane fix): 2.613 BPB (−0.155)
- Sprint 39 SDSA test: SDSA prozatím ne. Architectural skok blocked dokud neopravíme spike-firing v Q/K/V.

**Tests:** 51/51 unit + 6/6 GPU parity (+ 2 SDSA integration ze Sprintu 38). Žádná regrese.

---

## Sprint 40 — AdLIF (CPU, fixed params) ✓ DONE (negative result)

**Cíl:** Druhá architektonická intervence z research review — AdLIF adaptation current
(`w_t = ρ·w_{t-1} + b·spike_t`, subtracted from `v_pre`) — neurony co často firují si
budují "únavu" `w` co potlačuje subsequent firing. Stejný pattern jako Sprint 39: implementuj
minimal viable, A/B test, rozhodni o GPU portu.

**Implementace:**

1. `GatedState.adlif_w: Vec<f32>` (carries between chunks, empty když opt-out).
2. `MultiLayerGatedSnn.adlif_enabled: bool` + `enable_adlif()` opt-in.
3. Forward: `v_pre -= w_prev` před clamp, pak `w_t = ρ·w + b·spike` po spike.
4. Backward: per-layer `d_w_carry` carries reverse in time. `d_h_spike[t] += b·d_w_carry`,
   pak `d_w_carry = ρ·d_w_carry − d_v_pre[t]`.
5. Fixed `ρ=0.95, b ∈ {0.3, 0.1}` — žádné Adam (Sprint 41 by je mohl udělat learnable).
6. `aurora_train --adlif` flag (jen s `--cpu` zatím).

**A/B test** — tiny_200k, vocab=1024, hidden=128, 5 epoch, stejné jako Sprint 39 SDSA test:

| Epoch | bez AdLIF | AdLIF b=0.3 | AdLIF b=0.1 |
|---|---|---|---|
| 1 | 3.270 | 3.295 | 3.264 |
| 5 | **2.628** | 3.059 | 3.006 |

**AdLIF horší o 0.38–0.43 BPB.** Convergence pomalejší.

**Diagnóza:**

Aurora už má spike-rate l2 reg (Sprint 32–33), který dělá podobnou práci jako AdLIF —
suppression over-firing neuronů. AdLIF je redundantní damping který odebírá expressivitu.
Plus fixed b je špatně laděné: at rate=0.5, w_eq=b·rate/(1−ρ) → 3.0 (b=0.3) nebo 1.0 (b=0.1).
Membrane equilibrium `(b_h − w_eq)/(1−β)` klesá z 1.0 na −5.0 (b=0.3) nebo −1.0 (b=0.1) →
network silenced.

**Wzor: dvě architektonické intervence v řadě negativní (SDSA, AdLIF).**

Aurora's existující stack (gated cell + clamp + per-neuron rate reg + LSG α) je už dost
adaptivní pro daný compute budget. Inkrementální architektonické add-ons se utopí v
existující regularizaci.

**Doporučení (Sprint 41+):**

1. **Pivot na CAPACITY místo dynamics** — jediná cesta k podstatnému BPB skoku je víc
   parametrů (deeper, wider) nebo víc dat (real overnight enwik8). Ne jemná architecture
   úprava existujícího cell.
2. **Reálný overnight enwik8_10m** — Sprint 37 dynamics (clamp + b_gate=0 + small W_rec)
   nebyly nikdy testovány na 10 MB víc-epoch. Bez toho data point je každá další
   architektonická volba spekulativní.
3. **Učinit AdLIF/SDSA learnable** může pomoci, ale empirický signal po 2 negativních
   testech je proti.

---

## Sprint 41+ — Pokračování

---

## Sprint 33+ — možné větve (rozhodneme po měření 32)

| Větev | Co odemyká | Effort | Zdroj |
|---|---|---|---|
| **SDSA integrace do recurrent cell** | Aurora's novel contribution; první SOTA-class arch crossover | medium | Yao 2023 |
| **PLIF + AdLIF A/B** | Možná úspora `W_β_rec` matmul, +5-8 % kvalita | small-medium | Fang 2020, Bittar 2024 |
| **MPBN/tdBN** | Stabilita pro hlubší stack (4–6 vrstev) | small | Guo 2023 |
| **P-SpikeSSM / SiLIF parallel scan** | Největší throughput win — 256-step BPTT → log₂(256) = 8 syncs | large | arXiv:2406.02923, 2506.06374 |
| **enwik8 plný overnight benchmark** | Realistický BPB cíl 1.30–1.40 | medium (waiting + tuning) | — |

---

## Co tento plán **záměrně neřeší** (zděděno z 10)

- **LLM trénink scale** (GPT-X velikosti). Aurora není generický LLM; cíl je biologically-flavored architektura competitive na enwik8.
- **Distribuovaný trénink, multi-GPU**. Single GTX 1650 je hardware constraint.
- **Tensor-core specifické optimalizace** (FP16 compute s tensor cores) — TU116 je nemá.
- **ANN→SNN konverze** (nesedí na from-scratch gated recurrent).
- **Webové UI / vizualizace** (logging + Python notebook stačí).

---

## Open questions kde literatura mlčí

(Aurora pracuje v unsupported territory v těchto směrech — z research reportu:)

1. **Per-neuron sigmoid forget gate × subtract-on-spike** (bez hard reset). Většina gated-SNN má hard reset.
2. **BPE × byte-level SNN-LM interakce.** Long-tail token statistiky × spike rate reg → nestudované.
3. **Spiking SDPA zapojená do recurrent gated cell** (nikoliv parallel-time block). SpikeGPT používá RWKV time-mixing.
4. **Active-indices kompakce v WGSL workgroup memory** — žádné prior art na designu kernelu.
5. **GTX 1650 (no tensor cores) jako serious SNN training target** — všechny SOTA benchmarky jsou A100/H100 / Loihi.

Pokud Aurora dosáhne reasonable BPB, body 1, 3 a 5 jsou citovatelný originální příspěvek.
