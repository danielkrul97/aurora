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
- GPU side α-gradient (reduction kernel + Adam pro α buffer).
- λ scheduling — adaptivní λ podle current rate distribution. Mohlo by automatizovat trade-off.

---

## Sprint 34+ — Pokračování

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
