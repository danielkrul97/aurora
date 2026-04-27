# 03 — Trénování neuronových sítí a LLM

> Jak se moderní LLM s biliony parametrů skutečně trénují — od backpropagation přes optimizery, mixed precision a distribuovaný paralelismus až po Chinchilla scaling laws.

---

## 1. Backpropagation — matematické odvození

Backpropagation je algoritmický postup pro výpočet gradientu ztrátové funkce vzhledem k parametrům neuronové sítě pomocí **řetězového pravidla**. Pro síť s vrstvami `L_1, L_2, ..., L_n` a ztrátou `J` platí pro váhy `W^(l)`:

```
∂J/∂W^(l) = ∂J/∂a^(L) · ∂a^(L)/∂z^(L) · ∂z^(L)/∂a^(L−1) · ... · ∂z^(l)/∂W^(l)
```

kde `z^(l) = W^(l) a^(l−1) + b^(l)` a `a^(l) = σ(z^(l))`. Rekurzivní vztahy:

```
δ^(L) = ∇_a J ⊙ σ'(z^(L))                    # chyba výstupní vrstvy
δ^(l) = ((W^(l+1))^T δ^(l+1)) ⊙ σ'(z^(l))    # zpětná propagace
∂J/∂W^(l) = δ^(l) (a^(l−1))^T
∂J/∂b^(l) = δ^(l)
```

### Autograd v PyTorch a JAX

PyTorch používá **reverse-mode automatic differentiation**. Při forward passu se buduje **dynamický DAG**, kde listy jsou vstupní tenzory s `requires_grad=True`. Každá operace zaznamená atribut `.grad_fn`. Volání `loss.backward()` projde graf a aplikuje řetězové pravidlo.

```python
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # tensor([2., 4.])
```

JAX naopak používá **funkcionální transformace** (`jax.grad`, `jax.jit`, `jax.vmap`), které tracují funkci do jaxpr a poté provádějí XLA kompilaci.

Reverse-mode je preferován, protože pro `f: ℝ^n → ℝ` (typický případ ML, kde n je počet parametrů a výstup je skalár ztráty) je výpočetní složitost `O(n)` namísto `O(n²)` u forward-mode.

---

## 2. Gradient descent — varianty

| Varianta | Velikost batche | Update | Charakteristika |
|----------|-----------------|--------|------------------|
| **Batch GD** | celý dataset (N) | `θ ← θ − η ∇J(θ; X, y)` | Stabilní, paměťově náročné, pomalé |
| **Stochastic GD** | 1 vzorek | `θ ← θ − η ∇J(θ; x_i, y_i)` | Vysoký šum, úniky z lokálního minima |
| **Mini-batch GD** | B = 32–8192 | `θ ← θ − η ∇J(θ; X_B, y_B)` | Kompromis — de facto standard |

V LLM trénování se používá mini-batch GD s **gradient accumulation** — efektivní batch size dosahuje 4 M tokenů (Llama 3 405B trénován na 16 M tokenů per batch).

---

## 3. Optimizery — srovnání

### SGD with momentum

```
v_t = μ v_{t-1} + ∇J(θ_{t-1})
θ_t = θ_{t-1} − η v_t
```

Momentum (typicky `μ = 0.9`) urychluje pohyb v konzistentních směrech a tlumí oscilace.

### AdaGrad (Duchi 2011)

```
G_t = G_{t-1} + g_t²
θ_t = θ_{t-1} − η g_t / (√G_t + ε)
```

Akumulace všech čtverců gradientů → learning rate monotonicky klesá → problém pro dlouhé trénování.

### RMSProp (Hinton 2012)

```
v_t = ρ v_{t-1} + (1−ρ) g_t²
θ_t = θ_{t-1} − η g_t / (√v_t + ε)
```

Exponenciální průměr řeší problém AdaGradu.

### Adam (Kingma & Ba, 2014)

```
m_t = β_1 m_{t-1} + (1−β_1) g_t            # 1. moment
v_t = β_2 v_{t-1} + (1−β_2) g_t²           # 2. moment
m̂_t = m_t / (1 − β_1^t)                    # bias correction
v̂_t = v_t / (1 − β_2^t)
θ_t = θ_{t-1} − η m̂_t / (√v̂_t + ε)
```

Bias correction kompenzuje zkreslení k nule v prvních krocích. Defaultní hyperparametry: `β_1=0.9, β_2=0.999, ε=1e-8`.

### AdamW (Loshchilov 2017)

Decoupled weight decay — oddělení L2 regularizace od adaptivního scaling:

```
θ_t = θ_{t-1} − η (m̂_t / (√v̂_t + ε) + λ θ_{t-1})
```

**De facto standard pro LLM pretraining** (GPT-3, Llama, Mistral).

### Lion (Chen et al. 2023, Google Brain)

Objeven evolučním vyhledáváním. Používá pouze 1. moment a sign:

```
c_t = β_1 m_{t-1} + (1−β_1) g_t
θ_t = θ_{t-1} − η (sign(c_t) + λ θ_{t-1})
m_t = β_2 m_{t-1} + (1−β_2) g_t
```

**Úspora paměti ~33 %** (jen 1. moment), LR musí být 3–10× menší než u AdamW.

### Sophia (Liu et al. 2023, Stanford)

Druhý řád s diagonálním Hessianem:

```
m_t = β_1 m_{t-1} + (1−β_1) g_t
h_t = β_2 h_{t-1} + (1−β_2) ĥ_t              # Hutchinson estimate diag(H)
θ_t = θ_{t-1} − η · clip(m_t / max(h_t, ε), ρ)
```

**2× rychlejší než Adam** v počtu kroků na GPT modelech 125 M – 1.5 B. Hessian estimate jen každých ~10 kroků.

### Muon (Keller Jordan, 2024)

**MomentUm Orthogonalized by Newton-Schulz** — pro 2D parametry skrytých vrstev:

```python
# pseudokód
M_t = β M_{t-1} + g_t                # momentum
O_t = newton_schulz5(M_t)            # ortogonalizace
W_t = W_{t-1} − η O_t                # update
```

Newton-Schulz iterace nahrazuje SVD `USV^T → UV^T` pomocí kvintického polynomu (5 iterací) s koeficienty (3.4445, −4.7750, 2.0315).

**Výsledky 2025:** Moonlight 16 B MoE (Kimi.ai) trénován na 5,7 T tokenech. Muon dosahuje **~2× compute efficiency vs. AdamW**. Trénování za $500K s AdamW = $260K s Muonem.

### Srovnávací tabulka

| Optimizer | Paměť | Rychlost | LR rozsah | Použití |
|-----------|-------|----------|-----------|---------|
| SGD+momentum | 1× | nízká | 0.01–0.1 | CV, malé sítě |
| Adam/AdamW | 2× | střední | 1e-4 až 3e-4 | LLM standard |
| Lion | 1× | vysoká | 3e-5 až 1e-4 | velké batche, OOM |
| Sophia | 2× + Hessian | 2× rychlejší | 1e-4 | LLM pretraining |
| Muon | ~1× | 1.35–2× | hidden vrstvy | LLM 2025+ |

---

## 4. Learning rate schedules

### Cosine annealing (Loshchilov & Hutter 2017)

```
η_t = η_min + 0.5 (η_max − η_min)(1 + cos(π t/T))
```

Standard pro Transformers — postupný pokles s mírnou rychlostí na začátku/konci.

### Warmup

Lineární růst `η = η_max · t/T_warmup` po prvních ~2000 krocích. Klíčové pro velké modely — na začátku jsou Adam momenty nestabilní (malý vzorek pro odhad variance).

### WSD (Warmup-Stable-Decay)

Warmup → konstantní plateau → ostrá rychlá decay. Lepší pro continual pretraining; výzkum 2024–2025 ukazuje, že může překonat cosine schedule pro compute-optimal trénování.

### Cyclical LR (Smith)

Oscilace mezi `η_min` a `η_max` — méně používané v LLM.

Typické hodnoty pro Llama 70B: peak LR `1.5e-4`, cosine decay na 10 % peak, warmup 2000 kroků.

---

## 5. Regularizace

| Metoda | Popis | Použití v LLM |
|--------|-------|---------------|
| **L1** | `J' = J + λ Σ\|w_i\|` — sparsifikace | Vzácné |
| **L2 / Weight Decay** | `J' = J + λ/2 Σw_i²` | AdamW, λ=0.1 |
| **Dropout** | Náhodné maskování aktivací s p=0.1 | **Vypnutý v Llama, GPT-4** — velký dataset |
| **DropPath** | Náhodné vypouštění residuálních bloků | ViT, ConvNeXt |
| **Stochastic Depth** | Lineárně rostoucí drop podle hloubky | Vision modely standard |

V moderních LLM (Llama, GPT-4) je dropout prakticky vypnutý — velký dataset eliminuje overfitting.

---

## 6. Normalizace

### LayerNorm (Ba et al. 2016)

```
LN(x) = γ · (x − μ) / √(σ² + ε) + β
```

Stabilizuje aktivace nezávisle na batch size.

### RMSNorm (Zhang & Sennrich 2019)

```
RMSNorm(x) = γ · x / √(mean(x²) + ε)
```

Vynechává centrování → o ~25 % rychlejší výpočet. **Dominantní volba v moderních LLM** (Llama, Mistral, DeepSeek, Gemma).

### BatchNorm

Normalizuje přes batch dimenzi — závislost na batch size, problémy s distribuovaným tréninkem → v LLM **se nepoužívá**.

### Pre-LN vs Post-LN

- **Post-LN** (originál): `LN(x + Sublayer(x))` — lepší kvalita, ale nestabilní pro hluboké sítě
- **Pre-LN**: `x + Sublayer(LN(x))` — stabilní gradient, **standard 2025**
- **Peri-LN** (2025): kombinace — LN před i za blokem

**2026 recipe:** Pre-LN + RMSNorm + QK-Norm + BF16.

---

## 7. Inicializace vah

### Xavier/Glorot (2010)

Pro tanh/sigmoid aktivace:

```
W ~ U[−√(6/(n_in+n_out)), √(6/(n_in+n_out))]
```

### He/Kaiming (2015)

Pro ReLU:

```
W ~ N(0, 2/n_in)
```

V Transformerech se používá **scaled init**: `W ~ N(0, 0.02²)` pro embeddings a lineární vrstvy + scale residual outputs `× 1/√(2L)` (kde L je počet vrstev) — z GPT-2.

---

## 8. Vanishing/Exploding gradients

**Příčina:** při hluboké propagaci `∂J/∂x_1 = ∏ W_i^T · σ'(z_i)` — součin matic. Pokud `||W_i|| < 1` exponenciálně se zmenšuje, pokud `> 1` exploduje.

**Řešení:**

1. **Residual connections** (He 2015): `x_{l+1} = x_l + F(x_l)` — gradient přes skip má normu 1.0
2. **Gradient clipping**:
   ```
   if ||g|| > τ: g ← g · τ/||g||
   ```
   Typicky `τ = 1.0` pro LLM
3. **Normalizace** (LN/RMSNorm)
4. **ReLU/GELU/SiLU** — gradient ~1 v pozitivní oblasti
5. **Vhodná inicializace** (He/Xavier)

---

## 9. Loss functions pro LLM

### Cross-entropy (next-token prediction)

```
L = −(1/N) Σ_i log p_θ(x_i | x_{<i})
```

Pro slovník `V` a logity `z`: `p = softmax(z)`, `L = −log p[y_true]`.

### Label smoothing (Szegedy 2016)

```
y_smooth = (1−α) · one_hot + α/V
```

V LLM **se používá zřídka** — velký dataset to nepotřebuje.

### Perplexity jako evaluation metric

```
PPL = exp(cross_entropy) = exp(−(1/N) Σ log p_θ(x_i))
```

„Efektivní branching factor" — nižší = lepší. GPT-2 na WikiText: ~17, Llama 2 70B: ~3.3.

---

## 10. Pre-training paradigmata

### Causal Language Modeling (CLM) — GPT

Predikce dalšího tokenu zleva doprava: `p(x_t | x_1, ..., x_{t-1})`. Causal mask zajišťuje, že attention nevidí budoucí tokeny.

### Masked Language Modeling (MLM) — BERT

15 % tokenů náhodně maskováno (80 % [MASK], 10 % náhodný, 10 % původní), model predikuje maskované. Bidirectional kontext.

### Prefix LM (T5)

Obousměrná attention pro encoder, kauzální pro decoder.

### Multi-token prediction (DeepSeek-V3, Meta 2024)

Predikce N následujících tokenů paralelně — rychlejší trénování + speculative decoding.

**LLM standardně používají CLM** — CLM lépe škáluje a zero/few-shot generuje.

---

## 11. Trénování ve velkém měřítku — paralelismus

### Data Parallelism (DP)

Každý GPU má kopii modelu, různá data. Po backward all-reduce gradientů.

### Tensor Parallelism (TP) — Megatron-LM

Rozdělení individuálních vrstev (matrix mult). Pro `Y = XW`:
- Column-parallel: `W = [W_1, W_2]`, `Y = [XW_1, XW_2]`
- Row-parallel: kombinace s all-reduce

Vhodné v rámci jednoho serveru (NVLink).

### Pipeline Parallelism (PP)

Vrstvy rozděleny do „stages", micro-batche tečou pipelinou. Schémata: GPipe, 1F1B, interleaved.

### ZeRO (DeepSpeed) — 3 stupně

- **ZeRO-1** — shardování optimizer states
- **ZeRO-2** — + gradients
- **ZeRO-3** — + parameters

ZeRO-3 = každá GPU má jen 1/N parametrů, all-gather při forward/backward.

### FSDP (Fully Sharded Data Parallel) — PyTorch

Native PyTorch ekvivalent ZeRO-3. Benchmark 2025: **FSDP ~5× rychlejší per iteration** v rozsahu 100 M – 1 B; DeepSpeed lepší pro 70 B+.

### 3D Parallelism

Kombinace TP × PP × DP — běžná pro modely 100 B+. Llama 405B: TP=8, PP=16, DP=zbytek.

### Expert Parallelism (EP) pro MoE

Každý GPU drží podmnožinu expertů. DeepSeek-V3 (671 B / 37 B aktivních) — cross-node EP s overlap computation/communication.

---

## 12. Mixed precision training

### FP16 (half precision)

Range: ~6e-5 až 6.5e4 — úzký dynamic range. Standard recipe (Micikevicius 2017):
- Master weights v FP32
- Forward/backward v FP16
- **Gradient scaling**: `loss_scaled = loss × S` (typicky S=2^16)
- Unscale před optimizer step, dynamic loss scaling

### BF16 (brain float)

8 exponent bits (jako FP32), 7 mantissa — **stejný dynamic range jako FP32**. Není potřeba loss scaling. **Standard pro LLM trénink** (Hopper, Ampere, Blackwell).

### FP8 (E4M3, E5M2) — NVIDIA Hopper/Blackwell

Per-tensor nebo per-block scaling. **Transformer Engine** automatizuje:
- E4M3 pro forward (více precision)
- E5M2 pro backward (větší range pro gradienty)

**DeepSeek-V3 (2024)** — první open-source LLM trénovaný v FP8:
- Tile-wise 1×128 quantization aktivací
- Block-wise 128×128 quantization vah
- Embedding/output/attention v BF16/FP32
- Relativní chyba < 0.25 % vs BF16 baseline
- Komunikace mezi GPU 50 % redukce

### FP4 / NVFP4 — Blackwell (2025)

Mikro-scaling formáty (MXFP8, NVFP4). Další 2× zrychlení vs FP8. Validováno pro inference, training výzkum.

---

## 13. Curriculum learning, data quality, deduplication

**Kvalita dat > kvantita** — empirické pravidlo 2024–2026. Falcon RefinedWeb prokázal, že čistá web data překonají curated The Pile.

### Deduplication

- **Exact dedup** — suffix arrays, hledání >50 po sobě jdoucích tokenů
- **MinHash + LSH** — aproximace Jaccard similarity, klíčové pro trillion-token corpora
- **Semantic dedup** — embeddings + clustering (Llama 3 — RoBERTa embeddings)

Studie ukazují, že duplicity způsobují **memorizaci** a snížení generalizace — dedup zlepšuje zero-shot benchmarky o 2–5 bodů.

### Data quality filtering

1. URL/heuristic filtering (jazyk, délka, junk text)
2. Quality classifiers (FastText, BERT-based)
3. NSFW/PII filtering
4. Perplexity-based filtering

### Curriculum learning v LLM

- **Llama 3 mix:** 50 % general, 25 % math/reasoning, 17 % code, 8 % multilingual
- **Annealing** — na konci trénování upweightnout high-quality data
- **Long-context staging** — postupné zvětšování kontextu (Llama 3: 6 stupňů → 128K)

---

## 14. Scaling laws

### Kaplan et al. (OpenAI 2020)

```
L(N) = (N_c/N)^α_N    α_N ≈ 0.076
L(D) = (D_c/D)^α_D    α_D ≈ 0.095
L(C) = (C_c/C)^α_C    α_C ≈ 0.050
```

Závěr: při 10× compute → 5,5× model, 1,8× data. **Doporučili velké modely, méně dat**.

### Chinchilla (Hoffmann et al., DeepMind 2022)

Trénovali 400+ modelů 70 M – 16 B na 5–500 B tokenech. Compute-optimal:

```
N_opt ∝ C^0.5
D_opt ∝ C^0.5
```

**Optimální poměr: ~20 tokenů na parametr.** Např. 70 B model → 1,4 T tokenů.

### Praktické důsledky

Chinchilla 70B (1,4 T tok.) překonal Gopher 280B, GPT-3 175B, MT-NLG 530B. Kaplan modely byly **chronicky undertrained**.

### Současný trend (2025–2026)

Inference-aware scaling — modely se trénují **nad-Chinchilla** (Llama 3 8B na 15 T tokens = 1875 tok/par). Důvod: levnější inference dlouhodobě ospravedlňuje extra trénovací compute.

### DeepSeek-V3 case study (2024)

- 671 B / 37 B aktivních (MoE)
- Trénink: 14,8 T tokenů, 2,788 M H800 GPU-hodin
- FP8 mixed precision
- Náklady ~$5.5 M (vs. odhadované $100 M+ pro GPT-4)

### Muon scaling laws (2025)

Liu et al. (Moonshot/Kimi) ukázali, že **Muon má lepší scaling exponent než AdamW** — ~2× compute efficiency napříč velikostmi modelů, posun celé Pareto frontiers.

---

## 15. Klíčové závěry pro 2026

1. **Optimizer stack** — AdamW pro production, Muon pro frontier (2× rychlejší)
2. **Normalizace** — RMSNorm + Pre-LN + QK-Norm
3. **Precision** — BF16 default, FP8 pro frontier (DeepSeek), FP4 výzkum
4. **Parallelism** — 3D (TP×PP×DP) + EP pro MoE, FSDP pro mid-scale
5. **Data** — Quality > kvantita, dedup MinHash + suffix array, currikulum
6. **Scaling** — Chinchilla optimum překonán inference-aware overtraining (~100–200 tok/par)
7. **Schedule** — Cosine s warmup standard, WSD nastupuje pro continual

---

## Zdroje

- [Scaling Laws for Neural Language Models (Kaplan et al. 2020)](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal LLMs / Chinchilla](https://arxiv.org/abs/2203.15556)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Sophia: Scalable Stochastic Second-order Optimizer](https://arxiv.org/abs/2305.14342)
- [Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/)
- [Muon is Scalable for LLM Training](https://arxiv.org/html/2502.16982v1)
- [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1)
- [The Llama 3 Herd of Models](https://liweinlp.com/wp-content/uploads/2024/07/meta.pdf)
- [NVIDIA TransformerEngine FP8 / FP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [Megatron-LM (NVIDIA)](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed ZeRO Training](https://www.deepspeed.ai/training/)
- [LayerNorm vs RMSNorm](https://machinelearningmastery.com/layernorm-and-rms-norm-in-transformer-models/)
- [The RefinedWeb Dataset for Falcon](https://arxiv.org/pdf/2306.01116v1)
