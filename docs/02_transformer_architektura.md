# 02 — Architektura Transformer

> Hluboký technický průzkum architektury, která od roku 2017 dominuje moderní AI — od „Attention Is All You Need" po MoE modely roku 2026.

---

## 1. Historický kontext: „Attention Is All You Need" (2017)

V červnu 2017 publikovala skupina osmi výzkumníků z Google Brain a Google Research (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin) na konferenci NeurIPS práci **arXiv:1706.03762 — „Attention Is All You Need"**. Klíčová teze byla radikální: pro modelování sekvencí lze zcela zahodit rekurenci i konvoluci a vystačit si výhradně s mechanismem pozornosti (attention).

### Motivace a limity předchozích přístupů

Vaswani identifikoval tři zásadní problémy RNN/LSTM:

1. **Sekvenční výpočet** — RNN zpracovává tokeny jeden po druhém: skrytý stav `h_t = f(h_{t-1}, x_t)` nelze paralelizovat. Tréninkový čas roste lineárně s délkou.
2. **Délka cesty informace** — při vzdálenosti `n` tokenů musí gradient projít `O(n)` operacemi → vanishing/exploding gradient.
3. **GPU underutilization** — moderní hardware umí desítky teraflopů na maticových násobeních; RNN nedokáže tuto kapacitu využít.

Transformer odpověděl: každá pozice v sekvenci vidí všechny ostatní v **konstantním počtu kroků** `O(1)` (jeden self-attention layer) za cenu kvadratické výpočetní složitosti `O(n²·d)`. Trénink na WMT 2014 EN-DE zabral 3,5 dne na 8 GPU a dosáhl 28,4 BLEU; EN-FR 41,8 BLEU — state-of-the-art.

---

## 2. Pre-Transformer architektury: RNN, LSTM, GRU

### Vanilla RNN

```
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
y_t = W_hy · h_t + b_y
```

Kritický problém: backpropagation through time (BPTT) násobí Jacobian `∂h_t/∂h_{t-1} = diag(tanh') · W_hh^T` během každého kroku. Pokud spektrální poloměr `‖W_hh‖ < 1`, gradient zanikne; pokud `> 1`, exploduje. Ve vanilla RNN se gradient po cca 10–20 krocích stává numerickou nulou.

### LSTM (Hochreiter & Schmidhuber, 1997)

LSTM zavádí **cell state** `c_t` s gating mechanismy:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # input gate
ĉ_t = tanh(W_c · [h_{t-1}, x_t] + b_c) # candidate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ ĉ_t        # cell state update
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # output gate
h_t = o_t ⊙ tanh(c_t)
```

Klíčové: aditivní cesta `c_t = f_t ⊙ c_{t-1} + ...` umožňuje gradientu téct nepřerušeně.

### GRU (Cho et al., 2014)

Zjednodušení LSTM: spojuje forget + input gate do **update gate**, eliminuje cell state:

```
z_t = σ(W_z · [h_{t-1}, x_t])           # update gate
r_t = σ(W_r · [h_{t-1}, x_t])           # reset gate
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

Méně parametrů než LSTM, srovnatelný výkon. Stále **inherentně sekvenční**.

---

## 3. Self-attention — srdce Transformeru

### Query, Key, Value abstrakce

Pro vstup `X ∈ ℝ^{n×d_model}` (n tokenů, d_model = 512):

```
Q = X · W_Q   ∈ ℝ^{n×d_k}
K = X · W_K   ∈ ℝ^{n×d_k}
V = X · W_V   ∈ ℝ^{n×d_v}
```

V originálním paperu `d_k = d_v = 64`.

### Scaled Dot-Product Attention

Centrální vzorec celé éry deep learningu:

```
                      ⎛  Q · K^T  ⎞
Attention(Q, K, V) = softmax⎜─────────⎟ · V
                      ⎝   √d_k    ⎠
```

**Proč scaling √d_k?** Pokud `q, k ~ N(0, 1)`, pak `q·k = Σ q_i·k_i` má rozptyl `d_k`. Pro `d_k = 64` je směrodatná odchylka 8 — softmax by se saturoval do téměř one-hot distribuce a gradient by zmizel. Dělení `√d_k` udržuje varianci na 1.

### PyTorch implementace

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn
```

### Diagram toku dat

```
   Vstup X ∈ ℝ^(n×d)
         │
   ┌─────┼─────┐
  W_Q   W_K   W_V          ← lineární projekce
   │     │     │
   Q     K     V
   │     │     │
   └──┬──┘     │
      │        │
   Q·K^T       │           ← n×n matice skóre
   ÷ √d_k      │           ← scaling
   softmax     │           ← n×n attention weights
      │        │
      └──┬─────┘
         │
      A · V                ← vážený součet
         │
       výstup ∈ ℝ^(n×d_v)
```

---

## 4. Multi-Head Attention

Místo jediné sady `(W_Q, W_K, W_V)` použijeme **h paralelních hlav** (originál: h = 8). Každá hlava operuje v menším podprostoru `d_k = d_model/h = 64`:

```
head_i = Attention(X·W_Q^i, X·W_K^i, X·W_V^i)
MultiHead(X) = Concat(head_1, ..., head_h) · W_O
```

### Proč více hlav?

- **Reprezentační diverzita** — každá hlava se může specializovat (Clark et al. 2019: hlavy v BERT specializují např. na „atribut k subjektu", „objekt k předložce")
- **Robustnost** — kdyby jedna hlava selhala, ostatní pokrývají
- **Stejná výpočetní složitost** — `h · O(n²·d_k) = O(n²·d_model)`

### PyTorch implementace

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, n, _ = x.shape
        Q = self.W_Q(x).view(B, n, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, n, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, n, self.n_heads, self.d_k).transpose(1, 2)
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, n, -1)
        return self.W_O(out)
```

### Evoluce: MHA → MQA → GQA → MLA (2019–2026)

- **MHA** (originál): h hlav má vlastní K, V — KV-cache obrovský
- **MQA** (Multi-Query Attention, Shazeer 2019): všechny query hlavy sdílejí jediný K a V — KV-cache `h×` menší, ale ztráta kvality
- **GQA** (Grouped-Query Attention, Ainslie 2023): kompromis — query hlavy ve skupinách `G < h` sdílejí K, V (LLaMA 2-70B: h=64, G=8)
- **MLA** (Multi-head Latent Attention, DeepSeek V2/V3 2024–25): K a V se před cachováním komprimují do nízko-dimenzionálního latentního prostoru, při čtení se promítnou zpět

---

## 5. Positional Encoding

Self-attention je **permutačně ekvivariantní** — bez explicitní informace o pořadí by transformer nedokázal rozlišit „pes kouše muže" od „muž kouše psa".

### Sinusoidal (originál Vaswani 2017)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Vlnové délky tvoří geometrickou posloupnost `2π → 10000·2π`. Klíčová matematická vlastnost: pro libovolný offset `k` existuje **lineární transformace** `M_k`, taková že `PE_{pos+k} = M_k · PE_pos` — model se naučí relativní vztahy.

### Learned Positional Embeddings (BERT, GPT-2)

Místo fixního vzorce se naučí matice `E_pos ∈ ℝ^{n_max × d_model}`. Nevýhoda: nelze extrapolovat za `n_max`.

### RoPE — Rotary Position Embedding (Su et al. 2021)

**Dominantní metoda v moderních LLM** (LLaMA 1–3, Mistral, Gemma, Qwen, DeepSeek). Místo přičítání PE k embeddingu **rotuje** Q a K vektory v 2D podprostorech podle pozice. Pro pozici `m` a 2D pár `(x_1, x_2)`:

```
⎛ x_1' ⎞   ⎛ cos(mθ)  −sin(mθ) ⎞ ⎛ x_1 ⎞
⎜      ⎟ = ⎜                    ⎟ ⎜     ⎟
⎝ x_2' ⎠   ⎝ sin(mθ)   cos(mθ) ⎠ ⎝ x_2 ⎠
```

s frekvencemi `θ_i = 10000^(−2i/d)`. Klíčová vlastnost: dot-produkt `(R_m · q)·(R_n · k) = q^T · R_{n−m} · k` závisí **jen na rozdílu pozic**. RoPE je tedy přirozeně **relativní**, parameter-free, a pomocí YaRN/NTK-scaling lze extrapolovat na 128K–1M kontexty.

### ALiBi — Attention with Linear Biases (Press et al. 2022)

Žádné PE u embeddingu, místo toho se k attention skóre přičte **lineární bias úměrný vzdálenosti**:

```
softmax(q_i · K^T + m · [−(i−1), ..., −2, −1, 0])
```

Výhoda: extrémně dobrá extrapolace na delší sekvence.

### NoPE (2023–2026 trend)

Výzkum (Kazemnejad 2023) ukazuje, že causal masking sám o sobě v decoder-only modelech poskytuje implicitní pozicioální informaci. Modely jako SmolLM3 a Kimi Linear pozicioální encoding zcela vynechávají.

---

## 6. Feed-Forward Network (FFN)

Zatímco attention zajišťuje **mezi-token komunikaci**, FFN provádí **per-token transformaci** — každá pozice je zpracována nezávisle stejnou MLP.

### Originální FFN (Vaswani 2017)

```
FFN(x) = max(0, x·W_1 + b_1) · W_2 + b_2
```

`d_model = 512`, `d_ff = 2048` (4× expanze). Aktivace ReLU.

### Moderní varianty: GLU, SwiGLU, GeGLU

Shazeer (2020) v *„GLU Variants Improve Transformer"* navrhl **gated** FFN:

```
SwiGLU(x) = (Swish(x·W_1) ⊙ (x·W_3)) · W_2
GeGLU(x)  = (GELU(x·W_1)  ⊙ (x·W_3)) · W_2
```

kde `Swish(x) = x · σ(x)`, `⊙` je element-wise násobení. Aby zůstal stejný počet parametrů, používá se `d_ff = (8/3) · d_model` místo `4 · d_model`.

LLaMA, Mistral, PaLM, DeepSeek používají SwiGLU; Gemma používá GeGLU. Empiricky o ~1–2 % lepší perplexity.

### Role FFN

Geva et al. (2021) ukázali, že FFN funguje jako **key-value memory**: první matice `W_1` detekuje vzory ve vstupu (klíče), druhá matice `W_2` z paměti vrací distribuce nad vocabulary. Velká část faktických znalostí LLM žije právě v FFN — typicky 60–70 % parametrů modelu.

---

## 7. Residual Connections a Layer Normalization

### Reziduální spojení

```
output = LayerNorm(x + Sublayer(x))   # Post-Norm (originál)
```

Reziduum slouží jako „highway" pro gradient — bez něj by tréning hlubokých transformerů (12+ vrstev) zkolaboval.

### Post-Norm vs Pre-Norm

- **Post-Norm** (originál Transformer): `LayerNorm(x + Sublayer(x))` — narušuje identity-path, vyžaduje warmup
- **Pre-Norm** (od GPT-2): `x + Sublayer(LayerNorm(x))` — reziduální cesta zůstává netknutá, lze trénovat 100+ vrstev

```
Post-Norm:                          Pre-Norm:

   x ──┬──> Attn ──> Add ──> LN ──>     x ──┬─> LN ──> Attn ──> Add ──>
       │              ▲                     │                     ▲
       └──────────────┘                     └─────────────────────┘
```

### LayerNorm vs RMSNorm

**LayerNorm:**

```
LN(x) = γ · (x − μ) / √(σ² + ε) + β
```

**RMSNorm** (Zhang & Sennrich, 2019):

```
RMSNorm(x) = γ · x / √(mean(x²) + ε)
```

**Vynechává centrování** — empiricky stejná kvalita, ~7–10 % rychlejší díky méně FLOPům.

LLaMA 1–3, Mistral, Gemma, DeepSeek, Qwen — všechny používají Pre-RMSNorm. OLMo 2 a Gemma 3 experimentují s **hybrid Norm** nebo **QK-Norm** (RMSNorm na Q a K před RoPE).

---

## 8. Encoder-Decoder vs Decoder-only vs Encoder-only

### Encoder-only (BERT, RoBERTa, DeBERTa)

- **Architektura:** stack N encoderů, každý s bidirectional self-attention
- **Cíl tréninku:** Masked Language Modeling (MLM) — 15 % tokenů zamaskováno
- **Použití:** klasifikace, NER, extractive QA, embedding pro retrieval

### Decoder-only (GPT-1/2/3/4, LLaMA, Claude, Gemini, DeepSeek, Mistral)

- **Architektura:** stack N decoderů s **causal masked** self-attention
- **Cíl tréninku:** Next-Token Prediction
- **Použití:** generování textu, chat, in-context learning, většina moderních LLM
- **Důvod dominance:** jednoduchost, škálovatelnost (Chinchilla), univerzálnost (zero/few-shot)

### Encoder-Decoder (T5, BART, Whisper)

- **Architektura:** encoder zpracuje vstup bidirectionally; decoder generuje výstup autoregresivně, každá vrstva má **cross-attention** kde Q jsou z decoderu, K, V z encoder výstupu
- **Cíl tréninku:** span corruption (T5), denoising (BART)
- **Použití:** překlad, sumarizace, ASR (Whisper)

### Diagram tří paradigmat

```
ENCODER-ONLY (BERT)        DECODER-ONLY (GPT)        ENCODER-DECODER (T5)

input ──> [Enc]×N          input ──> [Dec]×N          input ──> [Enc]×N ──┐
            │                          │                                  │
            └─> classifier             └─> next token            tgt ─> [Dec]×N
                                                                        │ ↑
            bidirectional             causal mask                cross-attention
            (vidí všechno)            (vidí jen <=t)             (Q dec, K,V enc)
```

V 2025–26 dominuje decoder-only kvůli inference efektivitě (jediný KV-cache, žádné cross-attention).

---

## 9. Causal Masking

Pro autoregresivní generování musí token na pozici `i` vidět **jen tokeny `j ≤ i`**. Implementace: před softmax přičteme **lower-triangular masku** kde nepovolené pozice mají `−∞`:

```
         j=0  j=1  j=2  j=3
i=0    [  0   -∞   -∞   -∞ ]
i=1    [  0    0   -∞   -∞ ]
i=2    [  0    0    0   -∞ ]
i=3    [  0    0    0    0 ]
```

```python
def causal_mask(n, device):
    mask = torch.tril(torch.ones(n, n, device=device))
    return mask.unsqueeze(0).unsqueeze(0)

scores = Q @ K.transpose(-2,-1) / math.sqrt(d_k)
scores = scores.masked_fill(causal_mask(n, x.device) == 0, float('-inf'))
attn = F.softmax(scores, dim=-1)
```

Při **inferenci** používáme **KV-cache**: spočítané K, V pro tokeny `1:t−1` se uloží, pro nový token se spočítá jen jeho Q. Složitost generování `n` tokenů je `O(n²·d)` místo `O(n³·d)`.

---

## 10. FlashAttention v1 / v2 / v3

### Problém standardního attention

Naivní implementace `softmax(QK^T/√d)V` materializuje `n×n` attention matici v HBM. Pro `n=8192, h=32, batch=4` je to `4·32·8192² · 4B = 32 GB` — nevejde se ani na H100.

Klíčový postřeh: většinu času attention nestráví počítáním FLOPů, ale **přesouváním paměti** mezi HBM a SRAM.

### FlashAttention v1 (Dao et al. 2022)

1. **Tiling** — Q, K, V se rozdělí na bloky, které se vejdou do SRAM
2. **Online softmax** — softmax lze počítat inkrementálně, sleduje `m_i = max(scores)` a `ℓ_i = sum(exp(scores − m_i))`
3. **Recomputation v backward** — místo ukládání attention matice ji při backprop přepočítáme

Výsledek: `O(n)` paměť místo `O(n²)`, **2–4× rychlejší** wall-clock na A100.

### FlashAttention v2 (Dao 2023)

- Lepší work partitioning podle Q-bloků
- Sequence-level paralelismus
- Causal mask optimalizace

Cca 2× rychlejší než FA1, ~50–73 % SM utilization.

### FlashAttention v3 (Shah et al. 2024) — Hopper

1. **WGMMA** — asynchronní matmul instrukce
2. **TMA** (Tensor Memory Accelerator) — async global ↔ shared memory
3. **Asynchronní pipelining** — překryv GEMM ↔ softmax (ping-pong scheduling)
4. **FP8 + incoherent processing** — Hadamard transform pro outliers, 2,6× menší kvantizační chyba

Výsledky na H100:
- FP16: ~740 TFLOPS (75 % peak), **1,5–2× rychlejší než FA2**
- FP8: ~1,2 PFLOPS

V 2026 je FlashAttention součástí PyTorch jako `torch.nn.functional.scaled_dot_product_attention`.

---

## 11. Long Context: Sliding Window, Sparse Attention, Mamba/SSM

### Sliding Window Attention (SWA)

Každý token vidí jen **lokální okno** velikosti `w`:

```
  full attention          sliding window (w=3)
  ████████              ███░░░░░
  ████████              ████░░░░
  ████████              █████░░░
  ████████              ░█████░░
  ████████              ░░█████░
  ████████              ░░░█████
```

Složitost klesne na `O(n·w)`. Použito v **Longformer** (2020), **Mistral 7B** (2023, w=4096), **Gemma 2/3**. Po `k` vrstvách receptive field naroste na `k·w`.

### Sparse / Dilated Attention

- **Sparse Transformer** (Child 2019) — `O(n·√n)`
- **Longformer** — SWA + global attention na vybrané tokeny
- **BigBird** — random + window + global, Turing-complete
- **Reformer** — locality-sensitive hashing, `O(n log n)`

### Mamba / Selective SSM (Gu & Dao 2023)

Stavový prostor (SSM) modeluje sekvenci jako lineární dynamický systém:

```
h_t = A · h_{t-1} + B · x_t
y_t = C · h_t
```

Klíčová inovace Mamby: **selektivní (input-dependent) parametry** `A_t, B_t, C_t`.

- **Trénink:** paralelní pomocí parallel scan, `O(n)`
- **Inference:** rekurentní, `O(1)` per token
- **Paměť:** konstantní stav 2–4 GB nezávisle na délce kontextu

**Mamba-3** (2026) zlepšuje retrieval. **Hybridní modely** (Jamba, Zamba, Qwen3-Next) kombinují SSM vrstvy s občasnými attention vrstvami.

---

## 12. Mixture of Experts (MoE)

V roce 2026 je MoE **de facto standard** pro frontier modely: DeepSeek V3/R1, Mixtral, Kimi K2, Qwen3-MoE, GLM-4.5, Mistral Large 3, GPT-5.

### Princip: sparse activation

Místo aktivace všech parametrů máme **N expertů** a pro každý token aktivujeme jen **k z nich** (typicky k=2 nebo k=8).

```
   token x
     │
   ┌─┴─┐  router
   │ R │  scores = x · W_r ∈ ℝ^N
   └─┬─┘  vyber top-k indexů
     │
   ┌─┼─┬─┬─┬─...─┐
   ▼ ▼ ▼ ▼ ▼     ▼
  E_1 E_2 ... E_N
   │   │         │
   └───┴─weighted sum──> výstup
```

### Router

```python
class TopKRouter(nn.Module):
    def __init__(self, d_model, n_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True)
        return topk_probs, topk_idx
```

### Klíčové komponenty

1. **Load Balancing Loss** — bez něj router degeneruje:
   ```
   L_balance = N · Σ_i f_i · P_i
   ```
2. **Capacity Factor** — každý expert má kapacitu, tokeny nad limit „dropnou"
3. **Shared Experts (DeepSeek)** — vždy aktivní expert pro obecné rysy + top-k routovaní specialisté

### DeepSeek V3 jako příklad

- Total parameters: 671 B
- Active parameters per token: 37 B (5,5 %)
- Experts per MoE layer: 256 routovaných + 1 sdílený
- Top-k: 8 routovaných + 1 sdílený = 9 aktivních
- Vrstev: 61 (první 3 dense, ostatní MoE)

### Trade-offs MoE

**Plus:**
- Vyšší kapacita bez růstu inference compute
- Specializace expertů (domény, jazyky)

**Minus:**
- **Paměť** — všechny experty musí být v GPU (267 GB pro V3 v FP8)
- **All-to-all komunikace** — bottleneck mezi GPU
- **Tréninková nestabilita** — load imbalance, expert collapse

---

## 13. Stav v roce 2026

| Komponent | 2017 (originál) | 2026 (state of the art) |
|-----------|-----------------|-------------------------|
| Pozice | Sinusoidal absolute | RoPE / Partial RoPE / NoPE |
| Attention | MHA O(n²) | GQA / MLA + FlashAttention 3 + SWA |
| FFN | ReLU, dense | SwiGLU + MoE (256 expertů) |
| Norm | Post-LayerNorm | Pre-RMSNorm + QK-Norm |
| Precision | FP32 | FP8 / BF16 trénink |
| Kontext | 512 tokenů | 1 M+ (RoPE scaling, SSM hybridy) |
| Velikost | 65 M | 1 T+ (sparse) |

Architektura Transformer se za 9 let překvapivě **kvalitativně nezměnila** — pořád je to stack N decoder bloků s self-attention + FFN + reziduálními spojeními. Změnil se každý jednotlivý komponent.

**Open questions 2026:** úplně nahradí SSM/Mamba attention? Bude RWKV / linear attention dominovat na edge? Konečně narazíme na škálovací zeď?

---

## Zdroje

- [Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [The Transformer Family v2 — Lilian Weng](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
- [Karpathy nanoGPT GitHub](https://github.com/karpathy/nanoGPT)
- [The Big LLM Architecture Comparison — Sebastian Raschka](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
- [FlashAttention-3 PyTorch Blog](https://pytorch.org/blog/flashattention-3/)
- [FlashAttention-3 Paper (arXiv:2407.08608)](https://arxiv.org/abs/2407.08608)
- [Mamba: Linear-Time Sequence Modeling (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752)
- [Mixture of Experts Explained — HuggingFace](https://huggingface.co/blog/moe)
- [Rotary Embeddings: A Relative Revolution — EleutherAI](https://blog.eleuther.ai/rotary-embeddings/)
- [GLU Variants Improve Transformer (arXiv:2002.05202)](https://arxiv.org/pdf/2002.05202)
- [GQA Paper (arXiv:2305.13245)](https://arxiv.org/pdf/2305.13245)
