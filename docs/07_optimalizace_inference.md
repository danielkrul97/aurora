# 07 — Optimalizace inference LLM

> Hluboký technický průzkum optimalizace běhu LLM v roce 2026 — od KV cache a Flash Attention přes vLLM a speculative decoding po Cerebras WSE-3 a Apple MLX.

---

## 1. Co je inference a fáze běhu

LLM inference je **autoregresivní** proces generování textu, kde model predikuje další token podmíněně na všech předchozích: `P(x_t | x_1, ..., x_{t-1})`. Proces se rozpadá do dvou kvalitativně odlišných fází:

**Prefill (compute-bound):** Zpracování celého vstupního promptu (kontextu) v jediném dopředném průchodu. Útok je masivně paralelní — všechny tokeny se zpracovávají najednou. Výpočetní složitost je `O(N²·d)` pro attention. Limitující faktor: **FLOPs** (tensor cores). Zde se generuje první výstupní token a sestavuje **KV cache**.

**Decode (memory-bound):** Generování výstupu token po tokenu. Každý nový token vyžaduje načíst kompletní váhy modelu z HBM do SRAM. Limitující faktor: **paměťová propustnost**. Pro Llama 2 70B v FP16 (140 GB) na H100 (3.35 TB/s HBM) teoretický strop činí ~24 tok/s při batch size 1.

Tato dichotomie zásadně určuje optimalizační strategie: prefill se škáluje výpočtem (tensor parallelism), decode pamětí (kvantizace, KV cache komprese, batchování).

---

## 2. Latency vs Throughput, klíčové metriky

| Metrika | Definice |
|---------|----------|
| **TTFT** | Time To First Token — čas od submitu do prvního tokenu |
| **TPOT** | Time Per Output Token — `(E2E_latency − TTFT) / (output_tokens − 1)` |
| **TPS** | Tokens Per Second — `1 / TPOT` (single user) |
| **Throughput** | suma napříč souběžnými requesty |
| **Goodput** | useful tokens/s — jen tokeny splňující SLO |

Celková latence: `Latency = TTFT + TPOT × (output_tokens − 1)`.

**Klíčový tradeoff:** Zvyšování batch size lineárně zvyšuje throughput (až do compute saturace), ale prodlužuje TPOT i TTFT. Produkční systémy proto cílí dva odlišné SLO — typicky TTFT < 500 ms, TPOT < 50 ms.

---

## 3. KV Cache — paměťová achilova pata

Aby se nepočítaly attention klíče a hodnoty pro celou historii znovu, ukládají se v **KV cache:**

```
KV_size = 2 × batch × seq_len × n_layers × n_kv_heads × head_dim × bytes_per_param
```

Faktor 2 = K + V. Příklad Llama 2 7B (32 layerů, 32 head, 128 head_dim, FP16, seq=4096, batch=1):
`2 × 1 × 4096 × 32 × 32 × 128 × 2 ≈ 2 GB`

Pro Llama 2 70B (BF16, GQA s 8 KV head) při seq=4096 a batch=8: `2 × 80 × 8 × 128 × 4096 × 8 × 2 ≈ 10 GB` — k tomu váhy 140 GB. Při dlouhém kontextu KV cache **přerůstá váhy modelu**. Optimalizace: GQA/MQA, kvantizace cache (FP8/INT4), PagedAttention, prefix caching.

---

## 4. PagedAttention a vLLM

Klasické systémy alokovaly KV cache jako jeden souvislý buffer pro maximální plánovanou délku → **fragmentace 60–80 %**. PagedAttention (Kwon et al., SOSP 2023) přejímá koncept **virtuální paměti OS** — KV cache se rozseká na fixní bloky (typicky 16 tokenů), které se mohou v HBM nacházet **non-contiguously**. Block table mapuje logické na fyzické bloky.

Důsledky:
- Plýtvání < 4 % (téměř-nulové vnitřní fragmentace)
- **Copy-on-write** pro paralelní samplování / beam search
- **Prefix sharing** mezi requesty (system prompty, few-shot examples)
- Throughput **2–4× vs FasterTransformer/Orca**

vLLM 2026 dosahuje ~12 500 tok/s pro Llama 3.1 8B BF16 na single H100; SGLang/LMDeploy ~16 200 tok/s (29 % rozdíl, přes **RadixAttention** — radix-tree pro prefix cache).

---

## 5. Continuous batching (iteration-level scheduling)

Static batching čeká na nejpomalejší sekvenci → GPU utilizace 30–60 %. **Continuous batching** (Yu et al., Orca, OSDI 2022) plánuje na úrovni **každé iterace decoderu**. Když sekvence skončí (EOS nebo max_tokens), její slot se okamžitě naplní novým requestem.

Efekty:
- GPU utilizace **80–95 %** v reálném provozu
- Throughput **2–4×** oproti static batching
- Tail latence se **zužuje**

vLLM kombinuje continuous batching + PagedAttention + chunked prefill (rozsekání prefillu na menší chunky aby se nezablokovaly decode iterace) → 3–5× vs naivní PyTorch loop.

---

## 6. Speculative decoding

Myšlenka: **draft model** (malý, rychlý) navrhne `k` kandidátních tokenů; **target model** (velký, kvalitní) je v jediném forward passu **verifikuje paralelně**. Pokud target přijme prvních `m ≤ k` tokenů, ušetříme `m−1` autoregresivních kroků. Výstup je **matematicky identický** s vanilla decoderem (lossless).

| Metoda | Princip | Speedup | Poznámka |
|--------|---------|---------|----------|
| **Vanilla Spec.** | externí draft model | 2–3× | nutná shoda tokenizéru |
| **Medusa** | extra „Medusa heads" predikují t+1, t+2, t+3 | 2–3× | žádný draft model |
| **EAGLE-2** | feature-level autoregrese, dynamic tree | 3–4× | trénink draft headu |
| **EAGLE-3** | training-time test, fúze low/mid/high features | **3.0–6.5×** | 20–40 % nad EAGLE-2 |
| **Lookahead** | n-gram from past, žádný trénink | 1.5–2× | jen pro vzorce |

EAGLE je 1,6× rychlejší než Medusa (13B). V produkčních frameworcích (TensorRT-LLM, vLLM, SGLang) podporují EAGLE-3 v 2026.

---

## 7. Kvantizace — datové typy

| Formát | Bity | Reprezentace | Hardware | Použití |
|--------|------|--------------|----------|---------|
| FP32 | 32 | 1+8+23 | univerzální | training |
| BF16 | 16 | 1+8+7 | A100+, TPU | training/inference |
| FP16 | 16 | 1+5+10 | univerzální | inference |
| FP8 (e4m3) | 8 | 1+4+3 | H100, B200 | inference, FlashAttn-3 |
| INT8 | 8 | celé | univerzální | weight+act quant |
| INT4 | 4 | celé | weight-only | GPTQ, AWQ |
| **MXFP4** | 4 | block 32 + E8M0 scale | OCP standard | training/inference |
| **NVFP4** | 4 | block 16 + E4M3 scale | Blackwell B200 | inference |

**Weight-only** (W4A16) — kvantujeme jen váhy, aktivace v FP16 — typické pro decode (memory-bound).
**Weight+Activation** (W8A8) — obojí kvantováno — efektivní pro prefill (compute-bound).

NVFP4 vs MXFP4: NVFP4 používá 16-element bloky (vs 32) a E4M3 scale (vs E8M0) → jemnější granularita. **3,5× menší paměť vs FP16, 1,8× vs FP8**, ztráta < 1 %.

---

## 8. Kvantizační metody — algoritmy

| Metoda | Typ | Princip | Doba kalibrace | Kvalita W4 |
|--------|-----|---------|----------------|------------|
| **GPTQ** | weight-only | layer-wise OBQ, Hessian aproximace | hodiny | 90 % |
| **AWQ** | weight-only | aktivační scale-protect 1 % salient vah | minuty | **95 %** |
| **SmoothQuant** | W8A8 | migrace difficulty z aktivací do vah | rychlé | dobrá |
| **GGUF k-quants** | weight-only | block-wise s super-bloky | rychlé | 92 % |
| **GGUF i-quants + imatrix** | weight-only | importance matrix z reálných promptů | středně | nejlepší při <4-bit |
| **SpinQuant** | rotational | trénovaná Hadamard rotace + GPTQ | středně | excelent |
| **MR-GPTQ** | NVFP4 | mikroškálovaná GPTQ varianta | středně | 2,2× E2E na B200 |

Na 70B+ AWQ konzistentně překonává GPTQ. SmoothQuant je standard pro INT8 W8A8.

---

## 9. Pruning — řídké modely

| Metoda | Princip | Sparsity | Retraining | Speedup |
|--------|---------|----------|------------|---------|
| Magnitude pruning | jen \|w\| | unstrukt. | nutný | malý |
| **SparseGPT** | layer-wise OBS, druhý řád | 50–60 % | **žádný** | umírněný |
| **Wanda** | metric: \|w\| × ‖x‖₂ | 50 % 2:4 | žádný | **1,6×** lin layer |
| LLM-Pruner | strukt., gradient-based | 20–30 % | LoRA fine-tune | 2× |

---

## 10. Knowledge distillation

Teacher-student paradigm — malý student se učí napodobit distribuci velkého teacheru:

```
L = α · CE(student, hard_labels) + (1−α) · KL(student_soft || teacher_soft / T) · T²
```

V 2026 standard pro produkci: DeepSeek-Distill-Qwen, Phi-mini, Gemma 2B, **Anthropic Haiku** (distilace ze Sonnet/Opus). LIMA experiment ukázal, že 1000 vysoce kvalitních příkladů od teacher modelu stačí pro near-teacher výkon ve studentovi.

---

## 11. Mixture of Experts (MoE) inference

V 2026 frontier modely (DeepSeek-V3, Mixtral, GPT-OSS, Llama 4 Maverick) téměř výhradně MoE.

**Inference výzvy:**
- Velké celkové parametry (671 B DeepSeek-V3) ale jen ~37 B aktivních / token
- All-to-all komunikace mezi GPU (token routing)
- **Expert parallelism (EP)** — každý GPU drží podmnožinu expertů; tokeny se posílají kam patří
- Load imbalance — některé experty hot, jiné cold → DeepSeek používá **shared experts** (vždy aktivní) + **routed experts**

NVIDIA Blackwell NVL72 doménový switch + **DeepEP** (efficient expert-parallel comm library) → MoE běží **10× rychleji než dense ekvivalent při 1/10 nákladu na token**.

---

## 12. Flash Attention

Klasický attention: `O(N²)` paměti pro `QK^T` matrici → bottleneck na long-context. FlashAttention je **IO-aware** algoritmus — blokově rozsekané softmax-attention, který drží statistiky `(m, l)` pro online normalizaci a **nikdy nematerializuje plnou attention matici**.

| Verze | Rok | Technika | H100 BF16 | H100 FP8 |
|-------|-----|----------|-----------|----------|
| Flash-Attn-1 | 2022 | tiling, online softmax, recompute | — | — |
| Flash-Attn-2 | 2023 | méně non-matmul FLOPs, lepší WGMMA | 230 TFLOPs | — |
| **Flash-Attn-3** | 2024 | warp-specialization, async TMA, FP8 + incoherent processing | **740 TFLOPs (75 %)** | **~1,2 PFLOPs** |

I/O optimalita: `O(N·d)` HBM operací (asymptoticky důkazatelně optimální). FA-3 FP8 dosahuje **2,6× nižší numerické chyby** než baseline FP8 attention díky block-wise quant + incoherent processing (Hadamard rotace).

---

## 13. Tensor Parallelism (TP) a Pipeline Parallelism (PP)

**TP** — rozdělíme matice (Wq, Wk, Wv, Wo, FFN W1/W2) **přes hidden dimenzi**. Každý GPU drží shard; po každé linear layer all-reduce. Vyžaduje **NVLink/NVSwitch**. Praxe: TP=8 v rámci jednoho 8-GPU node.

**PP** — rozdělíme **vrstvy** mezi GPU. Komunikace jen mezi vrstvami, funguje přes pomalejší PCIe/Ethernet. Trpí **pipeline bubble** při malém batchi → micro-batching (GPipe, 1F1B).

**Kombinace 2026:** TP=8 v node + PP=N přes nodes + **EP** pro MoE + **DP** pro replikace.

NVLink 5 (Blackwell): 1,8 TB/s per GPU. NVL72 doména spojuje 72 B200 do single coherent domény — TP=72 reálné.

---

## 14. MQA a GQA — méně KV head

- **MHA** — každý query head má svůj K, V → KV cache je velká
- **MQA** — **1 K, 1 V** sdílené pro všechny query heady → 8–32× menší KV cache, ale ztráta kvality
- **GQA** — kompromis, `g` skupin query heads sdílí jeden KV pár

Llama 2 70B: 64 query head, 8 KV groups → 8× menší KV cache vs MHA. Llama 3 / Mistral / Qwen 2/3 / Gemma — všechny používají GQA jako standard 2025–2026.

---

## 15. Hardware pro inference 2026

| Akcelerátor | Mem | BW | FP4 dense | Cena |
|-------------|-----|-----|-----------|------|
| **NVIDIA H100 SXM** | 80 GB HBM3 | 3,35 TB/s | — (FP8: 4 PFLOPs) | $25–30k |
| **NVIDIA H200** | 141 GB HBM3e | 4,8 TB/s | — | $30–32k |
| **NVIDIA B200** | 192 GB HBM3e | 8,0 TB/s | **9 PFLOPs** | $35–40k |
| **NVIDIA B300** | 288 GB HBM3e | 8,0 TB/s | 15 PFLOPs | premium |
| **AMD MI300X** | 192 GB HBM3 | 5,3 TB/s | — | competitive |
| **TPU v5p / v6 (Trillium)** | 95 / 192 GB | ~2,7 TB/s | — | Google internal |
| **Cerebras WSE-3** | 44 GB SRAM on-die | „PB/s" interní | 125 PFLOPs | wafer-scale |
| **Groq LPU** | 230 MB SRAM/chip | 80 TB/s SRAM | — | scale-out |
| **AWS Trainium3** | 144 GB HBM3e | — | 2,52 PFLOPs FP8 | AWS native |

**Cerebras WSE-3** — celý wafer (46 255 mm²) jako 1 chip — 4 bilióny tranzistorů, 900 000 jader, **1000 tok/s na Llama 3.1 405B**. Independent Artificial Analysis bench: oss-gpt-120B na Cerebras ~3000 tok/s vs Groq ~493 tok/s (>6× rychlejší).

**Groq LPU** — deterministický scheduler, vše v SRAM (žádné HBM) — Llama 2 70B @ ~300 tok/s.

**AWS-Cerebras partnership** (březen 2026) — disaggregovaná architektura Trainium + WSE → 5× token capacity ve stejném footprintu.

---

## 16. Inference frameworky 2026

| Framework | Backend | Silné stránky | Slabé stránky |
|-----------|---------|---------------|---------------|
| **vLLM** | Python + CUDA | široký ekosystém, PagedAttention pioneer | mírně horší TPS než SGLang |
| **SGLang** | Python + C++ | **RadixAttention**, nejlepší pro multi-turn, RAG | méně modelů |
| **TensorRT-LLM** | C++/TRT | nejvyšší výkon na NV HW, FP4 native | NV-only |
| **LMDeploy (TurboMind)** | čisté C++ | nejnižší latence, kvantizovaná inference | menší community |
| **HF TGI** | Rust + Python | enterprise-friendly | nižší peak throughput |
| **llama.cpp** | C/C++ ggml | CPU/Apple Silicon/CUDA/Metal/Vulkan, GGUF | menší serving features |
| **MLC-LLM** | TVM | mobile, WebGPU | kompilace složitá |

**H100 srovnání 2026 (Llama 3.1 8B):**
- SGLang ~16 200 tok/s
- LMDeploy ~16 200 tok/s
- vLLM ~12 500 tok/s
- TensorRT-LLM dominuje na **B200**

---

## 17. Long context inference

**YaRN** (Peng et al., ICLR 2024) — rozšíření RoPE-based modelů na delší kontext:
- **NTK-by-parts interpolation** — nízké frekvence interpolovat, vysoké zachovat
- **Attention scaling** — přidá teplotu `1/√t` do logitů
- 10× méně tokenů a 2,5× méně steps než PI/PE
- Standard 2026: Qwen, DeepSeek, Llama, gpt-oss

**StreamingLLM / Attention Sinks** (Xiao et al., ICLR 2024) — keep-first-K + sliding window. Pozorování: prvních pár tokenů (BOS) přitahují velké attention skóry jako „sink". Řešení: zachovat sink tokens + recent W. Fluentnost přes **stovky promptů**, 22,2× speedup vs sliding-window-recompute baseline.

**Sliding Window Attention (SWA)** — Mistral 7B atenduje jen `W=4096` posledních hidden states per layer. Gemma 3 a GPT-OSS používají SWA + global attention v alternujících vrstvách.

**Ring Attention** — pro context >1M tokenů. Sequence se rozseká přes GPU, KV bloky cirkulují prstencově → memory-linear scaling. Použito Gemini 1.5 Pro a Llama 4 Scout (10M context).

---

## 18. Edge inference

**llama.cpp** zůstává králem CPU inference: AVX-512, NEON, ARM SVE, Vulkan, Metal, ROCm, CUDA. GGUF formát (k-quants Q4_K_M, IQ4_XS) + **imatrix** kvantizace.

**Apple MLX (M5 chip 2026)** — NeuralAccelerators v každém GPU jádře (dedikované matmul jednotky). MLX překonává Metal/llama.cpp na M5. **Foundation Models framework** ve Swift API pro on-device 3B model. **Ollama 0.19 přešel na MLX backend** na Apple Silicon.

**ONNX Runtime Mobile** — Android cross-platform standard. Pro iOS častěji **Core ML** (přes coremltools).

**On-device modely 2026:** Phi-4-mini, Gemma 3 4B, Qwen 3 4B, Apple Foundation 3B, Llama 4 Scout subset — vše distilované, INT4/MXFP4 kvantizované, běží reálně na iPhone 16 Pro / Pixel 9 / Snapdragon 8 Gen 4.

---

## 19. Cost optimization — prompt caching, batch API

**Prompt caching** — opakovaně používané prefixy cachují (50–95 % cena vs uncached).

| Provider | Cache discount | Min tokens | Storage fee |
|----------|----------------|------------|-------------|
| **Anthropic** | **90 %** off cached input | 1024 (Sonnet/Haiku), 2048 (Opus) | 25 % markup za zápis |
| **OpenAI** | 50 % off | 1024, automatické | žádné |
| **Google Gemini** | 75 % off | 4096–32 768 | $4,50/M tokens/hour |

**Batch API (50 % off, async <24h):**
- Anthropic Sonnet: $3/$15 → **$1,50/$7,50** per M tokens
- Anthropic Haiku: $1/$5 → **$0,50/$2,50**
- OpenAI all models 50 % off

**Kombinace caching + batch:** Anthropic uvádí až **95 % úspora**. Reálné případy: měsíční náklady $720 → $72 jen prompt cachingem v RAG/agent workflows.

---

## 20. Memory hierarchy a IO-awareness

```
Registry / SRAM (L1/L2 sdílené)  ≈ 256 KB / SM @ 19 TB/s   (H100)
HBM (DRAM-on-die)                = 80–288 GB @ 3–8 TB/s
NVLink (inter-GPU)               = 1,8 TB/s (NVL5)
PCIe Gen5                        = 128 GB/s
NIC / RDMA                       = 400 Gbps
```

Klíčová asymetrie: SRAM je **~1000× rychlejší** než HBM. Klasický attention dělá `Q·K^T` v HBM → memory bound. **Flash Attention** drží Q/K/V/intermediate v SRAM, HBM operace jen O(N·d) místo O(N²). Podobný princip pohání PagedAttention, kvantizaci, GQA.

Hierarchie Blackwell: 192 GB HBM3e + 8 TB/s + NVLink 5 (1,8 TB/s) + L2 cache 50+ MB → „AI factory" architektura, kde inference je primárně **memory-bandwidth-limited**.

---

## Pravidla 2026 produkční inference

1. **MoE + GQA + FA-3 + NVFP4** — defaultní stack frontier modelů
2. **vLLM/SGLang/TensorRT-LLM** dle workloadu (general / shared-prefix / NVHW-max)
3. **Continuous batching + PagedAttention** — povinná infrastruktura
4. **EAGLE-3 speculative decoding** — 3–6× free latence
5. **Prompt caching + Batch API** — 70–95 % cost reduction
6. **B200/B300/MI300X** pro datacenter, **Cerebras/Groq** pro low-latency niche
7. **MLX/llama.cpp** pro on-device, **Apple Foundation Models** pro iOS

---

## Zdroje

- [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608)
- [FlashAttention-3 PyTorch blog](https://pytorch.org/blog/flashattention-3/)
- [EAGLE-3 paper](https://arxiv.org/html/2503.01840v1)
- [NVFP4 NVIDIA technical blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [GPTQ AWQ SmoothQuant comparison](https://apxml.com/courses/practical-llm-quantization/chapter-3-advanced-ptq-techniques/comparing-advanced-ptq)
- [Wanda paper](https://arxiv.org/abs/2306.11695)
- [SparseGPT paper](https://arxiv.org/abs/2301.00774)
- [vLLM vs TensorRT-LLM vs SGLang H100 2026](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)
- [GQA paper](https://arxiv.org/pdf/2305.13245)
- [DeepSeekMoE paper](https://arxiv.org/abs/2401.06066)
- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)
- [Prompt caching guide 2026](https://tokenmix.ai/blog/prompt-caching-guide)
- [Cerebras CS-3 vs Groq LPU](https://www.cerebras.ai/blog/cerebras-cs-3-vs-groq-lpu)
- [YaRN paper](https://arxiv.org/abs/2309.00071)
- [StreamingLLM paper](https://arxiv.org/abs/2309.17453)
- [Apple MLX M5 research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Mastering LLM inference optimization — NVIDIA](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
