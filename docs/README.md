# Umělé neuronové sítě a velké jazykové modely

> Hluboký a intenzivní průzkum neuronových sítí a jejich použití v LLM, sestavený paralelní prací osmi výzkumných agentů z aktuálních internetových zdrojů (stav: duben 2026).

---

## Struktura dokumentace

Celkem 8 tematických dokumentů pokrývajících celou cestu od matematických základů jednoho neuronu po infrastrukturu pro inferenci bilionových modelů a ekonomiku celého oboru.

| # | Soubor | Téma | Klíčová slova |
|---|--------|------|---------------|
| 01 | [01_uvod_do_neuronovych_siti.md](./01_uvod_do_neuronovych_siti.md) | **Úvod do neuronových sítí** | Perceptron, MLP, aktivační funkce, backpropagation, loss funkce |
| 02 | [02_transformer_architektura.md](./02_transformer_architektura.md) | **Architektura Transformer** | Self-attention, MHA/GQA/MLA, RoPE, FFN, MoE, Flash Attention |
| 03 | [03_trenovani_neuronovych_siti.md](./03_trenovani_neuronovych_siti.md) | **Trénování LLM** | AdamW, Muon, Sophia, FP8, FSDP, Chinchilla scaling laws |
| 04 | [04_velke_jazykove_modely.md](./04_velke_jazykove_modely.md) | **Velké jazykové modely** | GPT, Claude, Gemini, Llama, DeepSeek, Mistral, Qwen, reasoning |
| 05 | [05_tokenizace_a_embeddingy.md](./05_tokenizace_a_embeddingy.md) | **Tokenizace a embeddingy** | BPE, WordPiece, SentencePiece, Word2Vec, BGE-M3, Matryoshka |
| 06 | [06_fine_tuning_a_alignment.md](./06_fine_tuning_a_alignment.md) | **Fine-tuning a alignment** | LoRA, QLoRA, RLHF, DPO, GRPO, Constitutional AI, scheming |
| 07 | [07_optimalizace_inference.md](./07_optimalizace_inference.md) | **Optimalizace inference** | KV cache, vLLM, PagedAttention, EAGLE, NVFP4, B200, Cerebras |
| 08 | [08_aplikace_a_trendy.md](./08_aplikace_a_trendy.md) | **Aplikace a trendy 2026** | RAG, GraphRAG, agenti, MCP, computer use, world models |
| 09 | [09_aurora_neuron.md](./09_aurora_neuron.md) | **Neuron Aurora — návrh** | LIF, dendrity, STDP, neuromodulace, event-driven, sparsity |
| 10 | [10_implementacni_plan.md](./10_implementacni_plan.md) | **Implementační plán** | Sprinty 0–6, exit kritéria, MNIST jako první benchmark |

---

## Doporučené pořadí čtení

### Pro začátečníky

1. **01 → 02 → 04** — pochopíte základy, architekturu, jaké modely existují
2. **05** — jak modely zpracovávají text
3. **08** — kde se to dnes používá

### Pro inženýry

1. **02 → 03 → 07** — technický stack od architektury přes trénink po produkci
2. **06** — jak naladit vlastní model
3. **08** — agenti, MCP, RAG vzorce

### Pro researchery

Celá řada od 01 po 08; klíčové reference v každém dokumentu vedou k arXiv paperům.

---

## Hlavní osy průzkumu

### Architektonický pokrok 2017 → 2026

- **2017** — Transformer (8 M – 65 M parametrů), Sinusoidal PE, MHA, ReLU FFN, Post-LN
- **2020** — GPT-3 (175 B), few-shot learning, RLHF era
- **2022** — Chinchilla scaling laws, ChatGPT
- **2023** — LLaMA open weights, Mistral, RoPE/SwiGLU/RMSNorm jako standard
- **2024** — Mixture of Experts dominance (DeepSeek-V3, Mixtral), 1 M+ kontext (Gemini 1.5)
- **2025** — Reasoning modely (o1, R1), Claude 4
- **2026** — 10 T+ MoE (Mythos 5), 5 M kontext (Llama 5), inference-time scaling, agenti

### Klíčové technologie roku 2026

- **Modely** — Sparse MoE (671 B – 2 T parametrů), reasoning (`<think>` tokens), multimodalita native
- **Architektura** — RoPE + GQA/MLA + SwiGLU + Pre-RMSNorm + FlashAttention 3
- **Trénink** — AdamW/Muon, BF16/FP8, FSDP/3D parallelism, Chinchilla+ overtraining
- **Alignment** — DPO/SimPO + Constitutional AI + GRPO pro reasoning
- **Inference** — vLLM/SGLang/TensorRT-LLM, NVFP4 kvantizace, EAGLE-3 spec decoding, prompt caching
- **Aplikace** — Agentní RAG, MCP standard, computer use, code agents

### Konvergence 2026

Frontier modely (GPT-5.5, Claude Opus 4.7, Gemini 3.1 Pro, DeepSeek V4 Pro) se na složených benchmarcích pohybují v rozmezí **4 bodů**. Open-source dohnal closed na **~3 % rozdíl**. Skutečnou bariérou je dnes infrastruktura (energie, čipy, voda), ne modely.

---

## Metodologie

Tento průzkum byl sestaven paralelním zapojením 8 výzkumných agentů, z nichž každý se specializoval na jednu z osmi oblastí:

1. Základy neuronových sítí
2. Architektura Transformer
3. Tréninkové metody
4. Krajina LLM
5. Tokenizace a embeddingy
6. Fine-tuning a alignment
7. Optimalizace inference
8. Aplikace a trendy

Každý agent provedl:
- Vyhledávání aktuálních zdrojů (rok 2025–2026) přes WebSearch
- Stažení a analýzu primárních textů přes WebFetch
- Syntézu detailního technického reportu
- Citaci primárních zdrojů (arXiv papery, oficiální blogy, dokumentace)

Výsledné dokumenty obsahují matematické vzorce, ukázky kódu (PyTorch), srovnávací tabulky, časové osy a odkazy na primární literaturu.

---

## Klíčové primární zdroje

Pokud byste měli číst jen několik prací, doporučení:

- **Vaswani et al. (2017)** — *Attention Is All You Need*
- **Hoffmann et al. (2022)** — *Training Compute-Optimal Large Language Models* (Chinchilla)
- **Ouyang et al. (2022)** — *Training language models to follow instructions* (InstructGPT/RLHF)
- **Hu et al. (2021)** — *LoRA: Low-Rank Adaptation of LLMs*
- **Bai et al. (2022)** — *Constitutional AI: Harmlessness from AI Feedback*
- **Rafailov et al. (2023)** — *Direct Preference Optimization*
- **Dao (2022, 2023, 2024)** — *FlashAttention v1/v2/v3*
- **Kwon et al. (2023)** — *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM)
- **DeepSeek (2024–2025)** — *DeepSeek-V3 Technical Report*, *DeepSeek-R1*

---

*Datum sestavení: 27. 4. 2026*
