# 04 — Velké jazykové modely (LLM)

> Vývoj a stav LLM v roce 2026 — od word embeddings k bilionovým MoE reasoning modelům.

---

## Úvod

Rok 2026 přináší dramatickou konvergenci na frontiéře AI: tři modely (GPT-5.4/5.5, Gemini 3.1 Pro a Claude Opus 4.6/4.7) se na složených benchmarcích pohybují v rozmezí čtyř bodů od sebe, zatímco DeepSeek V4 Pro otevřeně publikuje 1,6 bilionu parametrů a uzavírá propast mezi open-weight a uzavřenými modely. Tento průzkum mapuje cestu od statických word embeddings z roku 2013 až k současným bilionovým MoE modelům s reasoning schopnostmi.

---

## 1. Předchůdci: Cesta od statických embeddings k předtrénovaným modelům (2013–2018)

### Word2Vec (2013)

Tomas Mikolov a kolegové z Googlu představili **Word2Vec** jako dvojvrstvou neuronovou síť, která kódovala slova do statických vektorů. Modely Skip-gram a CBOW se staly průlomem díky efektivitě a snadné použitelnosti. Klíčové omezení: slovo mělo jeden vektor bez ohledu na kontext — „banka" v „banka peněz" a „banka řeky" měla identickou reprezentaci.

### GloVe (2014)

**Global Vectors for Word Representation** ze Stanfordu kombinoval globální matrix faktorizaci s lokálním kontextem. Na rozdíl od Word2Vec využíval count-based přístup s většími kontextovými okny.

### ELMo (únor 2018)

**Embeddings from Language Models** od Allen Institute for AI zavedl **kontextové** embeddings — poprvé měla slova různé reprezentace podle kontextu. Postavený na bidirectional LSTM, vyřešil polysémii.

### ULMFiT (2018)

Howard a Ruder zavedli **Universal Language Model Fine-tuning** — založený na AWD-LSTM s třemi fázemi: jazykové předtrénování, doladění na cílové doméně, doladění klasifikátoru. Snížil chybovost o 18–24 %.

---

## 2. BERT a éra encoder-only modelů (2018–2020)

### BERT (říjen 2018)

Google představil **Bidirectional Encoder Representations from Transformers** — první obousměrný předtrénovaný transformer. Trénován dvěma cíli:
- **Masked Language Modeling (MLM)** — 15 % tokenů náhodně maskováno
- **Next Sentence Prediction (NSP)** — predikce, zda věty na sebe navazují

BERT-Base měl 110 M parametrů, BERT-Large 340 M.

### Varianty BERTu

| Model | Rok | Klíčová inovace | Velikost |
|-------|-----|-----------------|----------|
| **RoBERTa** (Meta) | 2019 | Odstranění NSP, větší batche (8k vs 256), 160 GB dat | 125 M / 355 M |
| **DistilBERT** (HuggingFace) | 2019 | Knowledge distillation, 40 % menší, 60 % rychlejší, 97 % výkonu | 66 M |
| **ALBERT** (Google) | 2019 | Faktorizovaná embedding parametrizace, sdílení parametrů | 12 M / 18 M / 60 M |
| **DeBERTa** (Microsoft) | 2020 | Disentangled attention — oddělení obsahu a pozice | 100 M / 1,5 B |

---

## 3. GPT řada — od 117 M k bilionům parametrů (2018–2026)

| Model | Datum vydání | Parametry | Kontextové okno | Klíčové vlastnosti |
|-------|--------------|-----------|-----------------|---------------------|
| **GPT-1** | červen 2018 | 117 M | 512 | První transformer od OpenAI |
| **GPT-2** | únor 2019 | 1,5 B | 1 024 | WebText, zpočátku zadržen |
| **GPT-3** | červen 2020 | 175 B | 2 048 / 4 096 | Few-shot, in-context learning |
| **GPT-3.5 (ChatGPT)** | listopad 2022 | ~175 B | 4 096 / 16 K | RLHF, masová adopce |
| **GPT-4** | březen 2023 | ~1,8 T (odhad) | 8 K / 32 K / 128 K | Multimodalita, MoE 8×220B |
| **GPT-4o** | květen 2024 | neznámé | 128 K | Omni — text/audio/video |
| **GPT-4.1** | duben 2025 | neznámé | 1 M | Coding focus |
| **GPT-5** | srpen 2025 | neznámé | 256 K – 1 M | Hybridní router (Instant + Thinking) |
| **GPT-5.1** | listopad 2025 | neznámé | 1 M | Adaptivní reasoning, 2–3× rychlejší |
| **GPT-5.4** | Q1 2026 | neznámé | 1 M | High-mode reasoning, MMLU ~92 % |
| **GPT-5.5** | 23. duben 2026 | neznámé | 1 M | SOTA Terminal-Bench 2.0 (82,7 %) |

### GPT-5.5 (duben 2026) — klíčové detaily

- Cena: $5/$30 za 1 M input/output tokens
- Vede na Terminal-Bench 2.0 (82,7 % vs Claude 4.7 69,4 %)
- Expert-SWE: 73,1 % (testuje úkoly s mediánem 20 hodin lidské práce)
- GDPval: 84,9 % — 44 reálných profesí
- SOTA na FrontierMath, CyberGym, OSWorld-Verified

---

## 4. Claude řada (Anthropic) — Constitutional AI a „thinking"

### Constitutional AI — klíčová odlišnost

Anthropic používá **CAI** místo čistého RLHF. Proces má dvě fáze:
1. **Supervised** — model generuje sebekritiku a revize podle ústavy
2. **RLAIF (RL from AI Feedback)** — AI místo lidí porovnává odpovědi s ústavními principy

CAI vytváří **Pareto improvement** — více užitečné A méně škodlivé než RLHF.

### Časová osa Claude modelů

| Model | Datum | Klíčové vlastnosti |
|-------|-------|---------------------|
| **Claude 1** | březen 2023 | První veřejné vydání |
| **Claude 2** | červenec 2023 | 100 K kontext |
| **Claude 2.1** | listopad 2023 | 200 K kontext |
| **Claude 3** (Haiku/Sonnet/Opus) | březen 2024 | Multimodalita, vision |
| **Claude 3.5 Sonnet** | červen 2024 | Computer use, lepší coding |
| **Claude 3.7 Sonnet** | únor 2025 | Hybrid reasoning model |
| **Claude 4** (Opus/Sonnet) | květen 2025 | Extended thinking, agentic |
| **Claude Opus 4.5** | 24. listopad 2025 | První AI > 80 % na SWE-bench Verified |
| **Claude Opus 4.6** | 5. únor 2026 | 1M kontext beta, multi-agent koordinace |
| **Claude Opus 4.7** | 16. duben 2026 | +13 % coding, vision 3,75 MP |
| **Claude Mythos 5** | interní 4/2026 | 10T+ parametrů, ASL-4 — NEPUBLIKOVÁN |

### Claude Opus 4.7 — benchmarky

- SWE-bench Pro: 64,3 % (vs GPT-5.5 58,6 %) — vede multi-file coding
- GPQA Diamond: 68,4 %
- LMSYS Coding Arena Elo: **1561** — první model přes 1500
- Cena: $5/$25 za 1 M tokenů

---

## 5. Google modely — LaMDA → PaLM → Gemini

### Evoluce

- **LaMDA** (2021) — Konverzační seq2seq, 137 B parametrů
- **PaLM** (2022) — 540 B, Pathways infrastructure
- **PaLM 2** (květen 2023) — 340 B, 3,6 T tokenů, 100+ jazyků
- **Gemini 1.0** (prosinec 2023) — Native multimodal, varianty Ultra/Pro/Nano
- **Gemini 1.5 Pro** (únor 2024) — MoE, **1 M (rozšiřitelno na 2 M) tokenů**, 1 hodina videa v jediné výzvě
- **Gemini 2.0** (prosinec 2024) — Agentic, real-time multimodal
- **Gemini 2.5 Pro** (2025) — Thinking mode
- **Gemini 3 Pro** (2026) — 81 % na MMMU-Pro, 87,6 % Video-MMMU
- **Gemini 3.1 Pro** (2026) — Vede LMSYS Arena (1493 Elo), GPQA 94,3 %

Gemini 1.5 Pro byl prvním velkým komerčním nasazením **MoE architektury** v Gemini sérii.

---

## 6. Meta Llama — open-weights revoluce

| Model | Datum | Parametry | Kontextové okno | Licence |
|-------|-------|-----------|-----------------|---------|
| **LLaMA 1** | únor 2023 | 7B/13B/33B/65B | 2 048 | Pouze výzkum |
| **Llama 2** | červenec 2023 | 7B/13B/70B | 4 096 | Komerční |
| **Code Llama** | srpen 2023 | 7B/13B/34B | 16 K | Specializovaný |
| **Llama 3** | duben 2024 | 8B/70B | 8 192 | Open weights |
| **Llama 3.1** | červenec 2024 | 8B/70B/**405B** | 128 K | Open weights |
| **Llama 3.2** | září 2024 | 1B/3B/11B/90B | 128 K | Vision varianty |
| **Llama 3.3** | prosinec 2024 | 70B | 128 K | Vylepšený 70B |
| **Llama 4 Scout** | duben 2025 | 17B aktivní (109B), 16 expertů | **10 M tokenů** | Llama 4 License |
| **Llama 4 Maverick** | duben 2025 | 17B aktivní (400B), 128 expertů | 1 M | Komerční |
| **Llama 4 Behemoth** | trénováno | 288B aktivní (~2 T), 16 expertů | — | Preview |
| **Llama 5** | 8. duben 2026 | neznámé | **5 M tokenů** | Open-weights |

Meta CEO Mark Zuckerberg na vydání Llama 5 prohlásil, že překonává GPT-5 a Gemini 2.0 v reasoning, kódování a agentic chování. Llama 5 byla navržena se „System 2 thinking" — pomalou, deliberativní reasoning.

---

## 7. Open-source modely

### Mistral / Mixtral

- **Mistral 7B** (září 2023) — Sliding window attention, GQA
- **Mixtral 8×7B** (prosinec 2023) — SMoE, ~46,7 B total / 12,9 B aktivních
- **Mixtral 8×22B** (duben 2024) — 141 B total / 39 B aktivních, Apache 2.0
- **Mistral Large 2** (2024) — 123 B parametrů
- **Codestral** — 22 B specializovaný coding model

### Qwen (Alibaba)

- **Qwen3** (květen 2025) — Hybrid reasoning, 36 T tokens
- Dense varianty: 0,6B / 1,7B / 4B / 8B / 14B / 32B
- MoE varianty: 30B (3B aktivních) a **235B (22B aktivních)**
- Qwen3-235B-A22B přepíná thinking/non-thinking módy v jediném modelu, překonává DeepSeek-R1 na 17/23 benchmarcích
- **Qwen 3.6-27B** (duben 2026) — Dense model porovnatelný s 397B MoE

### DeepSeek — revoluce z Číny

- **DeepSeek-V2** (květen 2024) — 236 B total, MLA attention
- **DeepSeek-V3** (prosinec 2024) — 671 B total / 37 B aktivních, MoE, MLA
- **DeepSeek-R1** (leden 2025) — Reasoning model, RLVR, 128 K kontext, MIT licence
- **DeepSeek V3.1** (srpen 2025) — Hybrid V3+R1
- **DeepSeek V4-Pro** (24. duben 2026) — **1,6 T parametrů**, 1 M kontext, 80,6 % SWE-bench Verified
- **DeepSeek V4-Flash** — 284 B parametrů

V4 obsahuje:
- Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA)
- Manifold-Constrained Hyper-Connections (mHC)
- Muon optimizer
- Při 1 M kontextu jen 27 % FLOPs a 10 % KV cache vs V3.2
- Cena $0,10–0,30 za 1 M tokenů — až **50× levnější** než GPT-5.2

### Další open-source

- **Yi 34B** (01.AI) — Bilingvální, překonal Falcon 180B
- **Falcon 7B/40B/180B** (TII Abu Dhabi) — Trénován na RefinedWeb
- **Gemma 3** (Google, březen 2025) — 270M / 1B / 4B / 12B / 27B parametrů, 128 K kontext, 140+ jazyků
- **IBM Granite 4.0** — Hybrid Mamba-Transformer

---

## 8. Specializované modely pro kód

| Model | Vývojář | Rok | Parametry | HumanEval |
|-------|---------|-----|-----------|-----------|
| **AlphaCode** | DeepMind | 2022 | 41 B | Specialized |
| **Code Llama** | Meta | 2023 | 7B/13B/34B/70B | 67 % |
| **StarCoder** | BigCode/HF | 2023 | 15 B | 40,8 % |
| **DeepSeek-Coder V2** | DeepSeek | 2024 | 236 B (21 B aktivních) | 81,1 % |
| **Codestral** | Mistral | 2024 | 22 B | 81,1 % |
| **Qwen 2.5 Coder 7B** | Alibaba | 2024 | 7 B | 88,4 % |
| **Qwen 3 Coder** | Alibaba | 2025 | 480B MoE | ~92 % |

---

## 9. Reasoning modely a inference-time scaling (2024–2026)

### Filozofie inference-time compute

Tradiční LLM generuje odpověď okamžitě. Reasoning modely jsou trénovány **velkoplošným reinforcement learningem** k „myšlení před mluvením" — generují skrytý chain-of-thought, prozkoumávají strategie, identifikují vlastní chyby.

Klíčový vztah:
- **Train-time compute** (RL trénink) ↑ → výkon ↑
- **Test-time compute** (délka CoT při inferenci) ↑ → výkon ↑

### Časová osa reasoning modelů

| Model | Datum | Vývojář | Klíčové |
|-------|-------|---------|---------|
| **OpenAI o1** | září 2024 | OpenAI | První velký reasoning model |
| **DeepSeek R1** | leden 2025 | DeepSeek | Open-source reasoning, MIT licence |
| **OpenAI o3** | duben 2025 | OpenAI | O řád více compute, vize v CoT |
| **OpenAI o4-mini** | duben 2025 | OpenAI | Efektivní reasoning |
| **Claude 3.7 Extended Thinking** | únor 2025 | Anthropic | Hybridní reasoning |
| **Gemini 2.5 Thinking** | 2025 | Google | Thinking mode |
| **GPT-5 Thinking** | srpen 2025 | OpenAI | Integrovaný router |
| **Qwen3 Thinking** | květen 2025 | Alibaba | Toggle thinking/non-thinking |
| **GPT-5.1 Thinking** | listopad 2025 | OpenAI | Adaptivní rychlost |

### Jak inference-time scaling funguje

Modely mají různá „nastavení" výpočetního rozpočtu (low/medium/high), která regulují délku interního CoT. Vyšší nastavení = delší řetězec úvah = lepší výkon na složité problémy.

---

## 10. Multimodální modely

| Model | Vstupy | Maximální rozlišení | Pozoruhodné |
|-------|--------|---------------------|-------------|
| **GPT-4V** | Text, obrázky | 2048×2048 | První široce dostupný |
| **GPT-4o** | Text, obraz, audio, video | Native | Real-time hlas |
| **Claude 3 Vision** | Text, obrázky | 1568px long edge | 1,15 MP |
| **Claude 4.7 Vision** | Text, obrázky | 2576px (3,75 MP) | 3× více než předchozí |
| **Gemini 1.5 Pro** | Text, obraz, audio, video, kód | Native, 2M context | 1 hod video |
| **Gemini 3 Pro** | Multimodal | media_resolution param | 81 % MMMU-Pro |
| **Pixtral 12B** | Text, multi-image | Native | 128 K kontext, Apache 2.0 |
| **Llama 4 Scout/Maverick** | Native multimodal | — | 10 M kontext |
| **Molmo** | Text, obrázky | — | Open-source, na úrovni GPT-4V |

---

## 11. Velikost modelů — od milionů k bilionům

### Trend: Sparse > Dense

Místo růstu hustých modelů na biliony parametrů přešla industrie na **MoE (mixture of experts)** — obrovské celkové parametry, ale jen malá část aktivní.

| Model | Total | Aktivní | Architektura |
|-------|-------|---------|--------------|
| GPT-3 | 175 B | 175 B (vše) | Dense |
| Llama 3.1 405B | 405 B | 405 B | Dense |
| Mixtral 8×22B | 141 B | 39 B | Sparse MoE |
| DeepSeek V3 | 671 B | 37 B | Sparse MoE |
| DeepSeek V4-Pro | 1 600 B | 49 B (~3 %) | Sparse MoE |
| Llama 4 Maverick | 400 B | 17 B | Sparse MoE |
| Llama 4 Behemoth | ~2 000 B | 288 B | Sparse MoE |
| Kimi K2.6 | 1 100 B | — | Sparse MoE |
| Grok 4 | ~1 700 B | — | Sparse MoE (odhad) |
| Claude Mythos 5 | 10 000 B+ | 800–1 200 B | Sparse MoE |

**Klíčový insight:** DeepSeek V4 dosahuje frontier výkonu jen s 3 % aktivních parametrů.

---

## 12. Benchmarky a leaderboardy

### Hlavní benchmarky 2026

| Benchmark | Co testuje | SOTA 2026 | Lídr |
|-----------|------------|-----------|------|
| **MMLU** | Obecné znalosti (57 oborů) | ~92 % | GPT-5.4 |
| **MMLU-Pro** | Tvrdší MMLU (10 voleb) | ~85 % | GPT-5.5 |
| **GPQA Diamond** | PhD-level věda | 94,3 % | Gemini 3.1 Pro |
| **HumanEval** | Python kódování | >95 % | Top tier |
| **SWE-bench Verified** | Reálný software engineering | 81,0 % | DeepSeek V4 |
| **SWE-bench Pro** | Multi-file coding | 64,3 % | Claude 4.7 |
| **Terminal-Bench 2.0** | Terminal úkoly | 82,7 % | GPT-5.5 |
| **OSWorld-Verified** | Computer use | 78,0 % | GPT-5.5 |
| **AIME 2025** | Matematická olympiáda | >90 % | Reasoning modely |
| **FrontierMath** | Pokročilá matematika | SOTA | GPT-5.5 |
| **GDPval** | 44 reálných profesí | 84,9 % | GPT-5.5 |
| **MMMU-Pro** | Multimodální | 81 % | Gemini 3 Pro |

### LMSYS Chatbot Arena (duben 2026)

**Obecná aréna:**
1. Claude Opus 4.6 Thinking — 1504 Elo
2. Gemini 3.1 Pro Preview — 1493
3. Grok 4.20 Beta1 — 1491
4. GPT-5.4 High — 1484

**Coding aréna:**
1. Claude Opus 4.6 — **1561** Elo
2. Claude Opus 4.6 Thinking — 1545
3. Claude Sonnet 4.6 — 1523
4. Claude 4.5 Thinking — 1491

---

## 13. Komerční vs Open-source — kompromisy

| Kritérium | Komerční (GPT, Claude, Gemini) | Open-source (Llama, DeepSeek, Qwen) |
|-----------|-------------------------------|-------------------------------------|
| **Maximální výkon** | Mírně vyšší (frontier) | DeepSeek V4 jen 0,2 % od Claude 4.6 |
| **Cena za token** | $5–$30 / 1 M | $0,10–1 / 1 M |
| **Kontext** | Až 1–2 M | Llama 4 Scout: 10 M; V4: 1 M |
| **Fine-tuning** | Omezené | Plné, weights k dispozici |
| **Privacy/On-prem** | Nemožné (API only) | Plně samohostovatelné |
| **Compliance** | Vázáno na vendora | Lokální = lepší GDPR |
| **Reasoning** | Nejlepší (o3, GPT-5.5, Claude) | DeepSeek R1, Qwen3 thinking |

**Klíčový trend 2026:** DeepSeek V4 Pro za $0,10–0,30 nabízí 80,6 % SWE-bench — téměř identicky s Claude Opus 4.6, ale **50× levněji**.

---

## 14. Architektonické inovace 2024–2026

### Mamba a State Space Models (SSM)

**Mamba** (Albert Gu & Tri Dao, 2023) — selektivní SSM s lineární komplexitou v délce sekvence.

### Hybridní architektury

**Jamba** (AI21 Labs, březen 2024):
- První velký hybrid Transformer-Mamba-MoE
- 12 B aktivních / 52 B total
- 256 K kontext na jediném 80 GB GPU
- 3× throughput vs Mixtral 8×7B
- Poměr: 1 transformer vrstva na 8 vrstev celkem

**Jamba 1.5** (2024) — 398 B total / 94 B aktivních, 16 MoE expertů

**IBM Granite 4.0** (2025) — Hybrid Mamba-Transformer pro enterprise

### Trendy 2026

1. **Sparse activation** dominuje (MoE všude)
2. **Hybrid attention/SSM** pro dlouhý kontext
3. **Inference-time scaling** validovaný jako nové paradigma
4. **Multimodality jako standard** — text, vize, audio, video integrovány
5. **Agentic capabilities** — computer use, tool use, multi-step reasoning
6. **Safety scaling** — ASL-4 protokoly, cybersecurity safeguards

---

## Závěr — stav LLM v dubnu 2026

Rok 2026 je rokem **konvergence a fragmentace** současně:

**Konvergence:** GPT-5.5, Claude Opus 4.7, Gemini 3.1 Pro a DeepSeek V4 Pro se nacházejí v rozmezí 4 bodů na složených benchmarcích. Frontier není dál výhradně americká doména — DeepSeek V4 z Číny dorovnává proprietary modely za zlomek ceny.

**Fragmentace:** Žádný model nedominuje všem benchmarkům. GPT-5.5 vede na agentic/computer-use, Claude na coding arénách, Gemini 3.1 na vědě, DeepSeek V4 na cost-efficiency.

**Reasoning paradigma** se etablovalo jako nová osa scaling. Místo dalšího nafukování dense modelů investují labs do (1) MoE sparse activation, (2) inference-time compute, a (3) hybridních architektur Transformer-Mamba.

**Safety** se stává limitujícím faktorem na frontier — Claude Mythos 5 (10T+ parametrů) byl odložen kvůli ASL-4 trigerům.

**Open-weights revoluce** dorazila: Llama 5 (5 M kontext), DeepSeek V4 (1,6 T parametrů, MIT licence), Qwen 3.6.

---

## Zdroje

- [Introducing GPT-5 | OpenAI](https://openai.com/index/introducing-gpt-5/)
- [Introducing GPT-5.5 | OpenAI](https://openai.com/index/introducing-gpt-5-5/)
- [Introducing OpenAI o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)
- [Introducing Claude Opus 4.7](https://www.anthropic.com/news/claude-opus-4-7)
- [Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
- [Gemini 3 — Google DeepMind](https://deepmind.google/models/gemini/)
- [The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Mixtral 8×22B — Mistral AI](https://mistral.ai/news/mixtral-8x22b)
- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [DeepSeek-R1 paper](https://arxiv.org/pdf/2501.12948)
- [Jamba: Hybrid Transformer-Mamba](https://arxiv.org/abs/2403.19887)
- [Understanding Reasoning LLMs — Sebastian Raschka](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)
- [LLM Leaderboard 2026 — Vellum](https://www.vellum.ai/llm-leaderboard)
- [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/)
