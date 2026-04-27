# 05 — Tokenizace a embeddingy

> Hluboký průzkum tokenizace, vektorových reprezentací a vyhledávání pro LLM v roce 2026.

---

## 1. Co je tokenizace

**Tokenizace** je proces rozdělení textu na atomické jednotky (tokeny), které model umí zpracovat. Každý token je následně namapován na celočíselné ID, jež slouží jako vstup pro embedding vrstvu.

Vztah token ↔ slovo není 1:1:
- V angličtině typicky **1 token ≈ 0,75 slova** (≈ 4 znaky)
- V češtině obvykle **1 token ≈ 0,4–0,6 slova** (kvůli diakritice, flexi)
- Pro CJK jazyky může jeden znak generovat 2–3 tokeny

Pipeline: text → tokeny → ID → embedding → transformer → logity → token → text.

---

## 2. Word-level vs character-level vs subword

| Typ | Velikost slovníku | OOV problém | Délka sekvence | Sémantika |
|-----|-------------------|-------------|----------------|-----------|
| Word-level | 100k–1M+ | Velký (UNK) | Krátká | Vysoká |
| Character-level | ~256 | Žádný | Velmi dlouhá | Slabá |
| Subword (BPE/WP/SP) | 30k–200k | Žádný | Optimální | Dobrá |

**Word-level** trpí explozí slovníku — v češtině s bohatou flexí je to katastrofa (`pes`, `psa`, `psovi`, `pse`, `psem`, `psi`, …).

**Subword** je dnešní zlatý standard: časté kořeny zůstávají atomické, vzácná slova se rozpadnou.

---

## 3. Byte-Pair Encoding (BPE)

BPE byl publikován Philipem Gagem v roce 1994 jako kompresní algoritmus a později adaptován Sennrichem et al. (2016) pro NMT. OpenAI jej zpopularizovalo v GPT-2.

**Trénink krok po kroku:**

```
Korpus: ("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
Init slovník: ["b", "g", "h", "n", "p", "s", "u"]
```

1. Spočítej frekvenci všech sousedních párů. Pár `(u, g)` má frekvenci 20.
2. Sluč nejčastější pár: `(u, g) → ug`. Slovník: `[..., "ug"]`.
3. Přepiš korpus: `("h", "ug", 10), ("p", "ug", 5), ("p", "u", "n", 12), ...`
4. Opakuj. Další sloučení: `(u, n) → un` (16×).
5. Pak `(h, ug) → hug` (15×).
6. Konec po dosažení požadované velikosti slovníku (typicky 30k–200k).

**Použití v GPT řadě:** GPT-2/3 (50 257), GPT-4 (cl100k_base, 100k), GPT-4o (o200k_base, 200k).

---

## 4. WordPiece (BERT)

WordPiece byl vyvinut Googlem a používá ho BERT, DistilBERT, Electra. **Klíčový rozdíl od BPE — kritérium slučování:**

- BPE: maximalizuje **frekvenci** páru → `score = freq(a, b)`
- WordPiece: maximalizuje **likelihood korpusu** (PMI-podobné skóre):

```
score(a, b) = freq(a, b) / (freq(a) × freq(b))
```

WordPiece preferuje páry, kde se elementy spolu vyskytují **více, než by se čekalo náhodou**.

**Notace:** pokračující subword je označen prefixem `##`. Slovo `tokenization` → `token`, `##ization`.

**Tokenizace:** WordPiece neukládá merge rules, jen finální slovník. Použije **greedy longest-match-first**.

---

## 5. SentencePiece (Unigram, T5, Llama)

**SentencePiece** je tokenizační framework od Googlu (Kudo & Richardson, 2018), který:
- Pracuje přímo s **raw textem** (bez pre-tokenizace na slova podle mezer)
- Mezery reprezentuje speciálním znakem `▁` (U+2581)
- Podporuje BPE i Unigram algoritmy
- Je jazykově agnostický — ideální pro čínštinu, japonštinu

**Unigram algoritmus** (Kudo, 2018) funguje **opačně než BPE**:

1. Začne s **velkým slovníkem** (např. milion kandidátů)
2. Přiřadí každému tokenu pravděpodobnost (unigram language model)
3. **Iterativně odstraňuje** tokeny, jejichž odstranění nejvíce minimalizuje pokles likelihoodu
4. Trénuje pomocí **EM algoritmu**
5. Končí po dosažení cílové velikosti

**Klíčová výhoda:** každý token má pravděpodobnost — model může vzorkovat **různé tokenizace** stejného slova (subword regularization, data augmentation).

**Použití:** T5 (32 100), AlBERT, mBART, XLNet, **Llama** (32k SP-BPE), Gemini, Gemma.

---

## 6. Byte-level BPE (GPT-2/3/4)

**Problém klasického BPE:** pokud korpus neobsahuje znak (emoji, vzácný Unicode), vrátí UNK.

**Řešení byte-level BPE** (Radford et al., GPT-2, 2019):
- Slovník začne **256 byty UTF-8**
- Jakákoli sekvence znaků se prvně zakóduje do UTF-8 bytes
- BPE merge se provádí nad bytes
- **Žádný UNK**

```python
"Ahoj".encode("utf-8")  # b'Ahoj' = [65, 104, 111, 106]
"😀".encode("utf-8")     # b'\xf0\x9f\x98\x80' = [240, 159, 152, 128]
```

GPT-2 vocab = 256 (byty) + 50 000 merges + 1 (`<|endoftext|>`) = **50 257**.

---

## 7. Tiktoken a Hugging Face tokenizers

**Tiktoken** je Rust knihovna od OpenAI pro rychlé BPE tokenizace.

| Encoding | Vocab size | Modely |
|----------|------------|--------|
| `r50k_base` (gpt2) | 50 257 | GPT-2, GPT-3 (davinci) |
| `p50k_base` | 50 281 | Codex, text-davinci-002/003 |
| `cl100k_base` | 100 277 | GPT-3.5, GPT-4, text-embedding-3 |
| `o200k_base` | 200 019 | GPT-4o, o1, o3, o4-mini |
| `o200k_harmony` | 201k+ | Open-source GPT (gpt-oss) |

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
enc.encode("Ahoj svete!")  # [3848, 73, 7811, 26, 0]
```

`o200k_base` snižuje počet tokenů u neanglických textů cca o 30–50 % oproti `cl100k_base` díky lepšímu pokrytí jazyků.

---

## 8. Tokenizace pro různé jazyky

**Tokenizer Inequality** (Petrov et al., 2023):

| Jazyk | Tokeny / 1000 znaků (cl100k) |
|-------|------------------------------|
| Angličtina | ~250 |
| Čeština | ~450 |
| Čínština | ~600 |
| Barmština | ~1500 |

**Čeština:** kombinace bohaté flexe + diakritika + relativně malý podíl v trénovacích datech. Slovo `nejnespravedlivějšími` se tokenizuje na 8–12 subwordů místo 1. Důsledky: pomalejší inference, dražší API, kratší efektivní context window.

**Čínština:** každý znak je 3 byty UTF-8. Často 1 znak = 1–2 tokeny. Bez mezer mezi slovy je SentencePiece-like přístup nutností.

**Kód:** GPT-4 přidal speciální tokeny pro běžné kódové vzory (`    ` = 4 mezery jako 1 token), což drasticky zlepšilo coding performance.

**Emoji:** mohou být 1 token (časté, např. 😀) nebo až 4–8 tokenů (ZWJ kombinace, např. 🧑‍💻).

---

## 9. Speciální tokeny

| Token | Účel |
|-------|------|
| `<bos>` / `<\|startoftext\|>` | Začátek sekvence |
| `<eos>` / `<\|endoftext\|>` | Konec sekvence (stop generation) |
| `<pad>` | Padding pro batching |
| `<unk>` | Out-of-vocabulary |
| `<cls>` | Klasifikační token (BERT) |
| `<sep>` | Separátor segmentů (BERT NSP) |
| `<mask>` | Pro masked language modeling |

**Role tokeny v chat templatech** (Llama 3, Qwen, Mistral, OpenAI Harmony):

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Ahoj!<|im_end|>
<|im_start|>assistant
```

**Tool calling tokeny:** Llama 3.1+ má `<|python_tag|>`, Qwen má `<tool_call>...</tool_call>`, GPT-OSS používá strukturované Harmony JSON.

---

## 10. Glitch tokeny — SolidGoldMagikarp

V únoru 2023 výzkumníci (Rumbelow & Watkins) objevili „anomalous tokens" v GPT-2/3:

- `SolidGoldMagikarp` (Reddit username)
- `petertodd`
- ` davidjl` (Counter-Strike player)
- ` TheNitromeFan`

**Jak vznikají:**
1. Tokenizer GPT-2 byl trénován na **WebText** (Reddit links)
2. V tomto datasetu byly Reddit usernames jako `SolidGoldMagikarp` extrémně časté
3. BPE je sloučil do jediného tokenu
4. Trénovací dataset GPT-3 měl jiné složení → tyto tokeny se tam **téměř nevyskytly**
5. Embedding pro tento token zůstal blízko inicializace (random)
6. Model „neví, co s ním" → halucinuje, refuzuje, glitchuje

**Důsledek:** bezpečnostní attack vector. Moderní modely (GPT-4o, Claude) mají filtrovaný tokenizer training corpus.

---

## 11. Embeddings — úvod

**Embedding** = mapování diskrétního tokenu (či sekvence) na **dense vektor** v reálném prostoru ℝᵈ (typicky d = 256–4096).

**Distribuční hypotéza** (Harris, 1954; Firth, 1957): *„You shall know a word by the company it keeps."*

**Vlastnosti dobrých embeddingů:**
- Sémantická podobnost ↔ geometrická blízkost
- Lineární vztahy zachycují analogie: `king − man + woman ≈ queen`
- Možnost klastrování, vizualizace (t-SNE, UMAP)

---

## 12. Word2Vec (Mikolov et al., 2013)

**CBOW (Continuous Bag of Words):** predikuje cílové slovo z kontextu.
```
Input: [the, quick, brown, ___, jumps, over]
Output: fox
```

**Skip-gram:** predikuje kontext z cílového slova.
```
Input: fox
Output: [quick, brown, jumps, over]
```

Skip-gram funguje lépe pro vzácná slova; CBOW je rychlejší.

**Negative sampling:** místo softmaxu přes celý slovník (V ~ 1M, intractable):
- Pro pozitivní pár (target, context) trénuj jako binary classification
- Vzorkuj **K = 5–20 negativních slov** z noise distribuce P(w) ∝ freq(w)^0.75
- Loss: `L = −log σ(v_c · v_t) − Σ_{i=1}^{K} 𝔼_{w_i ~ P} log σ(−v_{w_i} · v_t)`

**Hierarchical softmax:** alternativa — slova v binárním Huffmanově stromě, redukce O(V) na O(log V).

---

## 13. GloVe (Pennington, Socher, Manning, 2014)

GloVe spojuje **globální matrix factorization** (LSA) s **lokálním context window** (Word2Vec).

**Postup:**
1. Spočítej **co-occurrence matrix** X, kde Xᵢⱼ = kolikrát se slovo j objevuje v kontextu slova i
2. Trénuj vektory tak, aby `w_i^T · w̃_j + b_i + b̃_j = log(X_ij)`
3. Loss s weighting funkcí f (potlačuje vzácné páry)

**Rozdíl od Word2Vec:** GloVe využívá **globální statistiku** najednou, nikoli lokální okénka iterativně.

---

## 14. Kontextuální embeddings — ELMo, BERT

**Limit Word2Vec/GloVe:** jeden vektor na slovo. `bank` (břeh) = `bank` (banka).

**ELMo** (Peters et al., 2018): bi-LSTM, každé slovo dostává **kontextový vektor**.

**BERT** (Devlin et al., 2018): Transformer encoder, MLM + NSP. Embedding tokenu závisí na celém kontextu.

**Empirická analýza** (Ethayarajh, EMNLP 2019): méně než 5 % variance kontextových embeddingů lze vysvětlit statickým embeddingem daného slova.

| Vlastnost | Statické (W2V/GloVe) | Kontextuální (BERT) |
|-----------|---------------------|---------------------|
| Reprezentace | Lookup table | Funkce kontextu |
| Polysémie | Neřeší | Řeší |
| Velikost | Malá | Velká |
| Inference | O(1) | O(L²) |

---

## 15. Sentence embeddings

Pouhé průměrování BERT token embeddingů dává **špatné** sentence embeddings (anizotropní prostor).

**Sentence-BERT (SBERT)** (Reimers & Gurevych, EMNLP 2019):
- Siamese network: 2× BERT s sdílenými váhami
- Trénink na NLI (SNLI, MNLI) s triplet loss / contrastive loss
- Mean pooling tokenů → fixed-size vektor (768 dim)
- Cosine similarity dvou vět: 65 hodin → **5 sekund** oproti BERT cross-encoderu

**LaBSE** (Language-agnostic BERT Sentence Embedding, Google 2020):
- Dual encoder trénovaný na 6 miliardách překladových párů
- Pokrývá **109 jazyků** v jednom prostoru

**Moderní 2026 sentence encoders:** BGE-M3, GTE-large, mxbai-embed-large, Nomic Embed v2.

---

## 16. Vektorové databáze (2026)

| DB | Latency p99 | QPS | Strength |
|----|-------------|-----|----------|
| FAISS (lib) | 3 ms | 15 000 | Max kontrola, lokální |
| Qdrant | 2 ms | 12 000 | Rust, low-latency |
| Milvus | 5 ms | 8 000 | Enterprise, billions vectors |
| Pinecone | 8 ms | 5 000 | Fully managed |
| Weaviate | 10 ms | 4 000 | Hybrid search, multimodal |
| Chroma | — | — | Embedded, dev-friendly |
| pgvector | — | — | PostgreSQL extension |

**Indexy:** HNSW (Hierarchical Navigable Small World) je dnes standard. IVF, PQ pro velké korpusy.

---

## 17. Dense vs Sparse retrieval, hybrid search

**Sparse (BM25)** — TF-IDF rozšíření s probabilistickou fundací:

```
BM25(D, Q) = Σ_{q ∈ Q} IDF(q) · [f(q,D)·(k_1+1)] / [f(q,D) + k_1·(1 − b + b·|D|/avgdl)]
```

s typickými hyperparametry k₁=1.2–2.0, b=0.75.

| Aspekt | BM25 | Dense (vector) |
|--------|------|----------------|
| Match | Lexikální (exact) | Sémantický |
| Selhává na | Synonymech, parafrázi | Vlastních jménech, číslech |
| Index | Inverted | HNSW/IVF |
| Trénink | Žádný | Miliony párů |

**Hybrid search — Reciprocal Rank Fusion (RRF):**

```
RRF(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))      (k=60)
```

Empirie 2026: BM25 sám ~65 % recall@10, dense ~78 %, **hybrid ~91 %**.

---

## 18. Embedding modely 2024–2026

| Model | Dim | MTEB | Cena $/1M tok | Specialita |
|-------|-----|------|---------------|------------|
| OpenAI text-embedding-3-large | 3072 (MRL) | 64.6 | 0.13 | Default, MRL |
| OpenAI text-embedding-3-small | 1536 (MRL) | 62.3 | 0.02 | Levný |
| Cohere Embed v4 | 1536 | 65.2 | 0.10 | 100+ jazyků |
| Voyage-3-large | 1024 | 65.5 | 0.18 | Code, RAG |
| Voyage-3-lite | 512 | 60.4 | 0.02 | Cheap |
| BGE-M3 (BAAI) | 1024 | 63.0 | self-host | Dense+sparse+ColBERT |
| Jina Embeddings v3 | 1024 | 62.4 | 0.05 | Long context (8k) |
| mxbai-embed-large | 1024 | 64.7 | self-host | Apache 2.0 |
| Gemini Embedding 2 | 3072 (MRL) | 66.1 | 0.15 | Multimodální |
| Nomic Embed v2 | 768 | 62.0 | open | Reproducible |

**BGE-M3** je unikátní — produkuje dense + sparse + multi-vector (ColBERT-style) v jednom modelu.

---

## 19. Matryoshka embeddings (MRL)

**Idea** (Kusupati et al., NeurIPS 2022): trénuj embedding tak, aby **prefix vektoru** byl sám platným embeddingem.

**Trénink:** místo loss jen na plné dimenzi, sečti loss na multiplicitě dimenzí:

```python
from sentence_transformers.losses import MatryoshkaLoss

base_loss = CoSENTLoss(model=model)
loss = MatryoshkaLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=[768, 512, 256, 128, 64],
    matryoshka_weight=[1, 1, 1, 1, 1],
)
```

**Výsledek:** prvních 64 dim 768-dim embeddingu funguje jako 64-dim embedding s 98,37 % výkonu.

**Praktická aplikace:**
- OpenAI text-embedding-3-large má `dimensions` parameter
- 256-dim text-embedding-3-large překonává 1536-dim text-embedding-ada-002
- **Adaptive retrieval** — shortlisting na 64-dim, reranking na 768-dim → 5× rychlejší search

---

## 20. Metriky podobnosti

| Metrika | Vzorec | Rozsah | Použití |
|---------|--------|--------|---------|
| Cosine | `cos θ = (a·b)/(‖a‖‖b‖)` | [−1, 1] | Text, normalizované |
| Dot product | `a·b = Σ a_i b_i` | (−∞, ∞) | Když magnituda nese info |
| Euclidean (L2) | `‖a − b‖_2 = √(Σ(a_i − b_i)²)` | [0, ∞) | Klastrování |
| Manhattan (L1) | `Σ \|a_i − b_i\|` | [0, ∞) | Robustní k outlierům |

**Klíčová věta:** pro **L2-normalizované vektory** (`‖a‖ = ‖b‖ = 1`):

```
‖a − b‖_2² = 2 − 2(a · b) = 2(1 − cos θ)
```

⇒ Cosine, dot product i Euclidean produkují **identické pořadí** výsledků.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cos_sim_batch(A, B):  # A: (n, d), B: (m, d)
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_norm @ B_norm.T  # (n, m)
```

---

## Shrnutí

Tokenizace se posunula od word-level (limity OOV) přes BPE/WordPiece až k byte-level BPE a SentencePiece, které dnes pohánějí všechny moderní LLM. OpenAI o200k_base zlepšil multijazyčnou efektivitu o 30–50 %. Embeddingy prošly cestou statických (Word2Vec, GloVe) → kontextuálních (ELMo, BERT) → sentence-level (SBERT) → moderních generalistů (text-embedding-3, Cohere v4, BGE-M3) s podporou Matryoshka MRL. Hybrid search (BM25 + dense + RRF) je 2026 zlatým standardem pro RAG, dosahuje 91 % recall@10.

---

## Zdroje

- [Hugging Face BPE tokenization](https://huggingface.co/learn/llm-course/chapter6/5)
- [Byte-pair encoding — Wikipedia](https://en.wikipedia.org/wiki/Byte-pair_encoding)
- [karpathy/minbpe](https://github.com/karpathy/minbpe)
- [tiktoken — openai](https://github.com/openai/tiktoken)
- [Hugging Face WordPiece](https://huggingface.co/learn/llm-course/en/chapter6/6)
- [SentencePiece Subword Tokenization](https://mbrenndoerfer.com/writing/sentencepiece-subword-tokenization-bpe-unigram)
- [SolidGoldMagikarp Alignment Forum](https://www.alignmentforum.org/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)
- [Word2Vec Tutorial — Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
- [GloVe Stanford NLP](https://nlp.stanford.edu/projects/glove/)
- [Sentence Transformers documentation](https://sbert.net/)
- [LaBSE](https://huggingface.co/sentence-transformers/LaBSE)
- [Matryoshka Embeddings — Hugging Face Blog](https://huggingface.co/blog/matryoshka)
- [Best Embedding Models 2026](https://www.openxcell.com/blog/best-embedding-models/)
- [Vector Database Comparison 2026 — DataCamp](https://www.datacamp.com/blog/the-top-5-vector-databases)
- [Hybrid Search Guide](https://blog.supermemory.ai/hybrid-search-guide/)
- [Vector Similarity — Pinecone](https://www.pinecone.io/learn/vector-similarity/)
