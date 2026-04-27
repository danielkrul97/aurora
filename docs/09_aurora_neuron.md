# Neuron Aurora — návrh pro úplné začátečníky

> Cíl: vysvětlit, jak má fungovat jeden umělý neuron v projektu Aurora. Bez předpokladu znalosti neurovědy nebo deep learningu. Když narazíte na pojem, který neznáte, **je vysvětlený o pár řádků níž** — neutíkejte na Wikipedii.

---

## 1. Proč nestačí klasický „neuron" z deep learningu?

Klasický umělý neuron (perceptron) dělá tohle:

```
výstup = aktivační_funkce( w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b )
```

Vezme čísla na vstupu, vynásobí je vahami, sečte, prožene funkcí (sigmoid, ReLU…) a vyplivne číslo. Hotovo.

**Co tomu chybí oproti skutečnému neuronu v mozku:**

| Mozek                                          | Klasická umělá síť                  |
| ---------------------------------------------- | ----------------------------------- |
| Komunikuje krátkými pulzy (**spike**)          | Posílá spojitá čísla                |
| Pracuje **asynchronně** (žádný globální takt)  | Všechno se počítá synchronně po vrstvách |
| Jeden neuron = malá výpočetní jednotka (dendrity dělají vlastní výpočty) | Jeden neuron = jedna vážená suma   |
| **20 wattů** na celý mozek                     | GPT-class modely: megawatty         |
| **Učí se za běhu**, při každém spike           | Učí se v separátní tréninkové fázi  |

Aurora se chce posunout směrem k levému sloupci. Ne dokonale (to neumí nikdo), ale dál než dnešní transformer.

---

## 2. Jak má neuron Aurora fungovat — analogie s vanou

Představte si **vanu s vodou**:

- Voda v ní postupně **uniká dírou v odtoku** (to je „leak", úbytek napětí).
- Občas někdo do vany **přilije kbelík vody** (to je příchozí spike z jiného neuronu).
- Když hladina vody přeleze **rysku** (práh `θ`), vana se **vyprázdní** (neuron vystřelí spike) a chvíli se musí znovu plnit (refrakterní perioda).

Tohle je celý model **Leaky Integrate-and-Fire (LIF)**, na kterém Aurora staví. Matematicky:

```
dV/dt = -(V - V_klid)/τ + Σ vstupy
```

- `V` = aktuální „hladina" (membránový potenciál)
- `V_klid` = kam hladina klesá, když nikdo nepřilévá (~ -70 mV v biologii, u nás libovolná konstanta)
- `τ` = jak rychle uniká voda (časová konstanta)
- `Σ vstupy` = součet kbelíků, které právě přiletěly

**Slovník:**
- **Spike** = krátký puls. Buď je, nebo není (1 nebo 0). Žádné „spike o velikosti 0,73". Informace je v **tom, kdy** spike přijde.
- **Membránový potenciál** = ta hladina ve vaně. Spojitá hodnota uvnitř neuronu, navenek se nikdy nezveřejní — ven jde jen spike.
- **Práh** = ryska, po jejímž překročení neuron vystřelí.

---

## 3. Dendrity — neuron jako malá síť uvnitř

Tady je největší skok od klasiky. V biologii **dendrity nejsou jen dráty**, co vedou vstup do těla buňky. Sami si dělají **lokální výpočty**.

Analogie: představte si firmu. Klasický neuron je ředitel, který dostane všechny e-maily naráz, sečte je a rozhodne. **Reálný neuron má vedoucí oddělení (dendrity)**, kteří si nejdřív filtrují e-maily v rámci svého oddělení a teprve když se v oddělení nasčítá dost důležitých zpráv, **pošlou ředitelovi shrnutí**.

To znamená: **jeden biologický neuron má sílu malé vícevrstvé sítě.** (Beniaguev et al. 2021 ukázali, že modelovat jeden pyramidální neuron přesně potřebuje 5–8 vrstev klasické umělé sítě.)

**Návrh pro Aurora:** každý neuron má víc **dendritických kompartmentů**. Každý kompartment:
1. Sbírá vstupy ze svých synapsí.
2. Má vlastní práh.
3. Když ho překročí, pošle do somy (těla neuronu) silnější „balík" — tzv. **dendritický spike**.

```
Neuron
├── Dendrit A: synapse 1, 2, 3 → lokální práh → výstup do somy
├── Dendrit B: synapse 4, 5, 6 → lokální práh → výstup do somy
├── Dendrit C: synapse 7, 8, 9 → lokální práh → výstup do somy
└── Soma: LIF (vana z bodu 2) → spike ven
```

Zisk: méně neuronů potřeba na stejný úkol → méně paměti, méně spotřeby.

---

## 4. Asynchronie — „když se nic neděje, nic se nepočítá"

Klasická síť: v každém kroku se přepočítá **každý neuron**, i kdyby měl na vstupu nuly. Plýtvání.

Aurora: **event-driven** běh. To znamená:

1. Spike je událost (event) v **prioritní frontě** (heap seřazený podle času).
2. Pracovní vlákno (worker thread) si vždycky vytáhne nejbližší event.
3. Doručí ho cílovým synapsím → ty zvednou hladinu v cílových dendritech.
4. Pokud cílový dendrit nebo soma překročí práh, **vznikne nový event** (spike) → vloží se zpátky do fronty.
5. **Když fronta je prázdná → CPU spí.**

Žádný globální takt. Žádný „update všech neuronů každých 10 ms". Neurony, které právě nikdo nedráždí, **nestojí ani jeden cyklus CPU**.

> **Proč tohle Rust?** Protože Rust dává plnou kontrolu nad pamětí, vlákny a schedulerem. Žádný garbage collector vám náhodně neusekne 50 ms. Knihovny: `std::collections::BinaryHeap` pro frontu, `rayon` pro paralelizaci, `tokio` kdyby šlo o async I/O.

A když budete jednou chtít, tenhle model **přirozeně sedne na neuromorfní čipy** (Intel Loihi 2, IBM TrueNorth, SpiNNaker) — jejich celý design je „doručuj eventy mezi neurony". Takže Aurora by tam šla portovat bez přepisu architektury.

---

## 5. Sparsity (řídkost) — proč je mozek tak úsporný

V mozku v každém okamžiku **střílí jen ~1–5 % neuronů**. Zbytek mlčí. Tohle je hlavní důvod, proč mozek vystačí s 20 watty.

Aurora to vynucuje **dvojím mechanismem**:

1. **Práh** — neuron musí dostat dost vstupů, aby překonal `θ`. Většina jich nedostane → mlčí.
2. **Homeostáza** — pomalá samoregulace prahu. Když neuron stříká moc často (řekněme >10 % času), práh se mu pomalu zvedne. Když nestříká vůbec, klesne. Cíl: každý neuron drží průměrnou „zaměstnanost" kolem cílové hodnoty (např. 3 %).

Bez homeostázy mají SNN sítě dva failure módy: buď **všechno mlčí** (nic se nenaučí), nebo **všechno střílí pořád** (žádná informace v signálu). Homeostáza je hlídací pes.

---

## 6. Učení v reálném čase — bez backpropagation

Tady je největší filozofický rozdíl od dnešních LLM.

**Backpropagation** (jak se učí GPT, Claude, Llama…):
1. Pošli vstup celou sítí dopředu.
2. Spočítej chybu na výstupu.
3. **Spočítej gradient zpátky** přes všechny vrstvy (řetězové pravidlo).
4. Uprav všechny váhy podle gradientu.

Tohle mozek **prokazatelně nedělá** — žádný neuron nemá přístup k „chybě na výstupu sítě". Učení v mozku je **lokální**: každá synapse vidí jen, co se děje na ní samotné.

**Aurora používá tři lokální pravidla:**

### 6a. STDP — Spike-Timing Dependent Plasticity

Pravidlo: **„Co střílí společně, drží se spolu."** (Hebbian rule, ale s časem.)

- Když **neuron A vystřelí krátce před neuronem B** (a synapse A→B se na tom podílela) → synapse **zesílí**. Logika: A asi pomohl B vystřelit, posilme to.
- Když **A vystřelí krátce po B** → synapse **zeslábne**. Logika: A přišel pozdě, k ničemu nebyl, oslabme.

Vzorec:
```
Δw = +A⁺ · exp(-Δt/τ⁺)   pokud pre vystřelil před post
Δw = -A⁻ · exp(+Δt/τ⁻)   pokud pre vystřelil po post
```

Kde `Δt` je časový rozdíl spike pre a post. Funguje **úplně lokálně** — synapse jen sleduje časy svého pre a post neuronu.

### 6b. Three-factor rule — kde se bere „odměna"

Samotná STDP nestačí — neumí síti říct „tohle bylo dobré rozhodnutí, posilni ho". K tomu potřebujeme **třetí faktor**: globální signál odměny/chyby. V mozku to dělají **neuromodulátory** (dopamin, serotonin…).

```
Δw = STDP × M(t)
```

`M(t)` je hodnota „dopaminu" v daný moment. Když síť udělá něco správně → `M` stoupne → STDP změny se „uloží". Když udělá chybu → `M` klesne nebo zporní. Bez `M` se nic neučí.

Tohle je **nejbližší dostupná náhrada za backprop**, která je pořád lokální (synapse vidí jen pre, post a globální `M`).

### 6c. Homeostatická plasticita

Viz bod 5. Pomalá úprava prahu, drží neurony v rozumném pásmu aktivity. Druh „garbage collection" pro učení.

**Klíčové: všechno se updatuje při příchodu spike na synapsi.** Žádný „teď je tréninková fáze, teď je inferenční". Síť **se učí pořád, jak běží**. To je ten „real-time" v zadání.

---

## 7. Co tohle stojí — trade-offy, o kterých musíte vědět předem

Nelze jen říct „bude to jak mozek a bude to skvělé". Tady jsou reálné překážky:

### 7a. Trénink na úrovni LLM je nevyřešený problém

Lokální učení (STDP + neuromodulace) **funguje na malých úkolech** (klasifikace MNIST a podobně). **Nikdo zatím neukázal**, že by čistě lokálním učením postavil jazykový model konkurující transformeru.

**Realistická cesta pro Aurora:**
1. **Pretraining** klasickým gradientem (surrogate gradients pro spiking sítě, nebo ANN→SNN konverze).
2. **Online fine-tuning a adaptaci** lokální plasticitou (STDP + neuromodulace).

Tj. „mozkovost" se uplatní hlavně po deploymentu — model se **dál učí během používání**, což dnešní LLM neumějí (jsou zamrzlé po tréninku).

### 7b. Energie je výhra **jen na správném hardwaru**

Na GPU je event-driven SNN často **pomalejší** než dense matmul. GPU je optimalizované pro „spočítej miliardu násobení paralelně", ne pro „doruč jednu zprávu mezi dvěma neurony".

**Skutečná energetická úspora přijde s:**
- Neuromorfním čipem (Loihi 2, SpiNNaker) — 100–1000× nižší spotřeba na inference,
- nebo aspoň dobře napsaným event loopem na CPU s lots of cores.

Rust je dobrá volba pro to druhé. Pro první budete potřebovat speciální HW, ke kterému se dostanete jen přes výzkumný program (Intel/IBM).

### 7c. „Custom LLM" + biologická věrnost je napětí

Čím blíž biologii, tím dál od osvědčených LLM triků: **attention mechanism, KV cache, batch inference, RoPE, FlashAttention**… nic z toho nemá biologický protějšek.

Musíte si **dopředu vybrat na ose**:

```
plně biologický SNN <─────────────────────────> klasický transformer
        ↑                                              ↑
   maximum úspory,                            maximum kvality,
   minimum kvality                            minimum úspory
```

Doporučení: **hybridní střed.** Biologicky inspirovaná páteř (spiking neurony, dendrity, lokální plasticita), ale půjčit si z transformeru to, co prokazatelně funguje (např. attention-like mechanismus, ale implementovaný přes spiking).

---

## 8. Konkrétní další krok

Než se začne kódovat:

1. **Tento dokument je v9. 0** — formálně probrat, co se v něm nelíbí, doplnit chybějící rozhodnutí.
2. Sepsat **detailnější specifikaci** s konkrétními rovnicemi, parametry (τ, θ, A⁺, A⁻…) a pseudokódem v Rustu.
3. **Prototyp jednoho neuronu** v Rustu — vana z bodu 2 + jeden dendrit + jedna synapse s STDP. Test: dokáže se naučit reagovat na opakovaný vstupní vzor (~50 řádků kódu).
4. **Mini-síť** — 2–3 neurony, klasifikace XOR. Sanity check, že event loop funguje.
5. **Klasifikace MNIST** spiking sítí — první „ostrý" benchmark, kde se ukáže, jestli tohle vůbec konverguje.
6. Až teď uvažovat o jazykovém modelu.

Bez kroků 3–5 jakákoli LLM diskuze pluje ve vzduchu.

---

## Glosář (rychlá referenční karta)

| Pojem | Český překlad / význam |
| --- | --- |
| **Spike** | Krátký puls, základní jednotka komunikace mezi neurony |
| **LIF** (Leaky Integrate-and-Fire) | Základní spiking model — vana s dírou |
| **Membránový potenciál** (`V`) | Hladina v té vaně |
| **Práh** (`θ`) | Kdy vana přeteče → neuron vystřelí |
| **Refrakterní perioda** | Krátká doba po spike, kdy neuron nemůže znovu vystřelit |
| **Soma** | Tělo neuronu, kde se rozhoduje o spike |
| **Dendrit** | Vstupní „rameno" neuronu, dělá lokální výpočty |
| **Synapse** | Spojení mezi dvěma neurony, má váhu a zpoždění |
| **STDP** | Pravidlo učení založené na časování spiků |
| **Neuromodulátor** | Globální chemický signál (analog dopaminu) — gate pro učení |
| **Homeostáza** | Samoregulace, drží neurony v cílové průměrné aktivitě |
| **Sparsity** | Řídkost — málo neuronů aktivních najednou |
| **Event-driven** | Počítá se jen když se něco děje |
| **Backpropagation** | Klasický algoritmus učení v deep learningu — mozek ho nedělá |
| **SNN** (Spiking Neural Network) | Síť spiking neuronů, na rozdíl od klasické ANN |

---

*Verze 0.1 — návrh k diskuzi, ne finální specifikace. Datum: 27. 4. 2026.*
