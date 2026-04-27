# 01 — Úvod do umělých neuronových sítí

> Hluboký průzkum základů, na kterých stojí celá moderní AI revoluce — od perceptronu z roku 1958 až po aktivační funkce moderních LLM.

---

## 1. Historie a vývoj (1943–2026)

### McCulloch-Pittsův neuron (1943)

Warren McCulloch (neurolog) a Walter Pitts (matematik) publikovali zlomovou práci **„A Logical Calculus of the Ideas Immanent in Nervous Activity"**. Představili první matematický model neuronu a argumentovali, že neurony s binární prahovou aktivační funkcí jsou ekvivalentní výrokům prvního řádu logiky.

**Omezení MCP modelu:**
- Pouze binární vstupy a výstupy {0, 1}
- Pouze prahová step funkce
- Žádné váhování vstupů
- Žádný učící mechanismus

Matematicky: `y = 1` pokud `Σx_i ≥ θ` (a žádné inhibiční vstupy nejsou aktivní), jinak `0`.

### Rosenblattův perceptron (1957–1958)

Frank Rosenblatt veřejně demonstroval **perceptron** 7. července 1958 ve Washingtonu, D.C. Šlo o první stroj schopný se učit. Klíčové inovace oproti MCP:

- Reálné (spojité) vstupy a váhy
- Aktivace založená na váženém součtu plus bias: `y = sign(w·x + b)`
- **Perceptron learning rule** — první algoritmus strojového učení pro neurony

### Minsky-Papertova kritika a první AI zima (1969)

Marvin Minsky a Seymour Papert v knize **„Perceptrons: An Introduction to Computational Geometry"** matematicky dokázali, že jednovrstvý perceptron nemůže řešit lineárně neseparovatelné problémy — zejména **XOR funkci**. Důsledkem bylo téměř 10 let stagnace („první AI zima").

### Renesance (1986, 2006, 2012)

- **1986** — Rumelhart, Hinton a Williams znovuobjevili a popularizovali **backpropagation** v práci *„Learning representations by back-propagating errors"*.
- **2006** — Hinton zavedl deep belief networks; počátek moderního hlubokého učení.
- **30. září 2012** — **AlexNet** (Krizhevsky, Sutskever, Hinton) vyhrál ImageNet s top-5 chybou 15,3 % (o 10,8 procentních bodů lepší než druhé místo). Konvergence tří faktorů — velké označené datasety, GPU výpočty, vylepšené tréninkové metody (ReLU, dropout) — definitivně ukončila AI zimu.

---

## 2. Biologická inspirace

### Biologický neuron

- **Dendrity** — přijímají signály od jiných neuronů (1 000–10 000 spojení)
- **Soma (tělo buňky)** — integruje vstupy
- **Axon** — přenáší akční potenciál
- **Synapse** — spojení s **synaptickou plasticitou** (Hebbovo pravidlo: *„neurons that fire together, wire together"*)
- **Neurotransmitery** — více než 100 typů
- Lidský mozek: ~86 miliard neuronů, ~10¹⁴ synapsí, ~20 W spotřeba

### Umělý neuron (matematický model)

Mnohem jednodušší abstrakce:

```
y = f(Σ w_i · x_i + b) = f(w·x + b)
```

### Hlavní rozdíly

| Vlastnost | Biologický neuron | Umělý neuron |
|-----------|-------------------|--------------|
| Signál | Spike (akční potenciál) | Spojité reálné číslo |
| Učení | Synaptická plasticita v reálném čase | Backpropagation, jen při tréninku |
| Energie | ~20 W celý mozek | Tisíce W na GPU farmu |
| Rychlost přenosu | ~100 m/s | Rychlost elektriky |
| Asynchronní | Ano | Ne (typicky synchronní) |
| Dendritické výpočty | Ano (komplexní) | Ne |

---

## 3. Perceptron — matematická definice a omezení

### Definice

```
y = sign(w·x + b) = sign(Σ_{i=1}^{n} w_i · x_i + b)
```

Výstup je `+1` nebo `−1` (nebo `0`/`1` v binární variantě).

### Učící pravidlo

Pro chybný vzorek `(x, y)`:

```
w ← w + η · (y − ŷ) · x
b ← b + η · (y − ŷ)
```

kde `η` je learning rate.

### Geometrická interpretace

Perceptron rozděluje vstupní prostor **nadrovinou** `w·x + b = 0`. Umí klasifikovat pouze **lineárně separovatelné** problémy.

### XOR problém

| x₁ | x₂ | XOR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

Žádná přímka v 2D nemůže oddělit body `(0,0), (1,1)` od `(0,1), (1,0)`. Řešení: **vícevrstvý perceptron** se skrytou vrstvou.

---

## 4. Multilayer Perceptron (MLP)

### Architektura

MLP se skládá ze:
1. **Vstupní vrstvy** (input layer)
2. **Jedné nebo více skrytých vrstev** (hidden layers)
3. **Výstupní vrstvy** (output layer)

Každý neuron je plně propojen se všemi neurony v předchozí vrstvě (fully connected).

### Forward pass — matematicky

Pro síť s L vrstvami, kde `a^(0) = x`:

```
z^(l) = W^(l) · a^(l−1) + b^(l)
a^(l) = σ^(l)(z^(l))
```

**Konkrétně pro MLP s jednou skrytou vrstvou:**

```
H = σ(X·W^(1) + b^(1))     # skrytá vrstva
O = H·W^(2) + b^(2)         # výstup (logits)
ŷ = softmax(O)              # pro klasifikaci
```

### Dimenze

- `X ∈ ℝ^{n×d}` (n vzorků, d features)
- `W^(1) ∈ ℝ^{d×h}` (h neuronů ve skryté vrstvě)
- `b^(1) ∈ ℝ^{h}`
- `W^(2) ∈ ℝ^{h×k}` (k tříd)
- `b^(2) ∈ ℝ^{k}`

### Hluboké MLP

Skládáním vrstev dostáváme expresivnější modely:

```
a^(1) = σ_1(W^(1)·x + b^(1))
a^(2) = σ_2(W^(2)·a^(1) + b^(2))
...
a^(L) = σ_L(W^(L)·a^(L−1) + b^(L))
```

---

## 5. Aktivační funkce

Aktivační funkce zavádějí **nelinearitu** — bez nich by byla síť pouze složením lineárních zobrazení (tedy také lineární).

### Sigmoid (logistická)

```
σ(x) = 1 / (1 + e^(−x))
σ'(x) = σ(x)(1 − σ(x))
```

- **Rozsah:** (0, 1)
- **Vlastnosti:** hladká, diferencovatelná, interpretovatelná jako pravděpodobnost
- **Problémy:** saturace (vanishing gradient — maximum derivace je 0,25 v bodě x=0), výstup necentrovaný kolem 0
- **Použití:** výstupní vrstva binární klasifikace

### Tanh (hyperbolický tangens)

```
tanh(x) = (e^x − e^(−x)) / (e^x + e^(−x)) = 2σ(2x) − 1
tanh'(x) = 1 − tanh²(x)
```

- **Rozsah:** (−1, 1)
- Centrovaný kolem nuly (výhoda oproti sigmoidu)
- Maximum derivace: 1.0
- Stále trpí vanishing gradientem pro velké |x|

### ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)
ReLU'(x) = 1 pro x > 0, 0 pro x < 0
```

- **Rozsah:** [0, ∞)
- **Výhody:** nesatuuje pro pozitivní vstupy, výpočetně velmi levná, urychluje konvergenci ~6× oproti sigmoidu
- **Problém „dying ReLU":** pokud neuron dostane velký negativní vstup, gradient bude 0 a neuron přestane učit

### Leaky ReLU

```
LeakyReLU(x) = x pro x ≥ 0, α·x pro x < 0   (typicky α = 0.01)
```

Řeší problém umírajících neuronů povolením malého negativního sklonu.

### GELU (Gaussian Error Linear Unit)

```
GELU(x) = x · Φ(x) = 0.5 · x · (1 + erf(x/√2))
```

kde Φ je kumulativní distribuční funkce standardního normálního rozdělení. Tanh aproximace:

```
GELU(x) ≈ 0.5x · (1 + tanh(√(2/π)(x + 0.044715x³)))
```

Hladší než ReLU, používaná v **GPT, BERT, Vision Transformers** (Hendrycks & Gimpel, 2016, arXiv:1606.08415).

### SwiGLU

Kombinace **Swish/SiLU** s **Gated Linear Unit**:

```
Swish(x) = x · σ(x)    (SiLU = Swish při β=1)
SwiGLU(x, W, V) = Swish(xW) ⊗ (xV)
```

kde ⊗ je element-wise násobení. Rozděluje feed-forward vrstvu na **gate path** (Swish) a **value path** (lineární). Používáno v **LLaMA 2/3, PaLM, Mistral**, často dosahuje lepší perplexity než GELU/ReLU.

### Softmax

```
softmax(z)_i = e^(z_i) / Σ_{j=1}^{K} e^(z_j)
```

- Převádí vektor logitů na pravděpodobnostní distribuci (suma = 1)
- **Numerická stabilita:** typicky se odečítá max — `softmax(z − max(z))`
- Při kombinaci s cross-entropy: gradient je krásně jednoduchý — `ŷ_j − y_j`
- **Použití:** výstupní vrstva multi-class klasifikace

### Doporučení podle architektury (2026)

| Architektura | Aktivační funkce |
|--------------|------------------|
| Klasické CNN | ReLU |
| Transformery (GPT, BERT) | GELU |
| Moderní LLM (LLaMA, Mistral) | SwiGLU |
| Vision modely (EVA-02) | SiLU/SwiGLU |
| Výstup binární klasifikace | Sigmoid |
| Výstup multi-class | Softmax |

---

## 6. Univerzální aproximační teorém

### Cybenkova verze (1989)

> Nechť σ je libovolná spojitá sigmoidální funkce. Pak konečné lineární kombinace ve tvaru:
>
> `G(x) = Σ_{j=1}^{N} α_j · σ(w_j^T·x + b_j)`
>
> jsou husté v `C(I_n)` (prostoru spojitých funkcí na `[0,1]^n` s topologií rovnoměrné konvergence).

### Hornikova generalizace (1991)

Kurt Hornik (s Stinchcombem a Whitem) dokázal pomocí **Stone-Weierstrassovy věty**, že **multilayer feedforward síť s alespoň jednou skrytou vrstvou** je univerzálním aproximátorem pro libovolnou borelovskou měřitelnou funkci, pokud má dostatek skrytých neuronů. Hornik dále ukázal, že **architektura sama, nikoliv konkrétní volba aktivační funkce**, je tím, co dává sítím univerzalitu.

### Význam

Teoreticky: jednovrstvá síť s dostatečným počtem neuronů a nelineární aktivací může aproximovat libovolnou spojitou funkci s libovolnou přesností.

### Kritická omezení

1. **Existenční důkaz** — zaručuje pouze existenci, neposkytuje konstruktivní metodu
2. **Nestanoví velikost sítě** potřebnou pro danou přesnost (může být exponenciální)
3. **Nezaručuje trénovatelnost** — SGD nemusí najít optimální parametry
4. **Neříká nic o generalizaci** na nových datech
5. **Hluboké sítě jsou exponenciálně efektivnější** — hloubka > šířka pro mnoho funkcí (Telgarsky 2016, Eldan & Shamir 2016)

---

## 7. Hluboké učení vs. tradiční ML

### Tradiční ML (SVM, Random Forest, XGBoost)

- **Manuální feature engineering** — data scientist + doménový expert ručně volí proměnné
- Funguje na **menších datasetech** (~50–100 datapointů na feature)
- **Interpretovatelné** (důležité ve financích, zdravotnictví, právu)
- Rychlejší trénink, méně výpočetně náročné
- Lepší pro **strukturovaná tabulková data**

### Deep Learning

- **Automatické učení reprezentací** — první vrstvy se učí jednoduché featury (hrany, barvy), hlubší kombinují do abstraktních konceptů
- Vyžaduje **velké datasety** (tisíce datapointů na feature)
- **Black-box** charakter (omezená interpretovatelnost)
- Excelentní pro **nestrukturovaná data:** obrázky, text, audio, video
- Vyžaduje GPU/TPU

### Kdy zvolit neuronové sítě (2026)

- Komplexní nestrukturovaná data (obrázky, NLP, audio)
- Velké množství tréninkových dat
- Maximální přesnost > interpretovatelnost
- Generative AI úlohy
- Transfer learning z předtrénovaných modelů

### Kdy preferovat tradiční ML

- Tabulková data s méně než ~10k řádky (XGBoost často vítězí na Kaggle)
- Nutnost vysvětlit predikci
- Omezené výpočetní zdroje
- Real-time inference s extrémně nízkou latencí

---

## 8. Matematické základy

### Lineární algebra

- **Vektor** `x ∈ ℝ^n` — reprezentace dat
- **Matice** `W ∈ ℝ^{m×n}` — parametry vrstvy
- **Maticové násobení:** `(AB)_{ij} = Σ_k A_{ik}·B_{kj}` — jádro forward passu
- **Tenzor** — zobecnění matic do vyšších dimenzí (např. RGB obrázek je tenzor `H×W×3`)
- **Skalární součin** `w·x = Σ w_i·x_i` — základ neuronu
- **Hadamardův součin** `A ⊙ B` — element-wise násobení (GLU, LSTM)
- **Normy** `||x||_2 = √(Σx_i²)` — pro regularizaci

### Diferenciální počet

- **Parciální derivace** `∂f/∂x_i`
- **Gradient** `∇f = [∂f/∂x_1, ..., ∂f/∂x_n]^T` — vektor směřující do největšího nárůstu
- **Jakobián** — matice prvních parciálních derivací vektorové funkce
- **Hessián** — matice druhých derivací (Newton, second-order optimization)

### Řetězové pravidlo

Pro složenou funkci `f(g(x))`:

```
df/dx = (df/dg) · (dg/dx)
```

Pro neuronovou síť s loss `L`, `z = Wx + b`, `a = σ(z)`:

```
∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
```

### Backpropagation

Algoritmus efektivního výpočtu gradientů přes celou síť:

1. **Forward pass** — spočti aktivace všech vrstev, ulož `z^(l), a^(l)`
2. **Backward pass** (od výstupní vrstvy zpět):
   ```
   δ^(L) = ∇_a L ⊙ σ'(z^(L))
   δ^(l) = ((W^(l+1))^T · δ^(l+1)) ⊙ σ'(z^(l))
   ∂L/∂W^(l) = δ^(l) · (a^(l−1))^T
   ∂L/∂b^(l) = δ^(l)
   ```
3. **Update:** `W ← W − η · ∂L/∂W`

Backpropagation je v podstatě efektivní aplikace řetězového pravidla (poprvé zapsaného Leibnizem v roce 1676), znovuobjevená pro neuronové sítě v práci Rumelharta, Hintona a Williamse (1986).

---

## 9. Loss funkce

Loss funkce kvantifikuje rozdíl mezi predikcí `ŷ` a skutečnou hodnotou `y`. Optimalizace = minimalizace průměrné loss přes dataset.

### Mean Squared Error (MSE)

```
MSE = (1/n) · Σ_{i=1}^{n} (y_i − ŷ_i)²
```

- **Použití:** regrese (predikce spojité hodnoty)
- **Vlastnosti:** konvexní pro lineární modely, hladká, snadno diferencovatelná
- **Derivace:** `∂MSE/∂ŷ_i = 2(ŷ_i − y_i)/n`
- **Slabina:** citlivá na outliery (kvadrát zvýrazňuje velké chyby)
- **Alternativy:** MAE (Mean Absolute Error — robustnější), Huber loss (kombinace)

### Cross-entropy loss

**Binární cross-entropy** (s sigmoidem na výstupu):

```
BCE = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
```

**Categorical cross-entropy** (s softmaxem na výstupu):

```
CE = −Σ_{i=1}^{K} y_i · log(ŷ_i)
```

- **Použití:** klasifikace
- **Klíčová vlastnost:** poskytuje **proporcionálně větší gradient** když je predikce daleko od pravdy → rychlejší učení
- **V kombinaci se softmax:** gradient se zjednoduší na `ŷ − y` (numericky stabilní)
- **Informačně-teoretická interpretace:** měří, kolik bitů informace navíc je potřeba k zakódování pravé distribuce, pokud používáme predikovanou distribuci (KL divergence + entropie)

### Hinge loss (SVM-style)

```
L = max(0, 1 − y·ŷ)        (kde y ∈ {−1, +1})
```

- **Použití:** SVM, max-margin klasifikace
- Maximalizuje **margin** kolem rozhodovací hranice
- Není pravděpodobnostní (neoperuje na probabilities)

### Doporučení

| Úloha | Loss funkce | Output activation |
|-------|-------------|-------------------|
| Regrese | MSE / MAE / Huber | Linear |
| Binární klasifikace | Binary cross-entropy | Sigmoid |
| Multi-class klasifikace | Categorical cross-entropy | Softmax |
| Multi-label klasifikace | Binary cross-entropy (per label) | Sigmoid (per label) |
| Max-margin klasifikace | Hinge loss | Linear |
| LLM next-token prediction | Cross-entropy | Softmax |
| GAN | BCE / Wasserstein loss | Sigmoid / Linear |

---

## Doporučené zdroje pro další studium

- **Goodfellow, Bengio, Courville:** *Deep Learning* (deeplearningbook.org), kapitola 6
- **Karpathy:** Neural Networks: Zero to Hero (micrograd, makemore, GPT from scratch)
- **3Blue1Brown:** vizuální série o neuronových sítích a backpropagation
- **Stanford CS231n:** cs231n.github.io
- **Dive into Deep Learning** (d2l.ai) — interaktivní kniha s kódem
- **arXiv:1606.08415** — Hendrycks & Gimpel: GELU paper

---

## Zdroje

- [Perceptron — Wikipedia](https://en.wikipedia.org/wiki/Perceptron)
- [AlexNet — Wikipedia](https://en.wikipedia.org/wiki/AlexNet)
- [Universal approximation theorem — Wikipedia](https://en.wikipedia.org/wiki/Universal_approximation_theorem)
- [Multilayer feedforward networks are universal approximators (Hornik)](https://www.cs.cmu.edu/~epxing/Class/10715/reading/Kornick_et_al.pdf)
- [Multilayer Perceptrons — Dive into Deep Learning](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)
- [Cross-Entropy Loss Function — DataCamp](https://www.datacamp.com/tutorial/the-cross-entropy-loss-function-in-machine-learning)
- [Backpropagation — Wikipedia](https://en.wikipedia.org/wiki/Backpropagation)
- [Softmax function — Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)
- [Gaussian Error Linear Units (GELUs) — arXiv 1606.08415](https://arxiv.org/abs/1606.08415)
- [CS231n — Neural Networks](https://cs231n.github.io/neural-networks-1/)
- [Neural Networks: Zero To Hero — Karpathy](https://karpathy.ai/zero-to-hero.html)
