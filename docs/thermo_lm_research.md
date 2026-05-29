# Termodynamický jazykový model: výzkumná teorie a implementační plán

*Spojitý fyzikální substrát jako náhrada transformeru — od metafory k funkční teorii*

---

## 0. Shrnutí pro netrpělivé

Tvůj nápad — 3D prostor vyplněný plynem, řízený teplotou, který se trénuje dodáváním nerovnoměrné termální energie — **není fantazie bez základu**. Rozpadá se na čtyři dobře prozkoumané komponenty, z nichž každá má fungující literaturu:

1. **Substrát (plyn)** → diferencovatelná mřížková dynamika. Buď differentiable LBM (Lattice Boltzmann), nebo learned neural operator s flux-form zachováním. Obojí existuje a běží na GPU.
2. **Teplota a annealing** → diffusion modely *jsou* doslova termodynamika. Tvoje teplotní metafora je matematicky identická s temperature annealingem v generativních modelech. Tvůj instinkt „začít na absolutní nule" je ale obrácený — viz §4.
3. **Hlava / readout** → reservoir computing. Bohatá nelineární dynamika + trénovaný lineární readout. Reservoir computing jako jazykový model byl publikován v 2025.
4. **Trénovatelnost** → backpropagation through time (BPTT) přes rollout, stabilizovaná difuzí, gradient checkpointingem a flux-form zachováním.

Klíčový poznatek z rešerše: **téměř každý kus tvého nápadu už někdo postavil — ale nikdo je nespojil přesně takhle.** To je dobrá zpráva (je to proveditelné) i varování (musíš vědět, kde jsou miny).

---

## 1. Mapa konceptu na existující paradigmata

| Tvoje komponenta | Etablované paradigma | Klíčová reference |
|---|---|---|
| „Plyn s fyzikálními zákony" | Differentiable LBM / Neural PDE / Neural CA | XLB, Lettuce, Growing NCA |
| „Teplota = absolutní nula → dodávání energie" | Temperature annealing v diffusion modelech | Statistical thermodynamics of diffusion |
| „Nerovnoměrná termální energie" | Lokální energetická injekce / forcing term | XLB corrector force |
| „Jak hlava čte výstup" | Reservoir computing readout | Reservoir Computing as a Language Model |
| Zachování hmoty/energie | Flux-form / Neural Conservation Laws | Neural Finite Volume, Richter-Powell 2022 |
| Trénink přes čas | Stabilized BPTT | StableBPTT (Schnell & Thuerey) |

Tvůj nápad je v podstatě **„reservoir computing, kde reservoir je diferencovatelná termodynamická simulace plynu a readout je trénovaná hlava."** To je legitimní a nezmapovaná pozice ve výzkumném prostoru.

---

## 2. Volba substrátu: tři úrovně poctivosti

Z minulé diskuse, teď podložené literaturou. Klíčové rozhodnutí: **Euler (pole na mřížce), ne Lagrange (částice).** Částicové srážky jsou nediferencovatelné v jádru. Pole jsou tenzory a konvoluce — plně diferencovatelné.

### 2.1 Úroveň A — Differentiable Lattice Boltzmann (fyzikálně nejpoctivější s rozumnou stabilitou)

Tohle je nejlepší kompromis a má hotové reference implementace (XLB v JAX, Lettuce v PyTorch). Stav každého voxelu = distribuční funkce `f_i` (typicky 19 nebo 27 směrů v 3D — D3Q19/D3Q27). Dynamika má dva kroky:

**Streaming (advekce):** `f_i` se posune o jeden voxel ve směru `c_i`. Čistý shift — *přesně* diferencovatelný, žádná interpolační chyba. V kódu je to roll/gather operace.

**Collision (relaxace k rovnováze):**
```
f_i ← f_i − (1/τ)(f_i − f_i^eq(ρ, u))
```
kde `f_i^eq` je hladká polynomiální funkce lokální hustoty `ρ = Σ f_i` a rychlosti `ρu = Σ c_i f_i`, a `τ` je relaxační čas svázaný s viskozitou.

**Tady vstupuje tvoje teplota:** relaxační čas `τ` a/nebo amplitudu šumu v collision kroku řídíš teplotním polem `T(x,t)`. Vysoká teplota = víc explorace; anneal dolů = systém „zamrzá" do struktury.

Proč je to dobré pro tebe konkrétně:
- Streaming i collision jsou hladké → plně diferencovatelné, žádný custom backward potřeba.
- Lokální, stencilové, paralelní → mapuje se 1:1 na WGSL compute shader (tvůj wgpu stack).
- LBM je notoricky stabilnější než přímé řešení Navier–Stokes.
- XLB doslova ukazuje, že přes celou LBM simulaci lze backpropagovat, aby síť „pochopila temporální dynamiku toku" — to je přesně tvůj trénink.

**Riziko:** gradienty přes desítky streaming-collision kroků. Řeš gradient checkpointingem (§5).

### 2.2 Úroveň B — Learned Neural Operator s fyzikálním biasem (nejtrénovatelnější — KDE BYS MĚL ZAČÍT)

Buď k sobě upřímný: cíl není simulovat plyn, ale mít *induktivní bias* plynu — lokalitu, zachování, difuzi, řízení teplotou. Tak nevnucuj přesné rovnice, ale **architekturu, která je nemůže porušit.**

Update krok:
```
s_{t+1} = s_t + Δt · N_θ(s_t, T_t)
```
kde `N_θ` je malá konvoluční síť se sdílenými vahami přes prostor, ale s těmito **tvrdými omezeními zabudovanými do architektury, ne do lossu:**

**Lokalita** — jen 3×3×3 konvoluce. Informace se šíří max o jeden voxel za krok (konečná rychlost zvuku). Tohle samo dává plynové chování.

**Zachování hmoty/energie přes flux-form** — tohle je nejdůležitější trik a má rigorózní oporu (Neural Conservation Laws, Richter-Powell et al. 2022; Neural Finite Volume 2025). Síť nepredikuje nový stav, ale **toky mezi sousedy** `F_{x→y}`. Update je:
```
s_{t+1}(x) = s_t(x) + Σ_sousedi [F_in − F_out]
```
Protože co odteče z jednoho voxelu, přesně přiteče do druhého (antisymetrie `F_{x→y} = −F_{y→x}`), celkové množství je **strukturálně** zachováno bez ohledu na váhy. Zachování není trestáno lossem — je nemožné ho porušit. Finite-volume literatura tomu říká „telescoping across cells" a je to jádro toho, proč FV schémata zachovávají hmotu exaktně.

**Difuze a teplota** — explicitní Laplaceův člen `κ(T)·∇²e` přičteš natvrdo, kde `κ` roste s teplotou (konvoluce s diskrétním Laplacian stencilem). Nerovnoměrná injekce energie (tvůj požadavek) = token přičte energii lokálně, ta pak difunduje sama. Difuze je navíc *kontrahující* operátor → stabilizuje gradienty (§5).

**Symetrie** — translační invariance zdarma z konvoluce. Pokud chceš rotační, použij steerable/grupově-ekvivariantní konvoluce.

Tahle úroveň je triviálně diferencovatelná (ConvNet rozvinutý v čase), stabilní (residuální update + difuze), a pořád „je to plyn" ve všech smyslech, na kterých ti záleží.

### 2.3 Úroveň C — Plná diferencovatelná PDE (Navier–Stokes)

Nejhezčí fyzika, nejhorší trénovatelnost. Semi-Lagrangeovská advekce (Stable Fluids) přes `grid_sample` je diferencovatelná, ale advekce + tlak trpí numerickou nestabilitou a gradienty explodují. **Nedoporučuji jako start.** Zvaž jen pokud A i B fungují a chceš poctivější fyziku.

**Verdikt:** Postav Úroveň B jako 2D prototyp. Když přebije bigram baseline, přejdi na 3D a/nebo Úroveň A.

---

## 3. Hlava: jak číst výstup (jádro tvé otázky)

Reservoir computing dává přesný rámec: **bohatá nelineární dynamika (tvůj plyn) + trénovaný readout.** Reservoir Computing as a Language Model (2025) ukazuje pipeline na character-level Shakespearovi: input→reservoir weight matrix `W_in`, fixní reservoir, a readout `W_out ∈ R^{V×N}` mapující stav reservoiru na logity nad slovníkem `V`. To je doslova kostra, kterou potřebuješ. Tři varianty readoutu:

### 3.1 Readout z momentů pole (nejjednodušší, začni tady)
Z LBM/pole spočítáš makroskopické momenty v každém voxelu — hustota `ρ`, rychlost `u`, energie `e` — a coarse-grained (pooling) do nižší dimenze, pak lineární projekce (nebo malá MLP) do logitů. Jednoduché, diferencovatelné, robustní. Ztrácíš část prostorové struktury agregací, ale jako baseline ideální.

### 3.2 Spektrální / modální readout (fyzikálně nejhezčí)
Plyn v dutině má charakteristické vlastní módy (stojaté vlny). Hlava projektuje pole na pár nejnižších módů (3D FFT nebo Laplacian eigenmódy domény) a z koeficientů predikuje. Výhody: low-dimensional, rozumně translačně/rotačně invariantní, a koeficienty nesou globální strukturu, kterou momentový pooling zahodí. Implementačně: `rfftn` na poli energie, vezmi K nejnižších frekvencí, flatten → lineární vrstva.

### 3.3 Flux readout přes hranici (reservoir-computing nejčistší)
Definuješ výstupní hranici domény (stěna s „detektory") a čteš jen tok energie skrz ni. Připomíná to fyzické reservoir computing (optika, mechanika), kde čteš jen okrajový stav. Méně informace, ale nejlevnější a nejvíc „fyzikální".

**Doporučení:** Začni 3.1 (momenty + lineární). Jakmile máš signál, zkus přidat 3.2 (spektrum) jako další feature kanály — empiricky bývá modální readout silnější pro periodické/strukturované signály.

### 3.4 Jak vůbec vstupuje token (encoder)
Symetricky k readoutu. Token (nebo jeho embedding) se vstříkne jako **lokalizovaná injekce energie/hybnosti** do pole — buď do fixní oblasti („vstupní tryska"), nebo embedding modeluje prostorový profil injekce. To je `W_in` z reservoir computingu, ale fyzikálně realizované jako forcing term (přesně jako XLB „corrector force" `F_nn` přes perturbaci rovnovážné distribuce). Autoregresivně: po readoutu navzorkuješ token, ten vstříkneš jako další injekci, simuluješ K kroků, čteš znovu.

---

## 4. Teplota a annealing: oprava tvého instinktu

Tvůj návrh: start na absolutní nule, trénink dodává energii. **Fyzikálně intuitivní, ale obráceně k tomu, co funguje.**

Proč: nulová teplota = nulová entropie = nulová stochasticita = **nulový gradient k dispozici pro učení**, a zároveň žádná difuze → žádná kontrakce → chaotická, nekontrahující dynamika → exploze gradientu při BPTT. Mrtvý start z obou stran.

Co dělá literatura (a co funguje): **začni HORKÝ, anneal dolů.** Tahle intuice je páteří celé třídy modelů:
- **Diffusion modely** jsou doslova termalizace→denoising. Statistická termodynamika generativních diffusion modelů ukazuje, že mají **fázové přechody, spontánní narušení symetrie a kritickou nestabilitu** přesně jako spinový systém chlazený přes Curie–Weissův bod. Vysoká teplota = jedno stabilní řešení (chaos/šum); chlazení = „rozštěpení" do strukturovaných mód (paměť/vzory). To je Hopfieldova/Boltzmannova asociativní paměť.
- **Simulated annealing**, **parallel tempering**, **PITA** (Progressive Inference-Time Annealing, 2025) — všechny jdou z vysoké teploty dolů, protože vysoká teplota *míchá módy a snižuje energetické bariéry*, zatímco chlazení vybírá strukturu.

**Reinterpretace tvého nápadu, která zachovává ducha:** Teplota není to, co dodáváš tréninkem globálně. Teplota je **schedule** (klesající přes inference/generování) NEBO **lokální, datově řízená veličina** (token vstříkne teplo lokálně, ono difunduje). Trénink upravuje váhy `θ`, které řídí, *jak* se energie injektuje, difunduje a relaxuje — ne absolutní teplotu systému.

Konkrétní teplotní schedule pro tvůj model:
- **Globální annealing** přes generování jednoho výstupu: `T(step) = T_max · (1 − step/K)^α`. Start horký (explorace), konec studený (commit na token).
- **Lokální teplota jako kanál:** `T(x,t)` je 1 kanál stavu, token do něj přičte, difunduje s `κ(T)`, řídí stochasticitu collision/sampling lokálně.

Doporučuji obojí kombinovat: lokální teplotní pole jako stav + globální anneal jako modulátor amplitudy injekce.

---

## 5. Trénovatelnost: hlavní technické riziko a jeho řešení

Ať zvolíš substrát A nebo B, rozvíjíš dynamiku na K kroků a backpropuješ skrz celý rollout (BPTT). Tohle je nejpravděpodobnější místo selhání. Literatura dává konkrétní obrany — použij všechny:

**1. Difuze jako stabilizátor (zdarma z fyziky).** Difuzní/relaxační členy *kontrahují* stav. Gradientová stabilita unrolled simulací je svázaná s Lyapunovým časem systému — kontrahující dynamika má krátký efektivní horizont a krotké gradienty. Proto je teplotně řízená difuze nejen feature, ale i nutný stabilizátor. Čím horčí start, tím silnější difuze, tím krotší gradienty.

**2. Gradient checkpointing.** Neukládáš všech K mezistavů pro backward, přepočítáš je. Pro 3D mřížku nutnost — jinak `L³ × kanály × K` sežere VRAM. (Tvůj Rust/wgpu stack to zvládne ručně; v prototypu v PyTorch/JAX je to `torch.utils.checkpoint` / `jax.checkpoint`.)

**3. Stabilized BPTT (StableBPTT, Schnell & Thuerey 2024).** Klíčový poznatek: fyzikální simulátory mají *vyvážený* gradient flow, na rozdíl od neuronových sítí. Jejich metoda modifikuje backward pass (gradient stopping + komponentové porovnání) tak, že nemění polohu minim, ale odstraní explozi/mizení. Implementovatelné jako sekvence stop-gradient operací. Pro tvůj případ kriticky relevantní.

**4. Kratší rollouty + curriculum.** Trénuj nejdřív na K=4–8 krocích, postupně prodlužuj. Unrolling na příliš mnoho kroků má diminishing returns a může i škodit (doloženo). Najdi minimální K, který dá dost „výpočetní hloubky" na predikci.

**5. Truncated BPTT.** Pokud autoregresivně generuješ dlouhé sekvence, useknI gradient po pevném okně, jako u RNN.

---

## 6. Konkrétní implementační plán (fázovaný)

### Fáze 0 — 2D toy prototyp (důkaz konceptu, levná smrt nápadu)
- **Substrát:** Úroveň B (learned neural operator, flux-form), 2D mřížka 32×32, ~5–8 kanálů (ρ, u_x, u_y, e, T + pár skrytých).
- **Dynamika:** 3×3 konvoluce predikující toky mezi sousedy (antisymetrické → zachování) + explicitní Laplacian difuze řízená T kanálem.
- **Injekce:** token embedding → lokalizovaná injekce do středu mřížky.
- **Readout:** momenty + global average pooling + lineární → logity (varianta 3.1).
- **Trénink:** K=4–8 kroků, BPTT, gradient checkpointing, character-level na malém korpusu (enwiki8 prefix nebo i Shakespeare jako v RC paperu).
- **Baseline k přebití:** bigram a malý RNN. **Pokud nepřebije bigram, nápad umírá tady — levně.**
- **Stack:** PyTorch nebo JAX. NE rovnou Rust/wgpu — chceš rychlou iteraci a hotový autodiff. Rust až po validaci.

### Fáze 1 — Stabilizace a readout ablace
- Přidej spektrální readout (3.2) jako další featury, porovnej.
- Přidej globální annealing schedule přes generování.
- Zapni StableBPTT trik, změř, jestli dovolí delší K.
- Ablace: flux-form vs. volný update (ukáž, že zachování pomáhá), difuze on/off (ukáž stabilizační efekt).

### Fáze 2 — 3D a/nebo LBM substrát
- Přepni na 3D mřížku NEBO na differentiable LBM (Úroveň A) — referenci ber z XLB/Lettuce.
- Tady teprve dává smysl Rust + wgpu/WGSL: streaming = roll shader, collision = lokální stencil shader, difuze = Laplacian stencil. Vše paralelní, mapuje se ideálně na compute shadery.
- Pozor na VRAM: 3D `L³` roste rychle. Začni `L=16` nebo `L=24`.

### Fáze 3 — Škálování a srovnání
- Větší korpus, srovnání perplexity vs. malý transformer stejné velikosti.
- Realistické očekávání (viz §7).

---

## 7. Realistická očekávání a rizika

**Co říká RC-as-LM paper:** transformery vyhrávají v kvalitě predikce, reservoir computing je *výrazně efektivnější* v tréninku/inferenci. Tvůj model bude pravděpodobně podobný profil — nečekej, že přebiješ transformer na perplexity. Cíl je: (a) funguje vůbec? (b) je v něčem zajímavě jiný — efektivita, interpretovatelnost (můžeš se „dívat" na teplotní pole), induktivní biasy, kontinuální latentní dynamika?

**Hlavní rizika, sestupně podle pravděpodobnosti:**
1. **Gradienty přes rollout** — nejpravděpodobnější killer. Mitigace v §5. Difuze + checkpointing + krátké K + StableBPTT.
2. **Readout bottleneck** — pooling zahodí moc informace. Mitigace: spektrální readout, víc detektorů.
3. **Expresivita** — lokální 3×3 konvoluce za K kroků má omezený receptivní field (max K voxelů). Pro dlouhé závislosti potřebuješ buď velké K, nebo multi-scale (coarse grid kanál), nebo dovolit „rychlou" globální módu. Tohle je analogie attention vs. konvoluce — promysli, jak plyn přenáší informaci na dálku (vlny? tlak?).
4. **Diskrétní vs. spojité** — text je diskrétní, plyn spojitý. Injekce a sampling jsou mosty; můžou být ztrátové.

**Zajímavá konexe k prozkoumání:** „Training Language Models via Neural Cellular Automata" (2025) zjistili, že 164M tokenů automat-dat přebije 1.6B tokenů přirozeného jazyka pro budování obecných reprezentací, protože každý token nutí model odvodit latentní pravidlo z kontextu. Tvůj plyn je spojitý bratranec NCA — možná má podobnou sample-efficiency výhodu. Stojí za měření.

---

## 8. Klíčové reference podle tématu

**Differentiable fluid/LBM substrát:**
- XLB: A Differentiable Massively Parallel Lattice Boltzmann Library (arXiv 2311.16080) — JAX, backprop přes celou LBM, corrector force jako injekce.
- Lettuce: PyTorch-based Lattice Boltzmann (arXiv 2106.12929) — stream-and-collide kompatibilní s ML.

**Learned update rule / neural CA:**
- Growing Neural Cellular Automata (Distill 2020) — diferencovatelné update rule, BPTT, per-variable L2 normalizace gradientů proti nestabilitě.
- Training Language Models via Neural Cellular Automata (2025) — NCA trajektorie jako pretraining data pro LM, sample efficiency.

**Zachování zákonů ve struktuře:**
- Neural Conservation Laws: A divergence-free perspective (Richter-Powell et al. 2022) — zachování by design.
- (U)NFV: Neural Finite Volume (arXiv 2505.23702) — learned flux, exaktní zachování, 10× nižší chyba než Godunov.
- Learning Under Laws: Constraint-Projected Neural PDE (arXiv 2511.03578) — projekce na fyzikální manifold.

**Termodynamika a annealing:**
- Statistical Thermodynamics of Generative Diffusion Models (arXiv 2310.17467 / Entropy 2025) — fázové přechody, narušení symetrie, Hopfield konexe.
- Energy-Based Diffusion Language Models (arXiv 2410.21357) — EBM jazykový model.
- Progressive Inference-Time Annealing / PITA (arXiv 2506.16471) — high→low temperature schedule, Boltzmann densities.

**Reservoir computing / readout:**
- Reservoir Computing as a Language Model (2025) — `W_in`, fixní reservoir, `W_out` readout do slovníku; srovnání s transformery na char-level.
- Recent Advances in Physical Reservoir Computing (IBM review) — substrát-nezávislost, fyzikální reservoiry.

**Stabilita BPTT:**
- Stabilizing Backpropagation Through Time to Learn Complex Physics / StableBPTT (arXiv 2405.02041) — vyvážený gradient flow simulátorů, gradient stopping.
- Differentiability in Unrolled Training of Neural Physics Simulators (arXiv 2402.12971) — gradientová stabilita ~ Lyapunův čas, diminishing returns dlouhých rolloutů.

---

## 9. Jedna věta na závěr

Postav **Úroveň B ve 2D s flux-form zachováním, lokální teplotou jako kanálem, momentovým readoutem, K=4–8 kroky a gradient checkpointingem v PyTorch/JAX** — a nech 2D toy verzi rozhodnout, jestli má smysl investovat do 3D LBM v Rustu. Tvoje teplotní intuice je správná, jen obrácená: horko na startu, chlazení k cíli.
