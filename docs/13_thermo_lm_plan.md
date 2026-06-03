# Termodynamický jazykový model — implementační plán

*Aurora research line B. Cíl: postavit model z `docs/thermo_lm_research.md`
přesně podle teorie (Úroveň B, 2D, flux-form zachování, lokální teplota
jako kanál, momentový readout, K=4–8 kroků, BPTT), ale na existující
Aurora infrastruktuře (Rust, ruční autodiff, AdamW/Muon, JSONL, checkpoint,
wgpu, Bevy viz) — **včetně GUI**.*

Aktivní transformer plán zůstává `docs/12`. Tohle je samostatná modelová
linie, proto nový dokument (12 je 31 KB, ne kvůli velikosti — kvůli tomu,
že jde o jiný model). Sprinty této linie číslujeme `T0, T0.5, T1, …`.

---

## 0. Vztah k výzkumnému dokumentu (co znamená „přesně podle")

`thermo_lm_research.md` definuje model jako **reservoir computing, kde
reservoir je diferencovatelná termodynamická simulace plynu a readout je
trénovaná hlava**. Konkrétní doporučení (§0, §6 Fáze 0, §9):

> Úroveň B ve 2D, flux-form zachování, lokální teplota jako kanál,
> momentový readout, K=4–8 kroků, gradient checkpointing; horko na startu,
> chlazení k cíli.

Tenhle plán je k té specifikaci **věrný v návrhu simulace** (§11 je
auditní mapovací tabulka požadavek → artefakt). **Jediná vědomá odchylka**
je stack: dokument doporučuje prototypovat ve **PyTorch/JAX** (rychlá
iterace, autodiff zdarma) a do Rustu portovat až po validaci (§6 „Stack",
§9). Zadání ale zní *využít existující infrastrukturu* = Aurora/Rust. Tu
odchylku řešíme vědomě — viz §3.

---

## 1. Co z Aurora infrastruktury znovupoužijeme (a co je nové)

Mapování proběhlo čtením zdrojů. Reuse je vysoký — termo-model je
**paralelní sestava** vedle transformeru, ne jeho úprava.

| Komponenta | Soubor | Reuse | Poznámka |
|---|---|---|---|
| Optimizer AdamW + Muon | `src/optimizer.rs` | **100 %** | `AdamGroup`/`MuonGroup` + `adamw_apply`/`muon_apply` jsou per-tensor generické — termo params zapojíme 1:1 |
| Cross-entropy + softmax | `src/transformer/loss.rs` | **100 %** | `cross_entropy_with_grad`, `softmax`, `argmax` |
| LR schedule | `src/transformer/schedule.rs` | **100 %** | `cosine_warmup` |
| Lineární projekce + LayerNorm | `src/attention.rs` | **100 %** | `LinearProj`/`linear_forward`/`linear_backward`, `LayerNorm` — pro readout/encoder |
| GELU | `src/linalg.rs` | **100 %** | `gelu`/`gelu_backward` — nelinearita conv operátoru |
| Embedding scatter pattern | `src/transformer/embedding.rs` | **vzor** | injekční encoder = `[vocab × C]` tabulka, backward = scatter-add (přesně `embed_backward`) |
| BPE tokenizer | `src/bpe.rs` | **100 %** | `train/load/save/encode/decode/vocab_size` |
| Korpus I/O + BPB + sampling | `src/text_io.rs` | **100 %** | `load_corpus`, `bits_per_byte`, `sample_token` |
| JSONL metriky | `examples/aurora_train.rs` (`MetricsWriter`) | **lift do lib** | viz §5.4 |
| Checkpoint binární formát | `src/checkpoint.rs` | **vzor** | nový magic `THERMO01`, stejný styl |
| Gradient-check harness | `tests/transformer_primitives.rs` | **vzor** | central diff `EPS=1e-3`, `TOL=5e-2`, dot-trik s náhodným upstreamem |
| GPU kernel infra | `src/gpu/*` | **vzor + reduce/elementwise/optim 100 %** | Fáze T2; reductions, elementwise, AdamW/Muon kernely, parity harness reusable |
| Bevy viz scaffolding | `src/viz/*` | **model-agnostické části 100 %** | kamera, Text2d panely, loss gizmo, app setup; heatmapy nové (§6) |

**Co je čistě nové:** termo-primitiva (conv, flux/divergence, difuze,
injekce, readout), rollout/BPTT assembly, termo checkpoint, termo viz mód,
`examples/thermo_train.rs`.

---

## 2. Architektura termo-modelu (Úroveň B, 2D)

### 2.1 Stav

Pole `s ∈ R^{H×W×C}`, row-major flat `Vec<f32>` délky `H·W·C` (default
H=W=32). Kanály: **fyzikální** `[ρ, u_x, u_y, e, T]` (C_phys = 5) +
**skryté** `h_1..h_{C_hid}` (default C_hid = 3) ⇒ **C = 8**. `T` je kanál
*lokální teploty* (§4 dokumentu).

### 2.2 Update krok (jeden sub-step) — `s_{t+1} = s_t + Δt·N_θ(s_t)`

Krok skládá čtyři operace přesně podle §2.2 dokumentu. Pořadí: conv →
flux-divergence (zachování) → difuze (stabilizace) → injekce (jen na
začátku tokenu, ne každý sub-step).

**(a) Lokální operátor `N_θ` — lokalita 3×3 (§2.2 „Lokalita").**
Dvojvrstvý konvoluční operátor se sdílenými vahami přes prostor:
```
raw = Conv1x1( GELU( Conv3x3(s) ) )      # 3×3 dává konečnou rychlost zvuku, 1×1 míchá kanály
```
`Conv3x3: C → C_h` (`W1 [C_h×C×3×3]`, `b1 [C_h]`), `Conv1x1: C_h → C_out`
(`W2 [C_out×C_h]`, `b2 [C_out]`). Translační invariance zdarma ze sdílení
vah. Okraj: zero-pad (= „za stěnou nic").

**(b) Flux-form zachování (§2.2 „Zachování", nejdůležitější trik).**
`C_out` obsahuje pro každý **konzervovaný** kanál `c ∈ C_cons` (default
`{ρ, e}`) dvě hodnoty: směrové hranové toky
- `Fx[i,j,c]` = tok z buňky (i,j) do (i,j+1) — pravá hrana,
- `Fy[i,j,c]` = tok z (i,j) do (i+1,j) — horní hrana.

Update konzervovaného kanálu (closed-wall: toky mimo mřížku = 0):
```
s_new[i,j,c] = s[i,j,c] + (Fx[i,j-1,c] - Fx[i,j,c]) + (Fy[i-1,j,c] - Fy[i,j,c])
```
Každý vnitřní tok vstoupí jednou s `+` (přítok souseda) a jednou s `−`
(vlastní odtok) → suma telescopuje na okraj → `Σ s_new = Σ s` **exaktně,
nezávisle na vahách**. To je „telescoping across cells" z FV literatury;
zachování není v lossu, je strukturálně nemožné ho porušit. Antisymetrie
`F_{x→y} = −F_{y→x}` je implicitní (jeden tok na hranu).

Pro **nekonzervované** kanály (`u_x,u_y,T,h_*`) volný reziduální update
`s + Δt·g_θ(s)` (zbytek `C_out`). Volba conserve/free per-kanál je
konfigurovatelná (ablace flux-form on/off, §9 Fáze T1).

**(c) Teplotně řízená difuze (§2.2 „Difuze", §5.1 stabilizátor).**
5-bodový Laplacian (zero-flux okraj ⇒ chybějící soused = vlastní hodnota,
tím se i difuze stává konzervativní):
```
Lap[i,j,c] = s[N]+s[S]+s[E]+s[W] - 4·s[i,j,c]
s_new[i,j,c] += κ_c(T[i,j]) · Lap[i,j,c]
κ_c(T) = κ0_c · sigmoid(a_c + b_c·T[i,j])      # roste s T, kladná, omezená
```
Params `a_c, b_c` (a `κ0_c`) per kanál — pár skalárů. Difuze kontrahuje →
krotí BPTT gradienty (§5.1).

**(d) Injekce tokenu — encoder (§3.4).** Jednou na začátku zpracování
tokenu (před K sub-kroky), ne každý sub-step:
```
s[i,j,c] += anneal · profile[i,j] · inj[token, c]
```
`profile` = fixní Gaussovský „vstupní trysek" ve středu (bez params),
`inj [vocab × C]` = učená tabulka (to je `W_in` z reservoir computingu;
backward = scatter-add do řádku tokenu, přesně `embed_backward`).
`anneal` = volitelný globální modulátor amplitudy (§4, default 1.0 při
tréninku, schedule při generování).

### 2.3 Readout — moment + pooling + lineární (§3.1, „začni tady")

Po K sub-krocích z finálního pole čteme logity:
```
pooled = AvgPool(s, P×P)        # H×W×C → P×P×C  (default P=4 ⇒ 4·4·8 = 128 feat)
logits = Linear(flatten(pooled))    # 128 → vocab   (LinearProj, +volitelně MLP s GELU)
```
AvgPool backward = uniformní rozdělení gradientu do okna; Linear backward
= `linear_backward`. Jednoduché, robustní, plně diferencovatelné.

**Spektrální readout (§3.2)** přidáme jako ablaci ve Fázi T1 *bez FFT
závislosti*: K nejnižších módů = **fixní předpočítaná reálná DFT bázová
matice** `B [K_modes × H·W]`; modální koeficienty = `B @ energy_channel`
(jen matmul, triviálně diferencovatelné). Konkatenujeme jako další feature
kanály k momentům.

### 2.4 Rollout přes sekvenci — BPTT (jádro odlišnosti od transformeru)

Transformer zpracuje všech `T` pozic paralelně. Termo-model je
**rekurentní fyzikální simulace**: jedno pole se vyvíjí, na každé pozici
injektujeme token, simulujeme K kroků, čteme logity pro predikci dalšího
tokenu. Teacher-forced trénink přes okno délky `T`:
```
field = init_field()                       # warm start: T-kanál vysoko (§4 „horko na startu")
for t in 0..T:
    field = inject(field, tokens[t])
    for k in 0..K:
        anneal_k = T_max·(1 - k/K)^α        # per-token anneal hot→cold (§4)
        field = step(field, anneal_k)
    logits[t] = readout(field)             # predikuje tokens[t+1]
loss = mean_t CE(logits[t], tokens[t+1])
```
Backprop jde pozpátku celým `T·K` unrollem. **Fáze T0: pole se resetuje
per okno** (okna nezávislá, jako teď transformer) — čistě ohraničí paměť.
Cross-window carry (pravé TBPTT, §5.5) je refinement Fáze T1.

**Receptivní pole:** lokální 3×3 za K kroků vidí max K buněk; přes okno
nese „paměť" carry pole. Pozor na expresivitu (§7 dokumentu, riziko 3) —
adresujeme multi-scale/spektrálním readoutem ve Fázi T1.

---

## 3. Klíčové rozhodnutí: ruční backward + odchylka od PyTorch-first

Aurora **nemá generický autodiff tape**. Každé primitivum má vzor:
`*Weights` + `*Grads` (`zero`/`zeros`/`add_assign`/`scale`/`l2_norm`) +
`*Cache` + `*_forward(w, x) -> (y, cache)` + `*_backward(w, cache, dy,
grads) -> dx`. Model je orchestruje (forward dopředu s cachí, backward
pozpátku) — viz `transformer/model.rs`.

**Důsledek:** každá nová termo-operace potřebuje *ruční forward i backward*
a finite-difference gradient check. To je hlavní cena Rust cesty. Je
**zvládnutelná**, protože:
- flux-divergence a Laplacian jsou **lineární** operátory → backward je
  jejich adjoint (gather/stejný stencil), triviální a numericky stabilní;
- conv backward je standardní vzor (jako `linear_backward`, jen se
  sdílením přes prostor);
- injekce backward = scatter-add (kopie `embed_backward`);
- readout = `linear_backward` + uniformní pooling backward.

**Odchylka od dokumentu (§6 „Stack", §9):** dokument radí prototyp v
PyTorch/JAX a Rust až po validaci. Jdeme rovnou do Rustu (zadání: existující
infra). Výhody: GPU, viz, trénink, checkpoint, metriky **zdarma**; PyTorch
cesta by je stavěla znovu. Nevýhoda: pomalejší iterace, víc upfront práce
na primitivech + gradient checkech.

> **Volitelný de-risk checkpoint (doporučeno zvážit):** než napíšeš ruční
> backward pro 6 operací, ~1 den v PyTorch ověř, že Úroveň B ve 2D vůbec
> přebije bigram na malém korpusu. Pokud ne, nápad umírá levně (§6 Fáze 0)
> a ušetříš ruční-backward práci. Pokud ano, port návrhu do Rustu je
> mechanický. **Rozhodnutí je tvoje** — plán je psaný pro přímou Rust
> cestu, ale tenhle krok ho jen předřadí.

---

## 4. Nové moduly a API (`src/thermo/`)

Zrcadlí layout `src/transformer/`. Každý soubor = jedno primitivum se
vzorem Weights/Grads/Cache/forward/backward.

```
src/thermo/
  mod.rs        — re-exporty + ThermoConfig { h, w, c, c_hidden, c_conserved: Vec<usize>,
                  c_h (conv hidden), k_steps, pool_p, anneal_t_max, anneal_alpha, vocab }
  field.rs      — indexace pole (idx(i,j,c)), init_field (warm T start), pomocné normy
  conv.rs       — Conv3x3 + Conv1x1: ConvWeights/ConvGrads/ConvCache + conv_forward/backward
  flux.rs       — flux projekce (z conv outputu) + divergence: flux_div_forward/backward
                  (lineární; backward = adjoint gather; + conservation assert helper)
  diffusion.rs  — DiffusionWeights{a,b,κ0} + lap_forward/backward (κ(T)·∇²s, dT příspěvek)
  inject.rs     — InjectWeights{ inj:[vocab×C] } + fixní profile + inject_forward/backward
  readout.rs    — AvgPool + LinearProj (reuse) → logits; readout_forward/backward
                  (+ feature flag pro spektrální DFT bázi)
  step.rs       — StepCache; step_forward(field, w, anneal) -> (field', cache);
                  step_backward(w, cache, d_field') -> d_field  (skládá conv→flux→diffuse)
  rollout.rs    — RolloutCache (trajektorie T·K); rollout_forward(tokens) -> (logits, cache);
                  rollout_backward(cache, targets, grads) -> loss
  model.rs      — ThermoLM { config, conv1, conv2, diffusion, inject, readout, adam, muon, step }
                  forward/backward/step_adamw/step_muon_hybrid/clip_grads/param_count
                  ThermoGrads (zero/zeros/add_assign/scale/l2_norm — jako ModelGrads)
                  ThermoAdamState / ThermoMuonState
  checkpoint.rs — THERMO01 magic, stejný binární styl jako src/checkpoint.rs
```

Klíčové signatury (vzorově, ne finální):
```rust
// step.rs — jeden sub-step
pub fn step_forward(f: &[f32], w: &StepWeights, anneal: f32, cfg: &ThermoConfig)
    -> (Vec<f32>, StepCache);
pub fn step_backward(w: &StepWeights, c: &StepCache, df_out: &[f32], g: &mut ThermoGrads)
    -> Vec<f32>;

// flux.rs — strukturální zachování
pub fn flux_div_forward(flux_x: &[f32], flux_y: &[f32], cfg: &ThermoConfig) -> Vec<f32>; // div
pub fn flux_div_backward(d_div: &[f32], cfg: &ThermoConfig) -> (Vec<f32>, Vec<f32>);     // adjoint
```

`ThermoGrads` musí mít `add_assign`/`scale`/`l2_norm`/`zero` jako
`ModelGrads` — kvůli rayon batch redukci a grad clippingu (stejný vzor
jako `cpu_optimizer_step`).

---

## 5. Trénink a integrace

### 5.1 Optimizer wiring (trivální)
Každé pole params dostane `AdamGroup::zeros(n)`; v `step_adamw` zavoláme
`adamw_apply(params, grads, group, cfg, bc1, bc2, wd?)`. Matice (conv W1/W2,
readout W_out, inject) můžou jít na Muon přes `muon_apply(.., rows, cols,
..)` — `step_muon_hybrid` jako u transformeru (drobné `a,b,κ0` na AdamW).
`cosine_warmup` pro LR. WD na maticích, ne na difuzních skalárech/inject.

### 5.2 Trénovací krok (rollout BPTT)
Analogie `cpu_optimizer_step`: rayon přes batch elementy, každý vlastní
`ThermoGrads`, `rollout_forward`+`rollout_backward`, redukce `add_assign`,
`scale 1/B`, `clip_grads`, `step_muon_hybrid`. Batching: stejný „stream"
model (`train_toks` → B streamů, `chunk_idx` okna `seq_len`).

### 5.3 Curriculum na K (§5.4)
CLI `--k-start 4 --k-end 8 --k-warmup-steps N`: K roste během tréninku.
Krátké rollouty nejdřív, postupné prodlužování — doloženě pomáhá stabilitě.

### 5.4 Metriky / checkpoint / sampling
- **Metriky:** `MetricsWriter` přesunout z `aurora_train.rs` do nového lib
  modulu `src/trainio.rs` (mechanický lift, ~80 řádků), použít z obou
  examples. Stejné JSONL schema (`run`/`step`/`epoch_done`/`checkpoint`),
  per-event flush. → `logs/<ckpt_stem>_<ts>.jsonl`.
- **Checkpoint:** `thermo/checkpoint.rs`, magic `THERMO01`, ukládá config
  dims + všechny params jako f32 slices (Adam/Muon momenty ne — fresh
  resume, jako AURORA03).
- **Eval/sampling:** sliding-window val BPB (jako `evaluate`); autoregresivní
  `generate` reusne `sample_token`, navíc volitelný globální anneal schedule
  přes generovaný výstup (§4).

### 5.5 Entry point `examples/thermo_train.rs`
Vlastní loop (rollout je jiný), ale reusne lib helpery (load_corpus, BPE,
trainio::MetricsWriter, bits_per_byte, sample_token, viz). CLI nadmnožina:
`--corpus --vocab --grid 32 --channels 8 --c-hidden 3 --k-start --k-end
--pool 4 --anneal-tmax --anneal-alpha --batch-size --epochs --lr
--warmup-steps --weight-decay --grad-clip --muon-lr --checkpoint --bpe
--metrics --seed --sample-len --sample-temp --readout moments|spectral|both
--conserve rho,e --gpu --viz`.

---

## 6. GUI / vizualizace (Bevy 0.18) — „včetně gui"

Dokumentem slíbená interpretovatelnost: *„můžeš se dívat na teplotní pole"*
(§7). Centrem GUI je **živá heatmapa pole** během tréninku.

### 6.1 Datový kontrakt — `ThermoSnapshot`
Nový `src/viz/thermo_snapshot.rs` (paralelně k `TrainingSnapshot`):
```rust
pub struct ThermoSnapshot {
    step, epoch, total_epochs, loss, bpb, lr, grad_norm, elapsed_s, params, backend,
    loss_history, bpb_history,                 // reuse vzor z TrainingSnapshot
    channels: Vec<ChannelField>,               // T, ρ, e, |u| — každý {name,h,w,data,vmin,vmax}
    k_of_k: (u32,u32), anneal_t: f32,           // rollout indikátor
    inject_xy: (f32,f32),                       // střed trysky (zvýraznění)
    input_preview, pred_preview, pred_correct,  // reuse
    step_tick, finished,
}
```
Trénink píše `try_lock` jednou per krok (throttle), render `try_lock`uje —
přesně stávající vzor `publish_snapshot`/`poll_snapshot`.

### 6.2 Rendering heatmap — dynamická textura (doporučeno)
Per zobrazený kanál: jeden `Image` (H×W RGBA8) + quad (Mesh2d/Sprite s
**reálným** image handle — to renderuje; `Sprite::from_color` ne). Každý
frame přepíšeme byty image z `ChannelField.data` přes colormap
(turbo/blue→red, auto-range z vmin/vmax). Škáluje i na větší mřížky a 3D
řezy (Fáze T2). Alternativa pro start: H·W cell entit (1024 = OK ve 2D),
jednodušší, ale neškáluje.

Layout: velká **Temperature** heatmapa vlevo (s vyznačenou tryskou), řádek
malých (ρ, e, |u|) vpravo, dole rollout indikátor `k/K` + `anneal_T`.

### 6.3 Reuse z `src/viz/`
- **100 % beze změny:** kamera, app scaffolding (`DefaultPlugins`, okno),
  `poll_snapshot` vzor, loss/bpb gizmo plot (`draw_loss_chart` čte history),
  status bar a tokens strip (Text2d panely), pulse/finished handling.
- **Nové:** `viz/thermo_scene.rs` (spawn quadů + Image assetů),
  `viz/thermo_systems.rs` (`update_heatmaps`: pole→colormap→Image byty),
  `run_thermo_window(snapshot, channels)` entry (blokuje main thread, jako
  `run_window`; trénink běží v threadu — stejný vzor jako `aurora_train.rs`
  ř. 470–490).
- Gated `--features viz`, čistě aditivní (default/CI build nezasahuje).

### 6.4 Stretch: animace sub-kroků
Pro „watch the gas" feeling publikovat pole **per sub-step** posledního
tokenu v batchi → v GUI vidíš K-krokový rollout hot→cold pro jednu
predikci. Volitelný „inference playground" mód (zadej prompt, sleduj).

Screenshoty GUI **vždy** do `screenshots/` (CLAUDE.md).

---

## 7. Stabilita a trénovatelnost (mapování §5)

| §5 obrana | Realizace zde |
|---|---|
| 1. Difuze jako stabilizátor | Vestavěná `κ(T)·∇²` v každém kroku — kontrahuje, krotí gradienty |
| 2. Gradient checkpointing | **Fáze T0 NE** (viz níže), **Fáze T2 ANO** (3D/GPU VRAM) |
| 3. StableBPTT (stop-grad) | Volitelný flag Fáze T1, měřit jestli dovolí delší K |
| 4. Kratší rollouty + curriculum | K-curriculum (§5.3), default start K=4 |
| 5. Truncated BPTT | Cross-window detach, Fáze T1 |

**Vědomá odchylka — gradient checkpointing ve Fázi T0:** dokument ho radí
i pro prototyp, ale tam jde o PyTorch. Ve 2D toy (H=W=32, C=8, T=64, K=8) je
plná trajektorie ~17 MB/sekvence; per-op cache ~5× ⇒ ~85 MB/sekvence ×
počet rayon threadů (ne × B, elementy běží po jednom). To je na CPU
zvládnutelné → **Fáze T0 ukládá plnou trajektorii** (exaktní, nejjednodušší).
Checkpointing zavedeme až když VRAM/3D tlačí (GTX 1650 = 4 GB). Knoby:
`grid`, `K`, `T`, `batch_size`. Odchylku uvádím vědomě.

---

## 8. Validace a anti-regrese (mirror `tests/`)

Vzhledem k ručnímu backwardu je gradient check **záchranná síť č. 1**.

- **`tests/thermo_primitives.rs`** — central diff (`EPS=1e-3`, `TOL=5e-2`,
  dot s náhodným upstreamem, vzor z `transformer_primitives.rs`) pro:
  conv (W1,W2,b, vstup), flux-divergence, difuze (vč. `dT` přes κ), injekce
  (inj řádek), readout (pool+linear), a **celý `step` + krátký `rollout`**
  (K=2, T=3) end-to-end.
- **Conservation test** — `Σ field[conserved]` konstantní přes `step`
  do fp tolerance (validuje flux-form strukturálně).
- **`tests/thermo_smoke.rs`** — copy/induction toy konverguje (vzor
  `transformer_smoke.rs`); single step ne-NaN.
- **Bigram kill-gate** — změřit bigram BPB baseline (z token counts);
  **pokud termo nepřebije bigram, nápad umírá tady** (§6 Fáze 0). Gate je
  explicitní akceptační kritérium Fáze T0.
- **`tests/thermo_quality_regression.rs`** — BPB práh + checkpoint
  round-trip (vzor `lm_quality_regression.rs`).
- **Stabilita** — grad health, sampling diverzita, LR schedule, K-curriculum
  monotónní (vzor `stability.rs`).
- **`tests/thermo_gpu_parity.rs`** (Fáze T2) — CPU↔GPU drift `< 5e-2`.

---

## 9. Fázový plán (s akceptačními bránami)

### Fáze T0 — CPU 2D toy *(levná smrt nápadu)*
Primitiva + ruční backward + gradient checks; rollout assembly; readout
3.1; injekce; flux-form + difuze; trénink na malém char-level korpusu
(enwik8 prefix nebo Shakespeare jako v RC paperu); reset pole per okno;
plná trajektorie (bez checkpointingu).
**Deliverables:** `src/thermo/*`, `tests/thermo_primitives.rs`,
`tests/thermo_smoke.rs`, `examples/thermo_train.rs`, bigram baseline.
**🚦 Brána:** všechny gradient checks zelené + conservation test +
**přebití bigramu**. Bez toho nepokračovat.

### Fáze T0.5 — GUI
Heatmapa pole live během tréninku + reuse loss/metrik panelů.
**Deliverables:** `src/viz/thermo_*`, `run_thermo_window`, screenshot do
`screenshots/`. **Brána:** vidíš teplotní pole „dýchat" během tréninku.

### Fáze T1 — Stabilizace + readout ablace
Spektrální readout (DFT báze) jako další featury; globální anneal přes
generování; StableBPTT trik (dovolí delší K?); cross-window TBPTT;
**ablace:** flux-form vs volný update (zachování pomáhá?), difuze on/off
(stabilizační efekt?). **Brána:** ablace potvrdí přínos zachování+difuze;
spektrální readout změřen.

### Fáze T2 — Škálování: GPU a/nebo 3D
GPU stencil kernely (odhad ~2000–2500 řádků WGSL+Rust; reuse reductions/
elementwise/AdamW/Muon kernelů a parity harnessu — viz GPU mapa) NEBO 3D
mřížka `L=16/24` + gradient checkpointing (VRAM). Diferencovatelný LBM
(Úroveň A) jako volitelná hlubší větev. **Brána:** GPU parity `< 5e-2`;
3D běží v VRAM rozpočtu.

### Fáze T3 — Srovnání
Perplexity/BPB vs transformer stejné velikosti; efektivita tréninku/
inference; interpretovatelnost (teplotní pole). Dokumentovat výsledky.

---

## 10. Rizika a realistická očekávání (mapování §7)

| Riziko (§7) | Mitigace zde |
|---|---|
| 1. Gradienty přes rollout (nejpravděpodobnější killer) | difuze + krátké K + curriculum + (T1) StableBPTT; gradient checks chytí chyby v ručním backwardu |
| 2. Readout bottleneck | momenty → (T1) spektrální readout, víc featur |
| 3. Expresivita (3×3 × K = malý receptive field) | carry pole přes okno; (T1) multi-scale / spektrální globální módy |
| 4. Diskrétní vs spojité | injekce + sampling jako mosty (můžou být ztrátové) |

**Očekávání (§7):** RC-as-LM paper — transformer vyhrává v kvalitě, RC je
efektivnější. Necíl: přebít transformer na perplexity. Cíl: (a) funguje
vůbec? (b) je *zajímavě jiný* — efektivita, interpretovatelnost, induktivní
biasy, spojitá latentní dynamika?

**Šumová podlaha (z paměti projektu):** BPB rozdíly `< ~0.035` jsou
screeningový šum, ne signál. Bigram kill-gate i srovnání s transformerem
poměřuj proti tomuhle prahu — sub-0.05 „výhry" neber jako potvrzené.

---

## 11. Auditní tabulka věrnosti dokumentu (požadavek → artefakt)

| Dokument | Požadavek | Artefakt v plánu |
|---|---|---|
| §2 | Euler (pole), ne Lagrange | `field.rs` — tenzorové pole `H×W×C` |
| §2.2 | Úroveň B start, ne A/C | conv operátor `N_θ`, ne LBM/NS |
| §2.2 | Lokalita 3×3 | `conv.rs` Conv3x3 |
| §2.2 | Flux-form zachování (strukturální) | `flux.rs` směrové hranové toky + divergence telescoping |
| §2.2 | Teplotně řízená difuze | `diffusion.rs` `κ_c(T)·∇²` |
| §2.2 | Translační symetrie | sdílené conv váhy |
| §3.1 | Momentový readout (start) | `readout.rs` AvgPool + Linear |
| §3.2 | Spektrální readout | DFT báze matmul (Fáze T1, bez FFT dep) |
| §3.4 | Injekce = lokalizované forcing, `W_in` | `inject.rs` profile × `inj[vocab×C]` |
| §4 | Lokální teplota jako kanál | `T` ∈ C, κ(T) coupling |
| §4 | Globální anneal hot→cold | per-token `anneal_k` + gen schedule |
| §5.1 | Difuze stabilizátor | vestavěná v každém kroku |
| §5.2 | Gradient checkpointing | Fáze T2 (T0 plná trajektorie — vědomá odchylka §7) |
| §5.3 | StableBPTT | Fáze T1 flag |
| §5.4 | Krátké K + curriculum | `--k-start/--k-end`, default K=4→8 |
| §5.5 | Truncated BPTT | cross-window detach, Fáze T1 |
| §6 Fáze0 | 32×32, ~5–8 kanálů, K=4–8 | defaults: H=W=32, C=8, K=4→8 |
| §6 Fáze0 | Bigram kill-gate | §8 explicitní brána Fáze T0 |
| §6 | PyTorch-first | **vědomá odchylka → Rust** (§3), volitelný de-risk |
| §7 | Interpretovatelnost teplotního pole | GUI heatmapa (§6) |

---

## 12. Pořadí prací (první commity Fáze T0)

1. `src/thermo/{mod,field}.rs` — config + pole + init.
2. `src/thermo/conv.rs` + gradient check. *(zelená brána)*
3. `src/thermo/flux.rs` + gradient check + conservation test. *(zelená brána)*
4. `src/thermo/diffusion.rs` + gradient check.
5. `src/thermo/inject.rs` + `readout.rs` + gradient checks.
6. `src/thermo/{step,rollout}.rs` + end-to-end gradient check (K=2,T=3).
7. `src/thermo/model.rs` — assembly + optimizer wiring + `ThermoGrads`.
8. `src/thermo/checkpoint.rs` (THERMO01) + `src/trainio.rs` (lift MetricsWriter).
9. `examples/thermo_train.rs` + `tests/thermo_smoke.rs` + bigram baseline. *(KILL-GATE)*

Každý krok: malý commit, gradient check zelený před dalším. Backward se
nikdy nepíše „naslepo" — nejdřív forward, pak finite-diff check, pak
napojení do rollutu.

---

## 14. Fáze T0 — výsledky (IMPLEMENTOVÁNO, kill-gate PASSED)

**Postaveno a ověřeno.** Celý `src/thermo/` (`field, conv, flux, diffusion,
inject, readout, step, model, checkpoint`), `examples/thermo_train.rs`,
`tests/thermo_primitives.rs` (8 gradient checků), `tests/thermo_smoke.rs`
(overfit + no-NaN + checkpoint round-trip). Všech 11 testů zelených.

**Korektnost autodiffu prokázána.** Random-projection finite-difference
(`EPS=1e-3`, abs/rel tolerance) na všech primitivech + **end-to-end
rollout BPTT** (inject→K kroků→readout, threaded zpět přes T pozic).
Strukturální zachování ověřeno (`Σ` konzervovaných kanálů invariantní).

**🚦 KILL-GATE PASSED.** tiny_200k (Shakespeare), byte-level (vocab 256):

| konfigurace | val bpb | vs bigram 3.674 |
|---|---|---|
| seq32·K4·dt0.5 (nestabilní) | 26.8 | ❌ exploze gradientů |
| seq8·K2·dt0.1 (stabilní) | 4.68 | ❌ |
| seq16·K4·pool2 | 4.44 | ❌ |
| + silná injekce (0.5) | 4.21 | ❌ |
| + **pool=1** (plný readout), 800 kroků | 3.69 | ~remíza (v šumu) |
| + **1500 kroků** | **3.531** | ✅ **−0.143** (>> 0.035 šum) |

Vítězná konfigurace: `grid 16, c 8, conv_hidden 24, K 4, dt 0.1, pool 1,
seq 16, batch 24, lr 2.5e-3, 1500 kroků`. train_bpb 3.19 (pod bigramem),
val monotónně klesá, sampling produkuje Shakespeare strukturu (jména
postav, dvojtečky, newlines). **Nápad žije.**

**Dvě klíčová empirická zjištění (potvrzují dokument):**
1. **Exploze gradientů přes rollout (§5/§7 riziko #1) je reálná.** Naivní
   dlouhý rollout (seq32×K4 = 128 sub-kroků) dal `‖g‖ ≈ 1.5e8` a počáteční
   loss 26. Fix přesně dle dokumentu: **kratší rollout + menší `dt`** →
   `‖g‖ ≈ 1.4`, healthy start (loss ≈ ln 256). Difuze sama nestačí; `dt`
   a hloubka rolloutu jsou hlavní páky stability v T0.
2. **Readout bottleneck (§7 riziko #2) byl rozhodující.** AvgPool přes
   celou mřížku ředil lokalizovanou injekci → model neuměl ani bigram.
   `pool=1` (plný field → readout) skočil z 4.21 na 3.53. Pro Fázi T1 je
   readout (spektrální / méně agresivní pooling) hlavní směr zlepšení.

**Vědomé odchylky vůči plánu (drobné):**
- `rollout.rs` sloučen do `model.rs` (rollout = model forward/backward —
  čistší, věrné transformer vzoru).
- `src/trainio.rs` (JSONL lift) odloženo — trainer tiskne na stdout;
  není potřeba pro kill-gate.
- T0 ukládá plnou trajektorii bez checkpointingu (dle §7, záměrně).

**Co dál (Fáze T0.5 + T1):** GUI heatmapa (§6) — HOTOVO, viz §15.
Spektrální readout (§3.2), K-curriculum + delší rollout se StableBPTT (§5),
cross-window TBPTT, škálování na 32×32 a delší trénink (train < val ⇒ je
prostor; trend ještě klesal).

---

## 15. Fáze T0.5 — GUI (IMPLEMENTOVÁNO)

`src/viz/thermo.rs` + `--viz` v `examples/thermo_train.rs`. Spuštění:
```
cargo run --release --features viz --example thermo_train -- \
  --corpus data/tiny_200k.txt --grid 16 --k-steps 4 --pool 1 --dt 0.1 \
  --seq-len 16 --batch-size 24 --steps 600 --lr 2.5e-3 --viz
```
Živé okno (1400×900, Bevy 0.18) ukazuje:
- **4 heatmapy pole** (Temperature, Density, Energy, |Velocity|) přes
  termální colormapu, s injekčním křížkem ve středu (tryska) — vidíš, jak
  token „nahřívá" pole a dynamika ho rozvádí. To je interpretovatelnost
  z §7. Pole se publikuje každý krok přes `ThermoLM::forward_field`.
- **train-bpb křivka** s červenou bigram baseline čárou (vizuální
  kill-gate — když křivka klesne pod čáru, přebíjí bigram).
- **status + config + tokens** panely (krok, loss, train/val bpb, lr, grad;
  vstupní text + predikovaný znak).

Architektura zrcadlí transformer viz: `Arc<Mutex<ThermoSnapshot>>`,
`try_lock` poll → `LiveSnapshot`, `Mesh2d` + per-buňka `ColorMaterial`
(jeden material na buňku, mutace barvy každý frame — vzor, který v této
Bevy 0.18 prokazatelně renderuje). Čistě aditivní, gated `--features viz`,
default/headless build nezasahuje. Screenshot → `screenshots/thermo_gui.png`.

Pozn.: in-app `save_to_disk` screenshot vrací na zdejším GPU/ovladači 0 B;
pro snímek použij externí grab okna (`ffmpeg -f x11grab` / spectacle).

---

## 16. Fáze T1 — readout ablace + stabilizace (IMPLEMENTOVÁNO)

Tři T1 položky hotové a gradient-checked (12 testů zelených):

**1. Spektrální readout (§3.2)** — `readout.rs` umí kombinovat momentový
pooling a/nebo projekci na nejnižší 2D-DCT módy (stojaté vlny). Báze je
fixní lineární mapa s ortonormálními řádky (`build_dct_basis`) — žádná FFT
závislost, backward = transpozice. `ThermoConfig.readout_{moments,spectral}`
+ `readout_modes` (modes² módů/kanál). CLI `--readout moments|spectral|both
--modes K`. Gradient check pokrývá všechny tři varianty.
- *Pozn. k designu:* při `pool=1` (plný field → readout) je spektrum
  redundantní (readout už má všechnu prostorovou info; módy jsou lineární
  funkce týchž featur). Přínos spektra se projeví jen při **lossy poolingu**
  (`pool>1`), kde momenty zahodí prostorovou strukturu a globální módy ji
  vrátí. Proto se T1 validace měří na `pool=4`.

**2. Ablace flux-form / difuze (§9 Fáze T1)** — `cfg.conserved=[]`
(`--no-conserve`, vše free-update, žádné zachování) a `cfg.use_diffusion`
(`--no-diffusion`). `step.rs` obě větve gateuje ve forward i backward;
gradient check ověřuje obě ablační cesty.

**3. K-curriculum (§5.4)** — `--k-start --k-warmup`: K lineárně roste
`k_start → k_steps` přes `k_warmup` kroků (mutace `model.config.k_steps`
per krok). Default `k_warmup=0` = bez curricula.

Checkpoint (THERMO01) rozšířen o 4 nové config pole.

### Výsledky ablačního sweepu (pool=4, grid 16, seq 16, K4, dt 0.1, 500 kroků, seed 12345)

| varianta | val bpb | Δ |
|---|---|---|
| A — jen momenty (lossy pool=4) | 4.280 | baseline |
| **B — momenty + spektrum (modes 6)** | **3.825** | **−0.455** ✅ |
| C — both, BEZ zachování | 3.857 | +0.032 (zachování pomáhá, ~na šumu) |
| D — both, BEZ difuze | 3.835 | +0.010 (difuze ~neutrální) |

**Zjištění:**
1. **Spektrální readout je velká výhra** (−0.455 bpb): globální módy vrátí
   prostorovou strukturu, kterou pool=4 zahodil. Silně potvrzuje §3.2.
   Readout je tím pádem potvrzeně hlavní páka (konzistentní s T0).
2. **Zachování** (flux-form) dává malý kladný signál (~na šumové podlaze
   0.035) — naznačuje přínos dle §2.2, ale na 500 krocích není silně nad
   šumem.
3. **Difuze** je při tomto šetrném configu (dt 0.1, K4, krátký rollout)
   ~neutrální — stabilitu už drží malé dt a krátká hloubka (zjištění z T0).
   Přínos difuze se projeví spíš u hlubších/delších rolloutů.

Pozn.: tyhle pool=4 / 500-krokové běhy jsou nad bigramem (lossy + krátké);
slouží k *izolaci efektů*, ne k překonání baseline.

### Headline T1 běh (pool=2 + spektrum, 1500 kroků)

`--pool 2 --readout both --modes 6`, grid 16, K4, dt 0.1, seq 16 →
**val bpb 3.489**, přebíjí bigram o **0.186** (T0 best byl 3.531, −0.143).
Spektrální módy umožnily, aby **pool=2 (¼ rozlišení readoutu oproti pool=1)
přebil plný-field pool=1** — přesně přínos §3.2. val ještě klesal (best v
kroku ~1350), train 3.14 < val 3.49 ⇒ stále prostor. **Nový nejlepší
výsledek linie, hlouběji pod bigramem než T0.**

---

## 17. Fáze T2 — GPU port (PROBÍHÁ, inkrementální)

Velký multi-session build; stavíme po jednom kernelu, každý **parity-tested
proti CPU twinu** (CPU↔GPU `< 5e-2`, `tests/thermo_gpu_parity.rs`), stejná
disciplína jako gradient checky na CPU. Reuse existujícího wgpu backendu
(`GpuContext`, `GpuBuffer`, kernel-boilerplate vzor z `elementwise.rs`,
`storage_entry`, push-constants).

**Hotovo (parity GREEN, CPU↔GPU < 5e-3):**
- conv **forward** (conv3×3+GELU, conv1×1) — `thermo_conv.rs`.
- conv **backward** — 2 nové kernely (conv3×3 `d_s` gather, `d_w1`,
  bez atomik) + reuse `GemmKernel` (d_h_post), `OuterProductAccumKernel`
  (d_w2), `RowReduceAddKernel` (d_b2, d_b1), `GeluBackwardKernel`
  (d_h_pre). Celý conv operátor (compute bottleneck) fwd+bwd ověřen na GPU.
- Pozn.: `OuterProductAccum` konvence = `dest[out,in]=Σ_t d_y[t,out]·x[t,in]`
  (první arg nese vstupní dim, druhý výstupní).
- **flux divergence** fwd+bwd (`thermo_stencil.rs`) — lineární stencil,
  parameter-free; bwd píše d_fx i d_fy (2 rw bindingy).
- **diffuze fwd+bwd** (κ(T)·∇², 5-bodový Laplacian + sigmoid gate).
  Backward = 3 kernely (≤8 storage bufferů limit): PRE (gbuf, cT),
  PARAM (per-kanál redukce d_k0/d_a/d_b přes buňky), DS (`∇²(gbuf)` +
  κ(T) coupling do CH_T). `[k0,a,b]` zabalené do jednoho `gate` bufferu.

- **inject fwd+bwd** — forward scatter (`out = s + anneal·profile·inj[token]`),
  backward `d_inj` = per-kanál redukce přes buňky (vlákno na kanál, token
  fixní → bez atomik); `d_s` = identita (v orchestraci se reusne d_out buffer).

- **readout (moment path) fwd+bwd** — 2 nové avgpool kernely (fwd reduce,
  bwd scatter 1/P²) + reuse `GemmTransposeB` (logits = pooled@projᵀ),
  `VectorAdd` (+bias), `Gemm` (d_pooled), `OuterProductAccum` (d_proj),
  `RowReduceAdd` (d_bias).

**Tím má KAŽDÉ primitivum thermo modelu GPU kernel ověřený na CPU parity
(conv, flux, difuze, inject, readout — vše fwd+bwd). 7 GPU parity testů
zelených.** Strategie potvrzena na celé šíři: nové jen stencil/scatter/pool
kernely (~10), zbytek reuse transformer kernelů.

**Orchestrace — rollout FORWARD na GPU HOTOVÝ + parity (`ThermoGpuForward`,
`src/gpu/thermo_step.rs`):** celý `inject → K×step → readout` zřetězený
přes T×K, pole drží on-device (ping-pong buffery), 2 glue kernely
(`ExtractFlux` = vytáhne flux sloty z conv outputu, `StepAssemble` =
`out = s + Δt·transport + Δt·anneal·difuze` s per-kanál index mapou).
**Logity GPU rolloutu == CPU `model.forward` do 2e-2.** 8 GPU parity testů
zelených. Celý forward thermo modelu běží end-to-end na GPU při parity.

**Step BACKWARD composite na GPU HOTOVÝ + parity:** celý reverz kroku
(conv→flux→difuze backward) zřetězený přes 3 nové glue kernely
(`Scale` = dt·anneal, `Disassemble` = adjoint assemble, `FluxScatter` =
adjoint extract) + ověřené op-backward kernely. `d_s` + všechny conv &
diffusion param grady == CPU `step_backward`. **9 GPU parity testů
zelených.** Tím je ověřeno celé GPU jádro: každé primitivum + step fwd +
step bwd.

**Orchestrace HOTOVÁ — `--gpu` trénink běží end-to-end (`ThermoGpu`,
`src/gpu/thermo_model.rs`):**
1. forward-with-cache: un-fused conv (`ThermoConv3x3Kernel` → h_pre +
   reuse `GeluKernel`) + plná trajektorie (`Vec<GpuBuffer>` per sub-step:
   s, h_pre, h_post, lap; per token: pooled).
2. rollout **backward**: reverz smyčky — `softmax_ce` (d_logits, 1/T) →
   readout bwd → K× step-bwd composite → inject bwd, threading d_field
   (ping-pong), akumulace 10 grad bufferů přes T×K (gate re-pack z k0/a/b).
3. `zero_grad` / `step_adamw` (reuse `AdamWKernel`) / `to_cpu`.
4. `--gpu` v `thermo_train` (`run_training_gpu`): `train_step` na device,
   periodický `to_cpu` pro viz/eval/sample.

**✅ Plný forward+backward param-grad parity vs CPU** (loss + všech 10
gradů). `--gpu` trénuje na GTX 1650; `--gpu --viz` kompiluje. **22 thermo
testů zelených** (12 CPU + 10 GPU parity). Celý thermo model — fwd i bwd —
běží na GPU při CPU-parity.

Pozn.: GPU cesta zatím trénuje **1 okno/krok** (batch=1, šumnější než CPU
batch-24, který dal 3.53). Parity garantuje, že gradienty == CPU, takže
batchovaná GPU cesta by 3.53 dorovnala — batching je budoucí optimalizace.
Při malé mřížce (16²) je navíc GPU dispatch-bound (per-krok ~0.09 s);
výhoda GPU se projeví u 32²/3D.

**Klíč k úspornosti — conv backward reuse existujících kernelů:**
- `d_h_post = d_raw @ W2` → `GemmKernel`
- `d_w2 = d_rawᵀ @ h_post` (suma přes buňky) → `OuterProductAccumKernel`
- `d_b2 = Σ_cells d_raw` → `RowReduceAddKernel`
- `d_h_pre = gelu'(h_pre)·d_h_post` → `GeluBackwardKernel`
- `d_b1 = Σ_cells d_h_pre` → `RowReduceAddKernel`
- **Nové jen 2 kernely:** conv3×3 `d_s` (gather, bez atomik — vlákno na
  vstupní element sčítá přes (ki,kj,co)) a `d_w1` (vlákno na váhu sčítá
  přes buňky). Žádné atomiky.

**Plán dalších kernelů:** flux divergence (stencil, lineární — fwd i bwd
triviální), difuze (κ(T)·∇²), injekce (scatter do `inj` tabulky — reuse
`EmbedScatterReduceKernel` vzor), readout (avgpool reduce + `GemmKernel` na
proj + spektrum přes Gemm s DCT bází). Pak orchestrace rolloutu na device
(pole zůstává on-device, řetězení přes T×K; reuse `AdamWKernel` +
grad-norm reduce na optimizer krok) a `--gpu` v traineru.

**Akceptace:** plný forward+backward parity vs CPU `< 5e-2`; pak rychlost
odemkne 32×32 / delší trénink.

---

## 18. Fáze T2 — 3D mřížka (IMPLEMENTOVÁNO)

Pole je teď `D×H×W×C` (`idx3`, channel-last); `D=1` reprodukuje původní 2D
dynamiku, takže stávající běhy fungují beze změny. 3D se zapíná
`--depth N` (default 1). Změny napříč stackem:

- **conv** 3×3 → 3×3×3 (9 → 27 tapů, `tap=(ki·3+kj)·3+kl` ↔ depth,height,width)
- **difuze** 5-bod → 7-bodový Laplacian (přední/zadní soused)
- **flux** `fx,fy` → `fx,fy,fz`, takže `c_out = c + 2·n_cons` (3 toky/kanál)
- **readout** pool `P²→P³`, spektrum `modes²→modes³`
- **kanály** přidán `CH_UZ` (`N_PHYS` 5→6, `[ρ,u_x,u_y,u_z,e,T]`); difuzní
  teplotní index jde do GPU jako uniform (`cht`), ne hardcode
- **checkpoint** `THERMO02` (+`d`); **viz** řez `z=D/2` + max-Z projekce
  na quantitu (8 panelů, reuse 2D-textura)

**CPU kill-gate (PASSED):** `--depth 4 --grid 16 --channels 16 --seq 32
--batch 24 --steps 400 --readout both` → **val bpb 3.199 vs bigram 3.674
(−0.475)**. Gradient checks zelené (toy `d=3`, vč. `d_fz`); smoke konverguje
v 3D i v `d=1` degeneraci.

**GPU parity (PASSED):** všech 16 `thermo_gpu_parity` testů `< 5e-2` při
`d=2/3` (conv 27-tap, fz flux, 7-bod difuze, readout, step-backward,
rollout, Muon, spektrum). GPU 3D trénink běží end-to-end na GTX 1650 a učí
se (loss klesá, bez NaN).

**VRAM pozn.:** plná trajektorie škáluje s `d` (`b·cells·c·nsub`). Tenká
hloubka (`D≤8`) + skromný batch se vejde do 4 GB; větší `d`/`B` chce
redukci batche nebo gradient checkpointing (§5.2 — odložené). `48-byte`
push-constant range v `thermo_conv::build` (fused transport-fwd má 9 slov).

### Profilace 3D (GTX 1650, d=4×16×16, B24, `thermo_microbench`)

Per-kernel µs/disp (cool): **dw1 splitK ~4200, conv3g fwd ~3650, conv_ds
~1500**, vše ostatní <300. **Conv je ~97 % per-substep GPU času** — 27-tap
3D conv je úzké hrdlo (3× MAC/buňka vs 2D + depth× buněk).

**Měřicí pozn.:** GTX 1650 má ~2× thermal šum mezi běhy (`dw1 splitK`
8718→4213 µs beze změny kódu) → **A/B MUSÍ být in-process** (obě varianty
v jednom běhu). Viz [[thermo-gpu-dispatch-bound]].

**Dvě optimalizace otestované a ZAMÍTNuté** (in-process A/B):
- *thread-per-cell forward conv* (akumulovat `c_h` na buňku, číst sousedství
  1× místo `c_h`×) → **~2× pomalejší**. `s` (786 KB) se vejde do L2 →
  redundantní čtení byla levná; dynamický `acc[]` spadl do local memory.
- *shared-mem dw1* (workgroup na `(g,co,ci)`, `dhp` sloupec do shared) →
  **3.7× pomalejší** (4213→15694 µs). Barriers + jen 27/32 aktivních vláken
  + sériový cell-loop přebily úsporu; `dhp` je dost L2-friendly.

**Optimalizace SHIPPED — im2col + GEMM pro `d_w1`** (největší kernel).
`d_w1[co,ci,tap] = Σ_cells dhp[cell,co]·s[neighbour(cell,tap),ci]` se
materializuje jako patch matice `[nb·cells, c·27]` (im2col kernel) a pak
`d_w1 = dhpᵀ ⊗ patch` přes `OuterProductSplitK` + `RowReduceAdd` — **stejná
rychlá cesta jako `d_w2`**. Column order `ci·27+tap` dá rovnou kanonický
`w1` layout, takže GEMM je beze změny. **In-process A/B: 4126 → 3274 µs
(~1.26×)**; parity zelená (rollout-backward + Muon testy jedou orchestrátor).
Conv-style split-K `d_w1` (scattered s-reads přes sériový cell-loop)
nahrazen. ~10 % per-substep úspora (dw1 je ~⅓ conv ceny).

**Závěr:** zbylé kernely (per-element conv fwd/`d_s`) jsou L2-rezidentní a
už dobře vyladěné — další micro-opt se nevyplácí. Páky: **config** (tenká
hloubka, menší `--batch-size`/`--conv-hidden`/`--k-steps`, `--pool>1`) a
případně **f16** (GTX 1650 ~2× fp16, velká změna, odložené).

---

## 19. Vír / černá díra — advekce + horizont (EXPERIMENT, CPU-only)

Nápad: upravit 3D prostor tak, aby obsahoval **vír nasávající vstupní data**,
a v „horizontu událostí" buď číst (read hlava), nebo **zničit informaci**.
Mapování na mechaniku modelu (analýza předcházela implementaci):

- **Vír = advekce.** Předepsané rychlostní pole `v(x)` (dovnitř `−r̂` + swirl
  `ω·θ̂` kolem osy z) biasuje hranové toky konzervovaných kanálů upwind členem
  `F += β·s_upwind·v·n̂`. Jede po **stejných telescoping hranách** jako učený
  flux, takže **zachování zůstává exaktní** (test `step_vortex_conserves_total`).
  `v` je parameter-free; backward = adjoint upwind gather.
- **Horizont jako read hlava** = Gaussova skořápka maskuje pole před readoutem.
- **Zničení informace = sink** = atenuace jádra `field *= 1−strength` jako
  poslední op sub-kroku. **Vědomě porušuje zachování** (to je ta „černá díra").
  Adjoint = stejná maska → **prostorový stop-gradient** (trunkace BPTT skrz
  jádro, tj. StableBPTT realizovaný geometricky).

**Strukturální nutnost (zachování to vynucuje):** v uzavřené krabici nelze
nasávat donekonečna — bez sinku advekce+difuze ustálí koncentrovaný profil
(hromadění, ztráta recency), se sinkem teče skutečný proud. „Buď read hlava,
nebo zničení" tedy není volba, ale fyzikálně vynucená dichotomie.

**Implementace (čistě aditivní, vzor `inject_every`/`delta_readout`):** nové
`src/thermo/vortex.rs` (VortexField + flux add/backward + sink mask); skaláry
na `ThermoLM` (`vortex_beta/swirl`, `inject_ring_radius`, `readout_horizon`,
`sink_radius/strength`), default off, `rebuild_experimental()`. `ThermoConfig`
**netknut** (žádný churn ~20 literálů). CLI `--vortex-beta --vortex-swirl
--inject-ring --read-horizon --sink-radius --sink-strength`. GPU rollout je
ignoruje (CPU-only guard). 14 gradient checků zelených (vč. vortex+sink d_s a
end-to-end rollout se všemi čtyřmi knoby).

### Výsledky (tiny_200k, grid16 c8 K4 dt0.1 pool1 seq16 B24 muon inject_every, seed 12345, 200 kroků)

| běh | konfigurace | best val bpb | Δ vs baseline | grad norm |
|---|---|---|---|---|
| baseline | bez víru | **3.526** | — | 2.2→0.9 |
| B | ring + vír, plný readout, bez sinku | 3.601 | **+0.075** (regrese > šum) | 6.6→1.4 |
| C | ring + vír + horizont + sink 0.5 | 3.512 | −0.014 (remíza, v šumu) | 3.9→0.8 |

**Zjištění (potvrzují předběžnou analýzu):**
1. **Vír/transport samotný škodí** (B +0.075, nad šumem). Koncentrace bez
   drénu + transportní zpoždění ředí čerstvý token v readoutu; navíc B má
   **nejhlučnější gradienty** (start gn 6.6). Konzistentní s tím, že transport
   je „marginální osa" (zachování/difuze byly v §16 taky na šumu), zatímco
   readout je rozhodující.
2. **Plná sestava (C) dorovná baseline s nejnižší grad norm** (tail 0.8).
   Ve stabilním režimu je ale všechno stabilní (i B), takže to je jen drobný
   rozdíl — skutečná stabilizace se ukáže až v agresivním režimu níže (a je to
   **pár vír+sink**, ne sink samotný — viz ablace).
3. **Read hlava na horizontu nepřidává kapacitu** (C ji obsahuje a jen
   remizuje — bottleneck, konzistentní s „víc info v readoutu vyhrává").
4. **Net BPB ve stabilním režimu: neutrální až záporný.** Vír je dynamická
   intervence; dynamika dělá ~šum, readout dělá signál (lekce T0/T1 platí).

### Stabilitní test + ablace — co vlastně stabilizuje (agresivní režim)

T0 doložil, že `seq32·K4·dt0.5` (128 sub-kroků) **exploduje**. Reprodukce +
úplná 2×2 ablace vír × sink (seed 12345, 150 kroků, ring jen aby injekce
nepadala do sinku):

| běh | vír | sink | best val | trajektorie ‖g‖ |
|---|---|---|---|---|
| D1 | ✗ | ✗ | 4.78 ❌ | 3e6 → 2e15 → **NaN** @100 |
| E1 | ✗ | ✓ (center) | 4.37 ❌ | klesá, pak vyletí @100 (gn 6826) |
| E2 | ✗ | ✓ (ring) | 5.31 ❌ | **NaN** @100 |
| F1 | ✓ | ✗ (ring) | 6.45 ❌ | drží do 75, pak **NaN** @100 |
| **D2** | ✓ | ✓ (ring) | **3.499 ✅** | ~1–3 stabilní, val monotónně dolů |

**Stabilizuje jen PÁR vír+sink — žádná půlka sama.** Ablace vyvrací první
(ukvapený) závěr „sink je stabilizátor": sink samotný (E1/E2) i vír samotný
(F1) **stejně divergují** jako baseline. Mechanismus = přesně ta vazba z úvodní
analýzy:
- **vír** = dovnitř mířící advekce (dopravník) → koncentruje pole do jádra;
  sám o sobě jen hromadí, až to bublne (F1 drží do kroku 75, pak runaway);
- **sink** = drén jádra; sám o sobě nemá co odebírat (do jádra nic neteče),
  distribuovanou conv-explozi v objemu neutlumí (E1/E2 → NaN);
- **spolu** = dopravník + drén = **ohraničený akreční průtok**: vír doveze
  energii pole k ústí horizontu, sink ji tam zničí. To drží magnitudu pole přes
  celý hluboký rollout → `‖g‖` zůstane O(1), val klesne na 3.499 (= dorovná
  *šetrný* baseline 3.526 v režimu, který je jinak netrénovatelný).

Tím je doslovný „akreční disk + černá díra" jako **celek** stabilizátorem —
přesně fyzikální coupling, který zachování vynucuje (uzavřená krabice: sání bez
drénu = hromadění, ne proud).

### Škálování dat — tiny.txt (1.1M, 5.5×), 3 seedy

Test, jestli „víc dat" promění neutrální stabilizátor ve výhru. Model fixní
(grid16 c8 conv24), 350 kroků, byte-level, bigram baseline 3.597:

| běh | konfigurace | best val (seed 12345) |
|---|---|---|
| G-base | seq16·dt0.1, bez víru | 3.342 |
| G-vtx | seq16·dt0.1 + vír+sink | 3.347 (remíza) |
| G-base32 | seq32·dt0.1, bez víru (kontrola) | 3.229 |
| A-vtx | seq32·dt0.5 + vír+sink | **3.172** |

**Dekompozice je klíčová** (vyhýbá se záměně délky sekvence s vírem):
1. **Většinu zisku udělala délka sekvence** (G-base→G-base32 = **−0.113**),
   bez víru. Potvrzuje „scale/kontext dominuje" ([[aurora-bpb-scale-dominated]]).
2. **Vír+sink ve šetrném režimu zůstává neutrální** i s víc daty (G-vtx remíza).
3. **A-vtx vs férová kontrola G-base32** (stejná seq32, jen dt0.1 vs dt0.5+vír):
   3-seedová replikace (seedy 12345/222/333):

   | | A-vtx | G-base32 | rozdíl |
   |---|---|---|---|
   | seedy | 3.172 / 3.198 / 3.161 | 3.229 / 3.222 / 3.196 | −0.057 / −0.024 / −0.035 |
   | **průměr** | **3.177** | **3.216** | **−0.039** |

   A-vtx přebíjí kontrolu **3/3 seedy** s téměř čistou separací (A-vtx ≤3.198,
   kontrola ≥3.196). Průměr **−0.039** je **těsně nad** šumovou podlahou 0.035.

**Závěr škálování:** s víc daty se vír+sink posouvá z „neutrální" na **slabý,
konzistentně replikovaný pozitiv ~0.04** — ale jen skrz odemčení `dt0.5` režimu
(který je bez stabilizátoru NaN). Tedy: hodnota = „smět trénovat při větším dt",
přístupná pouze přes vír+sink. Na šumové hraně, ne robustní výhra, ale 3/3
znaménkově konzistentní.

### Verdikt §19

Vír/černá díra **mechanicky funguje** (gradient-checked, vír zachovává exaktně,
trénuje). Empiricky:
- **BPB: ve šetrném režimu neutrální; s víc daty slabý pozitiv ~0.04** (3/3
  seedy nad kontrolou, těsně nad šumem) — ale jen jako vedlejší efekt
  stabilizace většího `dt`. Hlavní páka „víc dat" byla délka sekvence (−0.11,
  bez víru). Transport/ring/horizont samy jsou marginální (jako zachování/difuze
  v §16); readout zůstává dominantní páka.
- **Stabilizace patří PÁRU vír+sink, ne jedné komponentě.** V `seq32·dt0.5`
  všechny čtyři ostatní cely (D1/E1/E2/F1) divergují do NaN; jen vír+sink (D2)
  trénuje čistě na 3.499. Coupled dopravník+drén ohraničí magnitudu přes hluboký
  rollout — adresuje #1 killer linie (§10 riziko 1, §5) jako *geometrickou*
  alternativu k difuzi/malému dt.

**Další směr (jestli honit):** vír+sink pár jako stabilizátor pro K-curriculum /
delší seq / cross-window TBPTT (§5.5), kde difuze+malé dt nestačí — ale měřit
proti prostému „silnější difuze / menší dt / grad-clip", jestli ten geometrický
trik vůbec něco přidá nad levnější páky. Vír/ring/horizont jako BPB páky
**mrtvé** (šumová podlaha 0.035; dynamika ≪ readout). GPU port pouze pokud se
pár osvědčí nad baseline stabilizátory (vír = upravit flux extract, sink =
elementwise maska ve step fwd/bwd).

---

## 20. Stromový (hyperbolický) substrát — netradiční geometrie #2 (EXPERIMENT)

Po víru (prostorový *transport*) druhá geometrie: nahradit plochou mřížku
**k-árním stromem** — lokální operátor pak má *exponenciální* receptivní pole
(`~k^K` uzlů za K kroků vs `~K` na mřížce), geometrie záporné křivosti /
hierarchie jazyka.

**Design (klíč k levné implementaci):** strom = degenerovaná mřížka
`d=1,h=1,w=N`, `pool=1`. Jediná tree-specifická věc je **conv** (graph-conv přes
tapy `{self, parent, k×child}`); `inject` (profil na listy) + `readout` (pool=1
momenty přes N uzlů) + optimizer/rollout se **reusnou**. v1 bez flux/difuze
(~neutrální dle §16/§19). Standalone `TreeLM` — **nezasahuje grid/GPU path**.
Soubory: `tree.rs` (topologie + `tree_conv` fwd/bwd), `tree_model.rs` (`TreeLM`),
`examples/tree_train.rs`. Gradient checks (`tests/thermo_tree.rs`): `tree_conv`
(binary/ternary/path) + **end-to-end rollout BPTT** — zelené.

**Výsledky** (tiny.txt, seq16, 350 kroků, seed default, binary depth8 = 255
uzlů ≈ 16² mřížka, ~525–528K params jako grid):

| model | best val | vs flat |
|---|---|---|
| flat grid (G-base) | **3.342** | — |
| strom, conv_hidden 24 (lean operátor) | 3.383 | +0.041 |
| strom, conv_hidden 96 (fair operator capacity) | 3.411 | +0.069 |

Strom je **konzistentně horší** a víc operator-kapacity nepomohlo (vyloučen
param confound — total params ~stejné, readout dominuje obě). Trénuje stabilně
(K=8 = 128 sub-kroků, gn ~1–3, monotónní), přebíjí bigram, ale **ne plochou
mřížku**. Exponenciální dosah nepřinesl nic.

**Sdílená příčina s vírem (potvrzeno 2×):** `pool=1` readout vidí **každý**
uzel, takže prostorová geometrie — ať ji *přeskupíš* (vír) nebo *přepojíš*
(strom) — je pro loss **neviditelná**. Dosah nepomohl, protože readout stejně
čte všechny uzly.

### Meta-lekce po dvou geometriích

**Prostorová geometrie substrátu = neutrální až mírně záporná pro BPB.** Aby
geometrie hrála roli, musí cílit osu, kterou globálně-prostorový readout
*nedokáže* zahladit:
- **(a) readout sám** — multi-scale / fraktální pyramida (cílí prokázanou páku);
- **(b) časovou/sekvenční osu** — „spacetime" substrát, kde se sekvence rozloží
  jako prostorová geometrie. Long-range *temporal* je skutečná mezera (drží ji
  jen rekurentní carry), a readout je globální jen prostorově, ne časově.

Transport (vír) i konektivita (strom) jsou obě prostorové → obě mrtvé. Další
geometrie, pokud nějaká, jen (a) nebo (b).

---

## 21. Spacetime substrát — netradiční geometrie #3 (EXPERIMENT, VELKÁ VÝHRA)

Volba (b) z meta-lekce §20: **sekvence jako osa pole**, zpracovaná **paralelně**
(žádný rekurentní carry). embed-na-čas → K **kauzálních dilatovaných conv**
kroků → **per-pozici (časově-lokální) readout**. Tři důsledky, každý míří na
prokázanou slabinu:
- **readout je časově-lokální** (každá pozice čte svou kauzálně-akumulovanou
  buňku, ne globální pool) → časová geometrie je *viditelná* pro loss — to, co
  prostorová nikdy nebyla (§19/§20);
- **BPTT hloubka = K, ne T·K** (není rekurence) → likviduje #1 killer (exploze
  přes hluboký rollout, §10 riziko 1);
- **dilatované kauzální tapy** `d=base^k` → exponenciální *časový* dosah
  (WaveNet) — dilatace na ose, kde dosah konečně platí.

**Implementace:** novum je **kauzální dilatovaná 1D conv** (`spacetime.rs`,
`Conv1x1(GELU(CausalConv1d))`, tap `j` čte `t−j·d`, nikdy budoucnost).
Standalone `StLM` (`spacetime_model.rs`) — grid/GPU netknuté, reuse AdamW/Muon +
CE. Gradient checks (`tests/thermo_spacetime.rs`): conv (dil 1/2/4) + **striktní
kauzalita** (perturbace v `t0` nezmění výstup pro `t<t0`) + end-to-end StLM BPTT
— vše zelené. v1 = interní šířka S=1 (čistá 1D conv); S>1 (gas flavor + difuze
podél S) je možný refinement.

**Výsledek (tiny.txt, seq64, 350 kroků, seed default, c128 K6 dil_base2 reach~63):**

| model | best val | params | čas | pozn. |
|---|---|---|---|---|
| bigram | 3.597 | — | — | |
| flat grid seq16 @350 | 3.342 | 530K | — | rekurentní |
| flat grid seq16 @600 | 3.165 | 530K | — | nejlepší rekurentní session |
| strom | 3.38–3.41 | 525K | — | rekurentní |
| **spacetime seq64 @350** | **2.791** | **164K** | **144 s** | paralelní, kauzální |

**−0.37 pod nejlepším rekurentním, −0.55 pod matched gridem** — řádově nad
šumovou podlahou 0.035. A to s **⅓ parametrů** a **3–6× rychleji** (paralelní,
mělké BPTT). train 2.55 ≪ val 2.79 → **podtrénováno** (klesalo by dál), gn
stabilní ~1. Sampling viditelně koherentnější.

**Proč to vyhrálo (meta-lekce naplněná):** prostorová geometrie byla neviditelná
globálnímu readoutu; **časová geometrie s časově-lokálním readoutem** je velká
výhra — a zrušení rekurence sundá BPTT bottleneck. Dva „neúspěchy" (vír, strom)
vyrobily poznatek, který našel vítěze.

**Poctivá výhrada:** S=1 spacetime je v jádru **kauzální dilatovaná CNN (TCN)** —
známo-silná architektura. V rámci téhle linie je to ale jasný, velký, doložený
vítěz a správný konec geometrického oblouku (rekurentní gas → paralelní
spacetime).

**Potvrzeno (replikace + matched kontrola, vše seq64 @350, tiny.txt):**

| | best val | čas | params |
|---|---|---|---|
| rekurentní grid (matched kontrola) | 3.159 | 762 s | 530K |
| spacetime, 3 seedy (12345/222/333) | **2.791 / 2.777 / 2.770** (mean 2.779) | ~144 s | 164K |

Plně kontrolovaná shoda (stejný korpus/seq/kroky/seed): **Δ = 0.38**,
**~5× rychleji**, **⅓ parametrů**, seedově těsné (2.77–2.79) → výhra je
jednoznačná a replikovaná, ≫ šumová podlaha 0.035.

**Další směr:** spacetime je podtrénovaný (train ≪ val) → víc kroků / scale je
reálná páka; S>1 (gas flavor: interní šířka + teplotně-řízená difuze podél S,
readout pooluje S per časový řez) je možný novel refinement. Tohle je směr pro
BPB téhle linie, ne další rekurentní-gas tweaky.

---

## 22. Selektivní SSM — deep-research top pick (EXPERIMENT, nano caveat POTVRZEN)

Deep-research workflow (105 agentů, 23 zdrojů, 25 tvrzení 3-hlasně ověřeno) na
„jaká geometrie = BPB breakthrough" → **top pick: selektivní SSM** (Mamba-styl
vstupně-závislé forget hradlo; MambaByte má jediné replikované byte-BPB výhry).
Náš spacetime je statický LTI-SSM; lekce: statické míchání selhává na recall.

**Postaveno + gradient-checked** (`ssm.rs` selektivní scan fwd/bwd + kauzalita;
`ssm_model.rs` `SsmLM` = embed → N residual bloků [v=x·Win, a=σ(x·Wg+bg),
y=scan(a,v), x+=y·Wo] → per-pozici readout; `examples/ssm_train.rs`). Scan
adjoint `dh_t=dy_t+a_{t+1}·dh_{t+1}`, O(L·C), bez tape.

**A/B (tiny.txt seq64 @350, seed default) — caveat reportu HELD:**

| model | params | best val |
|---|---|---|
| spacetime (statický, kontrola) | 164K | **2.78** |
| selektivní SSM, 2 vrstvy | 164K | 2.959 |
| selektivní SSM, 4 vrstvy | 263K | 2.861 |
| selektivní SSM, 6 vrstev | 361K | 2.829 |
| selektivní SSM, 4 vrstvy c96 (matched) | 160K | 2.894 |

Hloubka pomáhá (2.96→2.83) ale **asymptotuje NAD spacetime — ani 2.2× params
nepřebije 2.78; matched je ~0.11 horší.** SSM trénuje rock-stable + 3× rychleji,
ale ne líp. **Potvrzuje explicitní caveat reportu: SSM výhry jsou na 350M–3B, na
nano (0.1–0.4M, char) se nepřenesou.** Praktický nano šampion zůstává **statický
spacetime 2.78**.

Výhrada: jde o zjednodušený Mamba-lite (gated EMA, diagonální, bez HiPPO/Δ-
diskretizace/input-dependent B,C) — plný Mamba *by mohl* mezeru zavřít, ale 4
kontrolované configy nad 2.78 + nano-caveat = slabá EV honit ho na téhle škále.

### Závěr geometrického oblouku (§19–22)
spatial (vír, strom) = mrtvé (globální readout je zahladí) → **temporal
(spacetime) = velká výhra, 2.78** → selektivita (SSM) = na nano nezabere.
**Šampion = statický spacetime.** Reálné páky dál: scale (data/kroky/params),
ne další exotická geometrie.

---

## 23. Spacetime GPU port (IMPLEMENTOVÁNO, parity GREEN, enwik8_10m na GPU)

Spacetime byl CPU-only; doportováno na GPU (`src/gpu/spacetime_gpu.rs`,
`StLmGpu`). Protože model je paralelní (žádná per-timestep rekurence), je port
jednodušší než `ThermoGpu`: jedno pole `[B·L·c]`, K reziduálních conv kroků,
time-local readout = GEMM, caches jen K+1/K (ne L·K).

**Nové kernely (4):** kauzální dilatovaná conv1d (gelu-cache / d_s / d_w1
direct), embed-inject, axpy. **Reuse:** ThermoConv1x1, ConvBwdHidden,
Gemm/GemmTransposeB, OuterProductAccum, RowReduceAdd, BiasRowAdd, SoftmaxCE,
EmbedScatterReduce, AdamW, celý Muon (`muon_one` verbatim — matrix-generic),
GradNorm. `--gpu` v `examples/spacetime_train.rs`. **`--gpu --viz` funguje** —
trénink běží na GPU ve vlákně, Bevy okno na main threadu (dvě wgpu kontexty
koexistují), pole se publikuje na eval krocích přes `gpu.to_cpu()` →
`forward_field` → heatmapy kanál×čas (vzor z mřížkového `--gpu --viz`).

**Parity GREEN** (`tests/thermo_spacetime_gpu.rs`): fwd logity vs CPU `< 5e-3`;
bwd všech 7 grad tenzorů + loss vs CPU `< 2e-2`. Celé jádro (fwd+bwd+optim)
ověřeno proti CPU `StLM`.

**Výsledek na enwik8_10m (GTX 1650, c128 K6 seq64 B32, 1500 kroků):
val bpb 2.7345 vs bigram 3.9405 (−1.21), za 259 s (~4.3 min)** — z CPU hodin na
minuty. train 2.33 ≪ val 2.75 (podtrénováno, prostor je). Sanity na tiny.txt:
trajektorie sedí na CPU (GPU 200 kroků 2.99 ≈ CPU).

**Dva reálné fixy z velkého configu (c256/conv512/seq128/K8):**
1. `d_w2` redukce dispatchla `c·ch=131072` workgroups > limit 65535 (grid model
   to nikdy netrefil, tam je `c·ch` malé) → přepnuto na output-tiled
   `OuterProductAccum`.
2. **`d_w1` direct kernel je memory-bound pro velké `ch·c`** (strided uncoalesced
   čtení `s`; GPU util ~13 % → ~3 s/krok). Fast path = **im2col + tiled GEMM**
   (přesně jako grid §18). Odložené; lean config (`ch·c` skromné) jede plnou
   rychlostí (0.17 s/krok). To je hlavní zbývající GPU optimalizace, jestli se
   bude škálovat spacetime výš.

**Závěr:** šampion linie (statický spacetime) teď běží **na GPU na cílovém
korpusu (enwik8) při parity**, řádově rychleji. Další: im2col `d_w1` (odemkne
velké configy) + scale (delší trénink / víc dat / params).
