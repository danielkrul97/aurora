# Implementační plán Aurora

> Cesta od prázdného `cargo new` k funkční spiking neural network připravené na první ostrý benchmark. Šest sprintů, každý s jasným cílem a exit kritériem. **Žádné odhady v týdnech** — postupujeme až po splnění exit kritérií.

---

## Princip plánování

1. **Discrete-time před event-driven.** Začneme s pevným časovým krokem `dt`, kde se v každém kroku updatují všechny neurony. Je to pomalejší, ale **debugovatelné** a deterministické. Event-driven přijde až jako optimalizace ve sprintu 3.
2. **Test-driven.** Každý sprint má integration test, který je zároveň exit kritériem. Když test prochází, sprint je hotový.
3. **Žádné předčasné abstrakce.** První tři sprinty řeší konkrétní problémy. Trait hierarchii, plug-in pluginy a generika necháváme na později.
4. **Žádné `unsafe`** dokud nebude profilováním prokázané, že to potřebujeme.

---

## Sprint 0 — Setup (~30 minut)

**Cíl:** Funkční Cargo projekt, CI, formátování.

**Deliverables:**
- `Cargo.toml` (edition 2021, name `aurora`, type lib + bin)
- `rustfmt.toml`, `clippy.toml` (defaults stačí)
- `.github/workflows/ci.yml` — `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`
- `src/lib.rs` — prázdný modul
- `src/main.rs` — `fn main() { println!("aurora"); }` (zatím jen smoke)

**Exit kritérium:** `cargo test && cargo clippy -- -D warnings && cargo fmt --check` projdou na čisté větvi.

**Otevřené otázky:** žádné.

---

## Sprint 1 — Jeden LIF neuron + STDP (~150 LOC)

**Cíl:** Jeden neuron, jedna synapse, STDP. Discrete-time. Důkaz, že se neuron dokáže „naladit" na opakovaný vstupní vzor.

**Deliverables:**
- `src/types.rs` — základní typy:
  ```rust
  pub type Time = f64;       // sekundy
  pub type Volts = f64;      // membránový potenciál
  pub type Weight = f64;     // synaptická váha
  pub struct Spike { pub t: Time, pub source: NeuronId }
  ```
- `src/lif.rs` — `Neuron` se stavem `(V, last_spike_t)` a metodou `step(&mut self, dt: Time, input: Volts) -> Option<Spike>`. Implementuje rovnici z dokumentu 09 sekce 2.
- `src/synapse.rs` — `Synapse { weight, pre_id, post_id, delay, last_pre_t, last_post_t }` s metodou pro STDP update (`update(now: Time)`)
- `src/stdp.rs` — čistá funkce `pub fn stdp_delta(dt_pre_post: Time, params: &StdpParams) -> Weight`
- `src/network.rs` — `Network` s `Vec<Neuron>` a `Vec<Synapse>`, metoda `step(dt)` projede všechno v pořadí: synapse doručí, neurony updatují, STDP aplikuje
- `tests/single_neuron_lif.rs` — neuron dostává konstantní proud, ověří se že začne stříkat s očekávanou frekvencí
- `tests/stdp_pairing.rs` — dva neurony, vynucené pre→post párování, váha musí monotónně růst po N opakováních

**Exit kritérium:** Oba testy procházejí. Frekvence vystřelování se shoduje s analytickou predikcí LIF v ±5 %.

**Otevřené otázky:**
- Jednotky času: sekundy (snadné papírové počty) vs. ms (běžnější v SNN literatuře). **Návrh: sekundy.**
- STDP parametry: `A⁺=0.01, A⁻=0.012, τ⁺=τ⁻=20 ms` (klasické hodnoty z Bi & Poo 1998).

---

## Sprint 2 — Dendritické kompartmenty + homeostáza (~250 LOC)

**Cíl:** Neuron přestane být „jedna vana" a začne mít víc dendritů. Homeostáza drží sparsity.

**Deliverables:**
- `src/dendrite.rs` — `Dendrite { v_local, threshold, boost }` s lokálním prahováním a „dendritickým spike" do somy přes `inject`/`tick`
- Refaktor `Neuron`: `Neuron { soma: Lif, dendrites: Vec<Dendrite>, homeo: Option<Homeostasis> }`
- Refaktor `Lif` a `Synapse`: synapse cílí buď dendrit nebo přímo somu (`post_dendrite: Option<usize>`); dodávky vstupují přes `inject` (instantní charge), sustained currents přes `tick` DC parametr
- `src/homeostasis.rs` — pomalá adaptace prahu `θ` přes exponentially-decaying spike trace (default τ=1 s)
- `tests/dendritic_xor.rs` — handwired síť (2 vstupní, hidden se 2 dendrity ±, 1 výstupní), validace že dendrity + signed weights počítají XOR
- `tests/homeostasis_balance.rs` — 100 neuronů s náhodnými drives, populační rate konverguje do pásma cílové frekvence

**Exit kritérium:** Handwired XOR dává správné spike counts pro všechny 4 patterns. Homeostáza drží populační rate v pásmu [1.5, 5.5] Hz při target 3 Hz.

**Otevřené otázky vyřešené ve sprintu:**
- Počet dendritů per neuron: konfigurovatelné, žádný default — uživatel volá `add_dendrite` explicitně.
- **STDP-učené XOR posunuto do sprintu 4** (jak předpokládal contingency plan) — bez three-factor rule není reward-driven učení proveditelné. Sprint 2 dodal architektonickou primitivu (dendrity), sprint 4 přidá učení skrze ni.

---

## Sprint 3 — Event-driven engine (~400 LOC + benchmarky)

**Cíl:** Přepsat scheduler z discrete-time na event-driven. Stejné chování, řádově méně CPU práce při řídké aktivitě.

**Deliverables:**
- `src/scheduler.rs` — `EventQueue` postavená na `BinaryHeap<Event>`, kde `Event { t, kind: EventKind }` (kind = `Spike { source, target_synapse }` | `Refractory { neuron, until }`)
- `src/engine.rs` — `Engine::run_until(t_end)` smyčka: vytahuj events, doručuj, generuj nové
- Refaktor `Network`: poskytuje stejné API, uvnitř používá engine. Discrete-time režim zachovat za feature flagem `--features discrete` jako referenci pro testy.
- `benches/sparsity_scaling.rs` — kriteriální benchmark (criterion crate): síť 10k neuronů s aktivitou 1 %, 5 %, 10 %, 50 %. Porovnání event-driven vs. discrete.
- `tests/parity.rs` — stejný setup v obou režimech musí dát stejné spike trains (deterministické seedování)

**Exit kritérium:** Parity test prochází. Při 1 % aktivitě je event-driven ≥10× rychlejší než discrete-time.

**Otevřené otázky:**
- Paralelizace: zatím single-thread. Multi-thread (Rayon) odložit na sprint 6, pokud bude potřeba. Lock-free queue je velký kus práce — neotvírat dřív, než to bude bottleneck.
- Determinismus: explicitně seedovaný RNG (`rand::SeedableRng`), žádné `HashMap` v hot pathu (nedeterministické pořadí iterace).

---

## Sprint 4 — Three-factor learning + neuromodulace (~200 LOC)

**Cíl:** Přidat globální „dopaminový" signál `M(t)`, který gateuje STDP. Bez tohohle se síť reálně nenaučí nic netriviálního.

**Deliverables:**
- `src/neuromodulator.rs` — `Modulator { value, decay }`. API `reward(amount)` zvedne `value`, `step(dt)` ho exponenciálně sníží.
- Refaktor STDP: `Δw = stdp_base × M(t)` — synapse drží **eligibility trace** (krátkodobá paměť změny), která se „commitne" jen při vysokém `M`
- `tests/reward_modulated_xor.rs` — **STDP-učené XOR** (přesunuté ze sprintu 2): trénink z reward signálu, target ≥95 % accuracy
- `tests/eligibility_trace.rs` — ověří že trace exponenciálně decaye a že late reward (~500 ms po spike) ještě stihne učení

**Exit kritérium:** Reward-driven XOR ≥95 % accuracy. Eligibility trace test prochází.

**Otevřené otázky:**
- Tvar trace: exponenciální vs. dvojexponenciální. **Návrh: exponenciální, `τ_e = 500 ms`** (lit. konsensus pro dopaminové delay).

---

## Sprint 5 — První ostrý benchmark: MNIST (~500 LOC + data pipeline)

**Cíl:** Klasifikace MNIST spiking sítí. První „nezávislé" měření — buď to funguje, nebo nefunguje, žádné výmluvy.

**Deliverables:**
- `src/encoding.rs` — převod obrázku 28×28 na spike train (rate coding: pixel intenzita → frekvence; alt. latency coding)
- `src/topology.rs` — builder pro vrstvené sítě (`784 → 400 → 10`), zatím feedforward
- `src/readout.rs` — klasifikace podle rate-based readoutu z výstupní vrstvy (počet spiků za okno)
- `examples/mnist_train.rs` — trénovací smyčka (online, jeden vzorek = jedna epizoda)
- `examples/mnist_eval.rs` — eval na test setu
- `data/` v `.gitignore`, download script v `scripts/get_mnist.sh`

**Exit kritérium:** ≥85 % test accuracy na MNIST. (Reference: dobré SNN mají 95–98 %, ale 85 % je důkaz, že architektura učí.)

**Otevřené otázky:**
- Pokud STDP+three-factor nedosáhne 85 %: **plán B** je surrogate gradient pretraining (jeden epoch klasickou backprop přes spike-based aktivace pomocí straight-through estimatoru), pak online fine-tuning lokální plasticitou. Tohle je hybridní cesta z dokumentu 09 sekce 7a.
- Pokud nefunguje ani plán B: **velký design review** než pokračovat. Možná je problém v topologii (chybí inhibiční neurony? lateral inhibition?) nebo encoding (rate vs. latency vs. population coding).

---

## Sprint 6 — Vyhodnocení a další směr (decision point, ne kódovací sprint)

**Cíl:** Rozhodnout, kam dál. Tady končí „základní výzkum" a začíná buď scale-up k LLM, nebo iterace na architektuře.

**Vstupy do rozhodnutí:**
- Výsledky MNIST (sprint 5)
- Profilace: kde se tráví čas, paměť per neuron, energie per spike
- Realistické zhodnocení: kolik neuronů můžeme uživit na našem HW při jaké rychlosti

**Možné větve:**
- **A: Scale-up.** CIFAR, pak sekvenční úkoly (Sequential MNIST, copy task, language modelling na malém korpusu). Cíl: dostat se k „spiking LM" na úrovni char-RNN.
- **B: Architektonické experimenty.** Spiking attention, recurrent topologie, multi-compartment dendrity, neuromodulační populace.
- **C: Efektivita.** Multi-thread engine, GPU prototyp (CUDA přes `cust` crate), exploratorní port na neuromorfní HW (Loihi 2 SDK).

**Toto rozhodnutí padne až po sprintu 5, ne dřív.** Předčasné větvení je nejčastější způsob, jak research projekt spadne.

---

## Co tento plán **záměrně neřeší**

- **LLM trénink.** Cesta k tomu je dlouhá; sprint 5 je první milník, kde má smysl o LLM vůbec mluvit.
- **Distribuovaný trénink, multi-GPU.** Problém pro daleko pozdější fázi.
- **Webové UI, vizualizace.** Pro debug stačí logging + Python notebook na post-processing.
- **Backwards compatibility.** Jsme ve výzkumné fázi, breaking changes jsou OK.
- **Benchmark vs. PyTorch SNN knihovny** (snnTorch, Norse, BindsNET). Až po sprintu 5 — porovnávat s referencí dává smysl, jen když máme co.

---
