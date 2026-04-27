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
- `src/scheduler.rs` — `EventQueue` nad `BinaryHeap<Event>`, min-heap přes `(t, seq)`; deterministický tiebreak podle pořadí vložení. `EventKind::Pulse { neuron, dendrite, amplitude }` sjednocuje synaptické dodávky i externí pulzy (provenance v hot pathu nehraje roli).
- `src/engine.rs` — `EventNetwork::run_until(t_end)`: pop event → `evolve_to(t)` cílového kompartmentu → inject → `check_threshold` → fan-out přes outgoing synapse.
- Refaktor LIF/Dendrite/Homeostasis: přidán `t_last_eval` a `evolve_to(t)` pro analytický rozpad. `tick()` zůstává jako tenký wrapper pro discrete režim.
- **Odchýlení od plánu:** discrete `Network` zůstává jako separátní typ vedle `EventNetwork` (ne za feature flagem) — parity test potřebuje oba současně. Oba typy mají stejné API (`add_neuron`/`add_synapse`/`schedule_input`).
- `benches/sparsity_scaling.rs` — criterion benchmark: 10k neuronů, FAN_OUT 20, T=0.05 s, vstupní rate 1/5/10/50 Hz. Porovnání event vs. discrete přes `iter_batched` (setup mimo měření).
- `tests/parity.rs` — chain stimulus, oba režimy musí dát stejné spike counts per neuron a fire times do tolerance 5×dt (discrete zaokrouhluje events na dt grid).

**Exit kritérium:** Parity test prochází. Bench při 1 Hz vstupu naměřil **24× speedup** event vs. discrete (plán: ≥10×). Při 50 Hz spadne na ~1.7× — event-driven má smysl jen v sparsity režimu.

**Otevřené otázky:**
- Paralelizace: zatím single-thread. Multi-thread (Rayon) odložit na sprint 6, pokud bude potřeba. Lock-free queue je velký kus práce — neotvírat dřív, než to bude bottleneck.
- Determinismus: explicitně seedovaný RNG (`rand::SeedableRng`), žádné `HashMap` v hot pathu (nedeterministické pořadí iterace).
- Sustained DC input v event módu: zatím nepodporováno (vyžadovalo by self-fire scheduling). Pro Sprint 4+ přidat, pokud bude potřeba.

---

## Sprint 4 — Three-factor learning + neuromodulace (~200 LOC)

**Cíl:** Přidat globální „dopaminový" signál `M(t)`, který gateuje STDP. Bez tohohle se síť reálně nenaučí nic netriviálního.

**Deliverables:**
- `src/neuromodulator.rs` — `Modulator { value, decay_tau, baseline }` s `reward(t, amount)` (zvedne `value`) a analytickým decay přes `evolve_to(t)` / `step(dt)`. Default `decay_tau=200ms`.
- `src/eligibility.rs` — `EligibilityParams { tau, learning_rate }`, default `tau=500ms`.
- Refaktor `Synapse`: `on_pre_spike` / `on_post_spike` vrací STDP delta (nemodifikuje `weight`); přidáno pole `eligibility`. Kdo deltu aplikuje, rozhoduje volající.
- Refaktor `Network`: when `eligibility_params + modulator` jsou nastavené, STDP delta se akumuluje do `eligibility`; per step `weight += η · M · e · dt` + decay. Default mode (None+None) = klasické STDP, beze změny chování.
- **Per-dendrite STDP routing** (`network.dendrite_local_stdp = true`): synapse cílící na dendrit dostávají post-spike STDP z firingu **dendritu**, ne somy. Klíčové pro credit-assignment v dendritických architekturách — bez něj všechny synapse cílící na neuron dostanou stejnou eligibility ze stejné post-spike, R-STDP nemůže rozlišit.
- `tests/eligibility_trace.rs` — trace decay přesně podle `exp(-Δt/τ_e)`; late reward (~τ_e/2 po pre-post páru) měří měřitelnou změnu váhy.
- `tests/reward_potentiation.rs` — opakované pre→post páry s rewardem rostou váhu; bez rewardu zůstává; negative reward depresuje.
- `tests/reward_modulated_xor.rs` — perturbovaný XOR detector (in1→A oslabeno na 5% magnitude místo correct -20%); R-STDP musí dotáhnout in1→A směrem k -inf bez poškození správně inicializovaných excitatory vah.

**Exit kritérium (revidované):** Mechanism testy procházejí (eligibility trace + reward potentiation). XOR test demonstruje **správný směr učení** pro perturbovanou váhu (Δin1→A < -0.005 za 60 epoch) při zachování excitatory váhy.

**Odchýlení od plánu (XOR 95% accuracy):** plné R-STDP XOR z náhodných vah na ≥95 % accuracy se nepodařilo. Symptom: čtyři synapse cílící na out-neuron dostávají identickou eligibility ze stejných (1,1) post-spiků, takže s globálním modulátorem nelze rozlišit, která má růst kterým směrem. Per-dendrite STDP routing tohle částečně řeší (správný směr učení), ale runaway excitatory potentiation překonává rychlost depresního učení. Standardní fix v literatuře: surrogate-gradient pretraining (Sprint 5 plán B) nebo population coding s WTA. Přesouváme plný XOR cíl na Sprint 5 jako součást MNIST stack.

**Otevřené otázky:**
- Tvar trace: exponenciální vs. dvojexponenciální. **Návrh: exponenciální, `τ_e = 500 ms`** (lit. konsensus pro dopaminové delay).
- Per-synapse plasticity flag (`plastic: bool`) by umožnil zafixovat hidden→output synapsy a omezil krádež eligibility na jiných cestách. Odloženo, dokud nepotřebujeme.

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
