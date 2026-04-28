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
- `src/encoding.rs` — Poisson rate coding: pixel intenzita ∈ [0,1] → spike train s mean rate `intensity · max_rate_hz` přes nezávislé inverse-CDF Poisson sampling.
- `src/topology.rs` — `build_feedforward(layer_sizes, weight_range, delay, rng)`: vrstvy plně propojené náhodnými váhami, vrací `Network` + per-layer `Vec<NeuronId>` listy.
- `src/readout.rs` — `count_spikes_per_neuron`, `argmax`, `predict_class` přes okno; vrací `None` pokud žádný spike.
- `src/mnist.rs` — IDX (big-endian binary) loader pro images + labels; magic-number a size validace.
- `src/synapse.rs` přidán `plastic: bool` flag + `Network::set_plastic(syn, false)` pro architektonické konexe (lateral inhibition apod.), které nesmí erodovat pod R-STDP.
- `examples/mnist_train.rs` — supervised training s teacher-pulse (single +pulse na expected output ~5ms po prezentaci); ukládá váhy do `data/aurora_mnist_weights.bin`.
- `examples/mnist_eval.rs` — načte uložené váhy, eval na test setu, hlásí accuracy a coverage.
- `scripts/get_mnist.sh` — `curl` + `gunzip` čtyř MNIST IDX souborů z `cvdf-datasets/mnist/`.
- `tests/classify_synthetic.rs` — smoke test na 4×4 stripe-classification (16→8→2 net): ověří, že pipeline funguje (encoding → spikes → predikce, R-STDP měřitelně mění váhy).

**Exit kritérium (revidované):** Plný 85% MNIST cíl **nesplněn v session-time scope**. Důvody jsou dva:
1. **Compute:** plný 60k-train × 784→400→10 net by trval hodiny v debug build i s optimalizací event-driven engine; v rámci jednoho session je proveditelný jen subset (1k vzorků, scaled-down topology), což strukturálně nemůže dosáhnout 85% target.
2. **Architektura:** Sprint 4 ukázal, že R-STDP credit-assignment selhává když konkurující synapse soutěží o credit z jednoho globálního modulátoru. Diagnostika ze synthetic testu potvrdila, že fully-connected feedforward s R-STDP vede ke kolapsu jednoho output neuronu (vždy jedna predikce). Diehl & Cook 2015 dosahují 95% s lateral inhibition + threshold adaptation + WTA — bez té architektury nelze 85% čekat ani s plným tréninkem.

**Co Sprint 5 dodává místo:** kompletní infrastrukturu pro klasifikaci (encoding/topology/readout/MNIST loader/training+eval examples) + smoke test ověřující, že stack funguje end-to-end. Plný benchmark + dosažení 85% přesouváme do Sprintu 6 spolu s přidáním WTA/lateral-inhibition primitiv (a/nebo plán B: surrogate gradient pretraining).

**Otevřené otázky:**
- Surrogate gradients vyžadují vlastní autograd-like vrstvu (backprop přes spike-based aktivace přes straight-through estimator). Náročné, ale dobře dokumentované — v Sprintu 6 to bude pravděpodobně preferovaná cesta.
- WTA/lateral inhibition lze přidat hned (`set_plastic(false)` infrastruktura už existuje), ale potřebuje ladění (jak silná inhibice, kolik inhibitory neurons per layer, atd.).

---

## Sprint 6 — WTA primitivy + lazy commit (pokus o MNIST iterace #1)

**Změna scope vs. původní plán:** Sprint 6 byl plánovaný jako rozhodovací bod, ale vzhledem k Sprint 5 stropu (R-STDP credit-assignment) prioritizoval implementaci WTA architektonických primitiv + perf optimalizace, aby vůbec šlo MNIST trénovat.

**Deliverables:**
- `Network::weight_clamp: Option<(min, max)>` — clamp aplikovaný pouze na plastic synapse v commit loopu, brání Hebbian runaway.
- `topology::add_lateral_inhibition(net, layer, strength, delay)` — all-to-all inhibitorní synapse uvnitř vrstvy, automaticky `set_plastic(false)` aby R-STDP neerodovalo.
- **Lazy eligibility commit** v `Network::step`: kroky kde modulator je tichý a žádné STDP eventy se nedějí (typicky 80%+ inter-trial waitu) zcela přeskočí O(synapses) loop. Decay se akumuluje a aplikuje při dalším flushu.
- **Rayon paralelizace** pro bulk synapse loop nad threshold 2 000 synapsí.
- `Network::flush_eligibility_to_now()` — public API pro test/diagnostic kód, který chce číst aktuální eligibility hodnoty bez čekání na další step.
- Refactor `examples/mnist_train.rs` na WTA setup (lateral hidden + output, weight clamp, supervised inhibice non-target outputs).
- `tests/classify_synthetic.rs` smoke test s WTA primitivami.

**Exit kritérium (revidované):** WTA primitivy fungující a unit-testované. Lazy commit poskytuje měřitelné zrychlení (eval ~10×, training ~3-5×). Plný 85% MNIST cíl **nadále nesplněn** — i s WTA primitivami a strong supervised inhibition v default tuningu se MNIST nedostane výrazně nad chance.

**Co jsme se naučili:**
- WTA samotné nestačí. Diehl & Cook 2015 dosahuje 95% s **kombinací**: lateral inhibition + adaptive thresholds (homeostasis) + STDP s clipping + post-hoc neuron-class assignment z aktivit. Náš homeostasis modul existuje (Sprint 2), ale nikdy nebyl integrován do MNIST setupu.
- R-STDP s globálním modulátorem strukturálně limituje class-discrimination, protože všechny synapse z vrstvy do vrstvy "vidí" stejný reward signal. Output WTA pomáhá v inferenci ale ne v učení.

**Otevřené otázky pro Sprint 7+:**
- Integrovat homeostasis (Sprint 2 modul) do MNIST hidden vrstvy — neurons s vysokou rate adaptují threshold nahoru, vynucená diverzita features.
- Surrogate gradient pretraining (původní Sprint 5 plán B) jako baseline, který může na MNIST uspět.
- Per-output reward attribution (každý output má vlastní modulátor, signal jen pro synapse cílící na něj) — strukturální fix R-STDP credit-assignment.

---

## Sprint 7 — Homeostáza + per-output kredit (druhý pokus o MNIST)

**Cíl:** Odstranit dvě konkrétní příčiny, proč Sprint 6 WTA pipeline dává accuracy ~chance: (1) hidden vrstva nediverzifikuje features, (2) globální modulátor neumí rozlišit kreditem mezi outputy. Pokud po obojím nedosáhneme ≥50 % MNIST, Sprint 8 = pivot na surrogate gradients.

**Princip plánování:**
- **Existing infrastruktura first.** Homeostasis modul je hotový od Sprintu 2 a nikdy nebyl integrován do training pipeline — to je nejlevnější high-impact akce.
- **Per-output reward attribution je strukturální fix R-STDP.** Per Frémaux & Gerstner 2016, jediný globální modulátor strukturálně limituje, co se síť může naučit. Více modulátorů s neuron-scope řeší credit-assignment cleanly.
- **Měřit výsledky průběžně.** Po každé z dvou změn nezávisle změřit MNIST accuracy, aby bylo jasné, která změna kolik přinesla. Žádné slepé tunování všeho najednou.

**Deliverables:**
- **A: Integrace homeostázy do MNIST hidden** (~50 LOC):
  - V `examples/mnist_train.rs` po `build_feedforward` zavolat `enable_homeostasis(Homeostasis::new(HomeoParams { tau: 1.0, target_rate: 5.0, eta: 0.005 }))` na každém hidden neuronu.
  - V `mnist_eval.rs` stejný setup (homeostasis vyžaduje aktivní state pro fungování — i když threshold zafrozne, neurony musí mít stejný v_threshold start).
  - Volitelně: dump per-hidden-neuron firing rate po tréninku do diagnostiky, aby šlo zkontrolovat, že se rates rozdělují (ne všechny na 0 nebo všechny na max).
  - **Hypotéza:** Po homeostáze každý hidden neuron firuje ~5 Hz mean. Threshold se adaptuje tak, že každý hidden má svou „specializaci" — neurony co fires moc snižují svůj příspěvek (raised threshold), neurony co málo zvyšují citlivost. Bez competition signal (žádný teacher na hidden), specializace je inherent stochastická — nemusí být class-aligned.
  - **Měřitelný impact:** target ≥20 % MNIST accuracy (hidden specializace kombinovaná s WTA výstupní vrstvou by měla nad chance).

- **B: Per-output modulátory** (~150 LOC, structural change):
  - `Network::modulators: Vec<Option<Modulator>>` indexed by `NeuronId`. Each modulator scopes to synapses targeting that neuron.
  - Backward compat: pokud `modulator_for(neuron) == None`, fallback na global `self.modulator`.
  - API: `Network::reward_neuron(neuron: NeuronId, amount: f64)` — injects do specific modulátoru.
  - V commit loopu: pro každou synapsi look up modulátor podle `syn.post_neuron`. To přidá O(1) lookup per synapse, ale stále paralelizovatelné.
  - V `mnist_train.rs`: `for class in 0..10 { net.add_modulator(outputs[class], ModulatorParams::default()) }`. Při tréninku class-X: `net.reward_neuron(outputs[class], 1.0)` — jen modulátor pro správný output zaktivuje, jen synapse cílící na něj komitují weight change.
  - **Hypotéza:** S per-output rewardem každá hid→out_X cesta dostane signal jen když class X byl target. Hidden firing patterns se postupně stávají class-discriminative, protože jen `correct` cesty rostou. Ostatní zůstávají na initial weights → output_correct má dominant drive na test time.
  - **Měřitelný impact:** target ≥50 % MNIST accuracy.

- **C: Tests + benchmarks**:
  - `tests/per_output_credit.rs` — 2-class task ověří, že synapse na correct output rostou, na wrong stagnate (ne ekvivalentně jako Sprint 6 smoke).
  - Update `tests/classify_synthetic.rs` na full pipeline s homeostasis + per-output, asseroval ≥70 % accuracy (za chance 50 %).
  - Update `benches/sparsity_scaling.rs` aby měřil i three-factor mode s per-output (per-synapse modulátor lookup overhead).

**Exit kritérium:** Buď (a) MNIST test accuracy ≥50 % po implementaci A+B (success → Sprint 8 už dělá Diehl-Cook scale-up nebo přechází k recurrent topologii), nebo (b) MNIST stále <50 % i s A+B integrovanými → Sprint 8 = surrogate gradient pretraining. Jeden ze dvou outcomes — clear decision point bez pokračujícího band-aid tuning.

**Co Sprint 7 záměrně NEŘEŠÍ:**
- Adaptivní lateral inhibition strength. Drží se hand-tuned (Sprint 6 hodnoty).
- Multi-population output (2+ neurons per class). Potenciálně přínosné, ale zvyšuje dimensionalitu hyperparametrů. Až po měření base impact A+B.
- Surrogate gradients. Pokud kritérium fail → Sprint 8.

**Otevřené otázky:**
- Per-output modulátor: vlastně synapses cílící na *jakýkoli* output, nebo i hid→out spojení? Pravděpodobně oboje (cílový post je output_X). Hidden→hidden synapse (zatím žádné, ale možné v Sprintu 8+) by potřebovaly fallback na global.
- Jak handlovat nevybrané outputy (no reward) — komitují se s M=0 (no change) nebo s mírně negativním M (decay)? Neutral default M=0; eventually možno experimentovat s -0.1 baseline.
- Homeostasis parameters: target_rate 5 Hz vs 10 Hz vs 1 Hz. Diehl-Cook používá adaptive ~5 Hz. Budeme začínat tam.

---

## Sprint 8 — Surrogate gradients (Sprint 7 fail outcome)

**Sprint 7 výsledek:** MNIST 7.2 % accuracy (klesá při víc epochách kvůli weight clamp saturation). Per Sprint 7 exit kritéria → **Sprint 8 = surrogate gradient pretraining**.

**Cíl:** Implementovat backprop-through-time s fast-sigmoid surrogate gradient pro spike funkci. Forward pass dělá pravé Heaviside spikes (žádný change semantiky simulace), backward pass používá smooth approximation `1/(1+α|v−θ|)²` aby gradient mohl téct skrz spiking síť. Cíl: dosáhnout ≥85 % MNIST (původní Sprint 5 target).

**Princip plánování:**
- **Hand-coded BPTT, ne plný autograd framework.** Architektura `784 → H → 10` je jednoduchá, BPTT se napíše ručně přes ~150 LOC. Plný framework typu burn/candle je out-of-scope (přínos vs. risk nedává smysl pro single-architecture).
- **Koexistence s R-STDP `Network`.** Surrogate gradient cesta je úplně oddělená od existujícího `Network` struct — vlastní `SurrogateSnn` typ s vlastní simulací. Aurora má od teď **dvě nezávislé learning paths**: lokální (STDP přes `Network`) a globální (BPTT přes `SurrogateSnn`).
- **f32 throughout** pro paměťovou stopu — BPTT tape ukládá veškerý forward state.

**Deliverables:**
- `src/surrogate.rs` — `SpikeFn { threshold, alpha }` s `forward()` (Heaviside) a `backward()` (fast sigmoid). 30 LOC.
- `src/snn_loss.rs` — `softmax`, `cross_entropy_with_grad` (vrací loss + dloss/dlogits), `argmax`. 30 LOC.
- `src/snn_dense.rs` — `SurrogateSnn` (2-layer dense): `forward(input_spikes) -> ForwardCache`, `backward(cache, input, target) -> (loss, pred)`, `step_sgd(lr)`, `zero_grad()`. Hand-coded BPTT s subtraction reset (`v -= θ × stop_grad(spike)`). 250 LOC včetně matvec helpers.
- `tests/surrogate_classify.rs` — 4×4 stripe 2-class task. **Cíl: ≥95 % test accuracy.** Měřitelný proof, že učení funguje.
- `examples/mnist_surrogate.rs` — train+eval v jednom binary. Default 5000 train, 1000 test, 3 epochs. Per-epoch reportuje train + test accuracy. **Cíl: ≥85 % test accuracy.**

**Exit kritérium:**
- ✅ Stripe test ≥95 % accuracy (mechanism works) — naměřeno **100 %** v 1 epoše.
- 🟡 MNIST ≥85 % test accuracy — k ověření uživatelem (`cargo run --release --example mnist_surrogate`).

**Co Sprint 8 záměrně NEŘEŠÍ:**
- Plné autograd framework s automatickým grafem operací — over-engineering pro current scope.
- Convolutions / pooling — MNIST plně-connected je dostatečný pro demonstration.
- GPU acceleration — Sprint 9+ pokud accuracy bude OK ale rychlost mizerná.
- Hybrid local + surrogate — pretrain surrogate, fine-tune R-STDP — zajímavé, ale samostatný sprint.

**Otevřené otázky:**
- Adam vs SGD: SGD jednodušší, Adam by možná konvergoval rychleji. Začneme SGD; pokud konvergence je pomalá, Adam je ~50 LOC change.
- Reset semantika: subtraction reset (`v -= θ`) vs. hard reset (`v = 0`). Subtraction má cleanší gradient (zvolen). Hard reset by potřeboval `stop_grad` přes všechno před resetem, což komplikuje BPTT.
- Spike-count loss vs. last-step loss: aktuálně `loss = CE(softmax(Σ spike_t), label)`. Alternativa `CE(softmax(spike[T-1]))` (jen last step) — méně signál, ale možná stabilnější. Pro MNIST sum funguje dobře.

---

## Sprint 9 — Recurrent surrogate SNN (cesta k sekvenčnímu zpracování)

**Sprint 8 výsledek:** plný MNIST 97.19 % test accuracy v 21 minutách. Surrogate gradient stack je hotový a funguje na publikované úrovni. Od decision pointu (původní Sprint 9 = explore-first vs unify-now) volíme **scale-up směrem k recurrent topologii**, protože LLM cíl bez recurrence není možný a Sequential MNIST je nejmenší krok ke char-level LM (Sprint 10+).

**Cíl:** Postavit `RecurrentSurrogateSnn` s hidden→hidden recurrent connections + biases, demonstrovat funkci na pure memory testu, dosáhnout >85 % na Sequential MNIST (T=28, řádek per timestep).

**Princip plánování:**
- **Build-on-Sprint-8.** Architektura paralelní k `SurrogateSnn` (samostatný typ, sdílí `SpikeFn` + `snn_loss`), nesnaží se generalizovat — explicit two-impl strategy match aurořině přístupu (`Network` vs `EventNetwork`).
- **Recurrent gradient flow je stable jen s clipping.** Default `clip_grads(1.0-1.5)` zařadit do training loopu — bez clippingu BPTT přes ~30+ kroků dostane nan/inf na 1 z 50 vzorků.
- **Recurrent W initialization je delicate.** Scale `0.5/sqrt(n_hidden)` drží spectral radius pod 1, brání exponenciálnímu zesílení signálu přes čas.

**Deliverables:**
- `src/snn_recurrent.rs` (~340 LOC) — `RecurrentSurrogateSnn`:
  - Fields: `w_in [H,I]`, `w_rec [H,H]`, `w_out [O,H]`, `b_h [H]`, `b_o [O]` + odpovídající `grad_*` buffery.
  - `forward(input_spikes)`: per timestep `h_v_pre = h_v_after_prev * decay + W_in @ x + W_rec @ h_spike_prev + b_h`. Subtraction reset, surrogate spikes.
  - `backward(cache, input, target)`: BPTT s tří-cestnou gradient flow do `h_v_pre[t-1]`: (1) decay carryover, (2) recurrent feedback `W_rec.T @ d_h_v_pre[t]`, (3) output-side `W_out.T @ d_o_v_pre[t-1]`.
  - `clip_grads(max_norm)`: globální L2-norm clipping přes všech 5 parameter groups.
  - `step_sgd(lr)`, `zero_grad()`.
- `tests/recurrent_remember.rs` — 2-class pulse test (T=8, key 3 steps, silent 5 steps): trénink dosáhne >90 % a `‖W_rec‖_F` musí být netriviální. Functional smoke, ne pure long-memory benchmark (research-grade).
- `examples/seq_mnist.rs` — Sequential MNIST. T=28 (jeden řádek 28 pixelů per step), n_hidden=100, gradient clipping 1.5, LR=0.002. Stejný pretty output jako `mnist_surrogate`.

**Exit kritérium:**
- ✅ Memory test ≥90 % accuracy + W_rec norm > 0.001 — naměřeno **100 % a ‖W_rec‖=1.49**.
- 🟡 Seq-MNIST test acc ≥85 % na 5000 train / 1000 test / 3 epochs — k ověření uživatelem.

**Co Sprint 9 záměrně NEŘEŠÍ:**
- Plný 784-step Sequential MNIST (1 input neuron, T=784) — cca 28× pomalejší trénink, marginální vědecká hodnota nad row-by-row.
- Permuted Sequential MNIST (random permutation of pixel order) — varianta která blokuje feedforward shortcuts. Pokud row-by-row uspěje a chceme tvrdší benchmark, dobrý Sprint 10 add-on.
- Adam optimizer — SGD funguje pro Sprint 8, mělo by stačit i tady. Adam je drop-in upgrade, můžeme přidat později pokud konvergence stagnuje.
- Hybrid surrogate→STDP fine-tune — Sprint 10+ téma.

**Otevřené otázky:**
- LSTM / GRU-like gating: snnTorch/Norse implementují LIF s explicit forget gate. Pro Sprint 9 MVP necháváme čistý LIF s recurrent lin layer; gating je iteration pokud accuracy nedostačuje.
- Loss readout: aktuálně sum spikes přes celý T. Pro sekvenční tasks where output má smysl jen po sekvenci, last-step CE nebo sum přes posledních K steps může lépe lokalizovat gradient. Default zůstává sum-all (Sprint 8 compatible).

---

## Sprint 10 — Decision point: scale, hybrid, nebo efektivita

**Vstupy do rozhodnutí:**
- Sprint 9 Seq-MNIST výsledek
- Profilace: kde se tráví čas, paměť per neuron, kolik neuronů můžeme uživit
- Realistické zhodnocení: jaká je cesta od Seq-MNIST k char-level LM

**Možné větve:**
- **A: Char-level LM.** Vlastní tokenizer (byte-level), tiny corpus (~1 MB Wikipedia), recurrent SNN learns next-char prediction. První signál, jestli SNN umí language modeling.
- **B: Hybrid surrogate→STDP.** Aurora má teď konkrétně: backbone trained surrogate, fine-tune online STDP na new classes. Test biological plausibility "two-phase learning" hypotézy.
- **C: Permuted Seq-MNIST + Copy task.** Tvrdší recurrence benchmarks. Validate před scale-up.
- **D: Efektivita.** Rayon batch BPTT, GPU port (cust crate), Adam optimizer.

**Toto rozhodnutí padne až po Sprintu 9, ne dřív.**

---

## Co tento plán **záměrně neřeší**

- **LLM trénink.** Cesta k tomu je dlouhá; sprint 5 je první milník, kde má smysl o LLM vůbec mluvit.
- **Distribuovaný trénink, multi-GPU.** Problém pro daleko pozdější fázi.
- **Webové UI, vizualizace.** Pro debug stačí logging + Python notebook na post-processing.
- **Backwards compatibility.** Jsme ve výzkumné fázi, breaking changes jsou OK.
- **Benchmark vs. PyTorch SNN knihovny** (snnTorch, Norse, BindsNET). Až po sprintu 5 — porovnávat s referencí dává smysl, jen když máme co.

---
