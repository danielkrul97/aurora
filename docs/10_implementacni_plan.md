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

## Sprint 10 — Char-level language model (cesta k LLM)

**Sprint 9 výsledek:** Seq-MNIST 71.8 % s Adamem (vs 37 % SGD). Recurrence funkční, vanilla baseline na úrovni literatury. Per Sprint 9/10 decision point + dlouhodobý cíl projektu (CLAUDE.md: "custom LLM") **volíme větev A: char-level LM** — první konkrétní krok od MNIST-style klasifikace ke generativnímu modelu nad text.

**Cíl:** Postavit byte-level LM trained next-byte prediction na recurrent SNN. Měřit bits-per-byte (BPB) jako standardní metric. Validovat, že stack vůbec dokáže reduce BPB pod uniform random (8.0).

**Princip plánování:**
- **Byte-level (256 vocab), žádný tokenizer.** Eliminuje vocabulary engineering, výsledky comparable s canonical char-RNN benchmarks.
- **Non-spiking lineární output.** SNN LM literatura (Lee 2020) standardně používá `logits = W_out @ h_spike + b_o` bez output spike fn — output spikes by byly noisy sampling z true distribution. Forward přes hidden zůstává spiking.
- **Per-step cross-entropy loss + truncated BPTT.** Train na chunks (default 64 bytes), full per-step gradient, žádná state persistence mezi chunks pro train (zjednodušuje implementaci).
- **State persistence pro inference.** `step_lm()` API drží `SnnState` přes volání pro autoregressive sampling.

**Deliverables:**
- `src/text_io.rs` — byte-level utility: `load_corpus`, `one_hot`, `sample_byte` s temperature, `bits_per_byte`. ~70 LOC.
- `RecurrentSurrogateSnn` rozšíření (~210 LOC navíc):
  - `LmCache { h_v_pre, h_v_after, h_spike, logits }` (logits per timestep, ne spike counts).
  - `forward_lm(input_seq) -> LmCache` — non-spiking output projection.
  - `backward_lm(cache, input, targets) -> nats` — per-step CE loss s gradient flow přes hidden surrogate + W_rec.
  - `SnnState { h_v_after, o_v_after, h_spike }` + `fresh_state()`.
  - `step_lm(input, &mut state) -> logits` — streaming inference pro generaci.
  - `step(input, &mut state)` — alternativa pro classification streaming.
- `tests/char_lm_smoke.rs` — predikovatelný 8-byte cycle × 64 = 512 bytes corpus. Cíl: BPB < 4.0 (uniform 8.0). **Naměřeno: 1.35 BPB za 6 epoch.**
- `examples/char_lm.rs` — full pipeline na uživatelem-poskytnutý corpus (default `data/tiny.txt`). 90/10 train/val split, per-epoch sample generation s temperature, pretty box output. Hint pro tiny-shakespeare download.

**Exit kritérium:**
- ✅ Smoke test BPB < 4.0 — naměřeno **1.35** v 6 epoch.
- 🟡 Tiny-shakespeare BPB ≤ 3.0 (canonical char-RNN s LSTM dosahuje 1.5; vanilla recurrent SNN 2.5-3.5 je realistický cíl) — k ověření uživatelem.

**Co Sprint 10 záměrně NEŘEŠÍ:**
- LSTM-style gating (Lee 2020) — pokud BPB nad 3.0, Sprint 11 add-on.
- Stateful BPTT přes chunk boundaries (carry hidden state s `detach`) — možné Sprint 11 zlepšení pro long-range dependencies.
- Mini-batching — single-sample forward/backward, scaling až bude potřeba.
- Sub-byte / BPE tokenization — kompromituje "language-agnostic" výhodu byte-level. Ne na agendě.

**Otevřené otázky:**
- Reset hidden state mezi chunks vs. carry-with-detach. Aktuálně reset (jednodušší). Carry by zachytil multi-chunk kontext, ale BPTT zůstává truncated na chunk length. Lee 2020 carry-with-detach.
- Output decay vs. žádné decay. Aktuálně žádné (čistá projekce). Decay by hrál roli jen pokud loss není per-step.
- Generation determinism: temperature 0.0 = greedy, ale recurrent SNN má spike randomness ze hidden — generation není deterministická i při T=0.

---

## Sprint 11 — Gated LIF cell (LSTM-style forget gate)

**Sprint 10 výsledek:** tiny-shakespeare 3.115 BPB v 3.5 minutách, slovní fragmenty v generaci. Pod bigram baseline (3.7), nad LSTM char-RNN baseline (1.5). Sprint 10/11 decision point → **B (gated LIF)**: kategorialní upgrade kapacity sítě.

**Cíl:** Nahradit fixed `decay = exp(-dt/τ)` v recurrent SNN cell **per-neuron, input-dependent forget gate** `β_t = sigmoid(W_β_in @ x_t + W_β_rec @ h_spike_{t-1} + b_β)`. Membrane integrace se stane `h_v_pre = h_v_after_prev * β_t + ...`. Síť se naučí, kdy "pamatovat" (gate ≈ 1) a kdy "resetovat" (gate ≈ 0) per neuron, per timestep.

**Princip plánování:**
- **Nový typ vedle existujícího.** `GatedRecurrentSnn` paralelní k `RecurrentSurrogateSnn`, ne jeho rozšíření. Pure baseline zůstává jako reference pro benchmark; Aurora má teď "vanilla" a "gated" verzi recurrent stacku.
- **Bias = 1.0 init.** Jozefowicz et al. 2015 LSTM-forget-gate-bias-1 trick: počáteční sigmoid(1.0) ≈ 0.73 znamená "remember by default", brání tomu aby gate okamžitě zničil hidden state.
- **Sparse-by-default.** Stejné techniky jako Sprint 10: one-hot input (column read), binary spike (skip nulové sloupce). Gate má vlastní `W_gate_in` / `W_gate_rec` váhy se stejným sparse paternem.
- **Žádný cell state separátní od membrány.** GRU-style simplifikace nad full-LSTM: jen forget gate, žádný input gate ani output gate. Output stále non-spiking lineární projekce hidden spikes (jako Sprint 10).

**Deliverables:**
- `src/snn_gated.rs` (~480 LOC) — `GatedRecurrentSnn`:
  - 5 váhových matic: `W_in`, `W_rec`, `W_out`, `W_gate_in`, `W_gate_rec`
  - 3 bias vektory: `b_h`, `b_o`, `b_gate` (init `1.0` pro `b_gate`)
  - `forward_lm_bytes(&[u8]) -> GatedLmCache` se sparse path
  - `backward_lm_bytes(&cache, &[u8], &[u8]) -> nats` s BPTT přes 4 paths (decay carryover přes gate, recurrent W_rec, recurrent W_gate_rec, output W_out)
  - `step_lm_byte(byte, &mut state) -> logits` pro autoregressive generaci
  - `clip_grads`, `step_adam`, `zero_grad` přes všech 8 parameter groups
- `tests/gated_lm_smoke.rs` — 8-byte cycle. **Naměřeno: 0.92 BPB v 6 epochách** (vanilla měla 1.35 → gating zlepšil o 35 % na stejném tasku).
- `examples/char_lm_gated.rs` — drop-in replacement pro `char_lm` s gated cell, stejné CLI args, stejný pretty output.

**Exit kritérium:**
- ✅ Smoke test < 1.5 BPB — naměřeno **0.92** (vs vanilla 1.35).
- 🟡 tiny-shakespeare BPB ≤ 2.5 (vs vanilla 3.12) — k ověření uživatelem.

**Co Sprint 11 záměrně NEŘEŠÍ:**
- Full LSTM (input + output gate). Forget gate sám většinou stačí pro kratko-dobé memory tasks; full LSTM přidává komplexitu výměnou za marginal capacity gain.
- Stateful BPTT přes chunk boundaries (carry-with-detach). Pokud Sprint 11 BPB stagnuje, Sprint 12 add-on.
- Mini-batching. Sparse cesta nás drží v "OK rychlosti"; batch je optimalization pro Sprint 12+.

**Otevřené otázky:**
- Init scale `W_gate`: `0.1 × scale_main` znamená sigmoid input je dominován bias=1.0 → sigmoid(~1.0) ≈ 0.73 počáteční gate. Pokud gating se nedostatečně diferencuje per-neuron, zvýšit init scale.
- Gate-gradient flow přes silně nasycený sigmoid: pokud gate je ≈ 0 nebo ≈ 1, derivative `g(1-g)` ≈ 0 → vanishing gradient. Surrogate gradient pro sigmoid jsme nezavedli (na rozdíl od Heaviside) — pokud je problém, lze.

---

## Sprint 12 — Scale-up + mini-batching (cesta z 2.71 BPB níž)

**Sprint 11 výsledek:** tiny-shakespeare gated 2.705 BPB (vanilla 3.115 → −0.41 zlepšení), ale lehký overfitting po 1 epoše ukazuje strop kombinace 256 hidden + 1 MB corpus + bez regularizace. Sprint 12 řeší obojí: **A (scale-up: hidden 512, větší corpus) + C-batching (mini-batch parallel BPTT)**.

**Co Sprint 12 ZÁMĚRNĚ NEŘEŠÍ:** GPU port. Plný GPU port (CUDA přes `cust` nebo wgpu compute shaders) je 1000+ LOC a vyžaduje paralelní parity testy proti CPU. Mixovat to s mini-batchingem v jednom sprintu = dva nezávislé risky najednou. **Odloženo na Sprint 13.**

**Princip plánování:**
- **Externalizovat gradienty.** Místo `grad_*` polí uvnitř `GatedRecurrentSnn` zavést `GatedGrads` struct vedle. Per-thread workers v batched trainingu mají vlastní `GatedGrads` (sdílí read-only weights), main thread sumuje. Standard PyTorch-style separace weights/grads.
- **Rayon přes batch dim.** Forward + backward každého samplu nezávisle. Per-thread `GatedGrads` accumulators se sumují deterministicky do `self.grads` po dokončení paralelního pass.
- **Hidden 512 a větší corpus** řeší overfitting strop. 4× víc params, ale paralelní batching to z části vykompenzuje (parallel speedup). Kombinace by měla dotlačit BPB pod 2.0.

**Deliverables:**
- `src/snn_gated.rs`:
  - `GatedGrads` struct (8 grad bufferů) s `zeros`, `zero`, `add`, `l2_norm`, `scale`.
  - Refactor: `grad_*` fields → single `grads: GatedGrads` field. `zero_grad`, `clip_grads`, `step_adam` čtou z `self.grads`.
  - `backward_lm_bytes_into(&self, cache, inputs, targets, grads: &mut GatedGrads) -> f32` — read-only na self, zapisuje do externí grad bufferu. Public API.
  - `backward_lm_bytes(&mut self, ...)` — thin wrapper přes `mem::take` zachovává zpětnou kompatibilitu.
  - `forward_backward_batch(&mut self, batch_inputs: &[&[u8]], batch_targets: &[&[u8]]) -> f32` — rayon parallel batch, sums per-thread grads do `self.grads`.
- `tests/gated_batch_parity.rs`:
  - `batch_grads_match_serial_sum_within_fp_tolerance` — batch == serial sum (FP tolerance 1e-5)
  - `batched_training_progresses` — Adam s batch reduces loss
- `examples/char_lm_gated.rs` — refactored: `N_HIDDEN=512`, `BATCH_SIZE=16`, contiguous chunks z corpu (každý batch sample má vlastní hidden state).
- `scripts/get_corpus.sh` — multi-corpus download: `tiny`, `shakespeare-complete`, `wikitext2`.

**Exit kritérium:**
- ✅ Batch parity test prochází (max FP diff < 1e-5)
- ✅ Batched training redukuje loss
- 🟡 tiny-shakespeare s hidden=512 + batch=16: BPB ≤ 2.0 (vs Sprint 11: 2.71) — k ověření uživatelem
- 🟡 Sprint 12 wall-clock ≤ Sprint 11 wall-clock i přes 4× víc params (mini-batching kompenzuje)

**Otevřené otázky:**
- Stateful BPTT přes batch chunks (carry h_v_after / h_spike s detach mezi po sobě jdoucími chunks ve stejné batch sequence). Aktuálně reset = 0 mezi chunks. Lee 2020 carry-with-detach. Pokud BPB stagnuje, dobrý add-on.
- Optimal batch size: 16 default, ale 32-64 by mohlo být rychlejší na multi-core CPU. CPU má 4-16 jader; batch > num_cores má klesající návratnost.
- Adam state checkpoint: ne v Sprintu 12 scope. Pokud chceme resume training, Sprint 13 add-on.

---

## Sprint 13 — GPU port (původně Sprint 12 odložené C)

**Sprint 12 status:** mini-batching dodaný (deterministická batch-parita), Czech corpus stack hotový. GPU port byl explicitně odložen na Sprint 13. Aktuální CPU rychlost: ~3-5 min na 1 MB corpu × 3 epochy. Pro scale-up na 10-100 MB text nebo hidden ≥1024 budeme potřebovat GPU.

**Cíl:** Portovat `GatedRecurrentSnn` forward + backward na GPU, dosáhnout **≥10× speedup** vs CPU multi-thread (Sprint 12 baseline). Zachovat numerickou paritu s CPU implementací (FP tolerance ~1e-4 kvůli SIMD reordering).

**Princip plánování:**
- **wgpu, ne CUDA.** Vyhodnocení frameworků:
  - `cust` (CUDA): nejrychlejší, ale NVIDIA-only, vyžaduje toolchain. Nepřizpůsobivé na uživatelově HW.
  - `wgpu` + WGSL: cross-platform (Vulkan/Metal/DX12), pure Rust dependency, kernel jazyk WGSL je uchopitelný. Performance ~70-90 % CUDA.
  - `cubecl`: Rust-native abstraction nad wgpu/cuda. Mladší, méně dokumentace.
  - **Volba: wgpu** — risk: hardware compatibility = nízký, debugging = vlastní (žádný cuda-gdb), perf strop = ~85 % CUDA.
- **Inkrementální parita.** Implementovat po vrstvě: nejdřív GPU matvec (jádro forward), pak hidden cell, pak batch dim, pak backward. Po každém kroku CPU↔GPU parity test. Žádný "big bang" port.
- **Jen `GatedRecurrentSnn` LM cesta.** Vanilla `RecurrentSurrogateSnn`, MNIST examples zůstávají CPU — jejich rychlost je dostatečná. GPU stack je `gated_lm_*` only.
- **Float precision.** GPU shadery jsou f32 only. Naše CPU implementace už je f32, takže přechod je clean. Souborový formát checkpointů zůstává f32 little-endian.

**Deliverables (~1500 LOC):**
- `src/gpu/mod.rs` — `GpuContext` (instance, adapter, device, queue), single-instance držený přes lifetime training.
- `src/gpu/buffers.rs` — typed wrappers nad `wgpu::Buffer`: `GpuMatrix [rows×cols]`, `GpuVec [n]`, helpery `upload_from_slice`, `download_to_vec`. Shared storage policy.
- `src/gpu/kernels.wgsl` — WGSL compute shadery:
  - `matvec_sparse_onehot` — `out[h] = W[h*I + byte_idx]` (column copy)
  - `matvec_sparse_binary` — `out[h] = sum over c where mask[c]==1 of W[h*N + c]`
  - `cell_step_forward` — fused gate/membrane/spike per timestep
  - `cell_step_backward` — fused backward s surrogate gradient
  - `adam_step_apply` — vectorized Adam update přes parameter group
- `src/snn_gated_gpu.rs` — `GatedRecurrentSnnGpu` mirror struktura `GatedRecurrentSnn`. Public API:
  - `from_cpu(net: &GatedRecurrentSnn, ctx: &GpuContext) -> Self` — upload weights
  - `to_cpu(&self) -> GatedRecurrentSnn` — download weights
  - `forward_backward_batch_gpu(&mut self, batch_inputs, batch_targets) -> f32`
  - `step_adam_gpu`, `clip_grads_gpu`, `zero_grad_gpu`
- `tests/gpu_parity.rs` (gated by `#[cfg(feature = "gpu")]`):
  - Forward parity: GPU vs CPU output difference < 1e-4 max
  - Backward parity: gradient difference < 1e-3 (chyba akumuluje přes BPTT)
  - End-to-end training: 10 batched steps GPU vs CPU stejný direction
- `benches/lm_gpu_speedup.rs` — měření throughput chunks/sec GPU vs CPU multi-thread:
  - Hidden 256, 512, 1024
  - Batch 16, 32, 64
  - Identifikovat batch threshold kde GPU vyhraje
- `examples/char_lm_gated_gpu.rs` — drop-in replacement `char_lm_gated` s `--gpu` flag (default), fallback na CPU pokud `wgpu` initialization fails.
- `Cargo.toml`: `wgpu = "23"` (nebo aktuální stable), volitelný `bytemuck` pro safe pod casts. Feature flag `gpu` aby CPU-only build šel postavit bez GPU dependencies.

**Exit kritérium:**
- Forward parity < 1e-4 max diff vs CPU
- Backward parity < 1e-3 max diff per gradient buffer (BPTT accumulation tolerance)
- End-to-end training na char_lm_gated tiny-shakespeare s GPU: BPB ≤ Sprint 12 výsledek + 0.1 (allow for FP noise)
- Throughput benchmark: hidden=512 batch=16 → **≥5× wall-clock** speedup vs CPU multi-thread; hidden=1024 batch=32 → **≥10×**

**Co Sprint 13 záměrně NEŘEŠÍ:**
- CUDA backend přes `cust`. Pokud wgpu performance je nedostatečná, můžeme přidat jako Sprint 14 add-on.
- GPU port `RecurrentSurrogateSnn`, `SurrogateSnn`, MNIST examples. Ne core LM cesta.
- Multi-GPU. Zatím irrelevant.
- Mixed precision (f16). f32 je default; f16 je další optimization sprint.
- Stateful BPTT (carry-with-detach). Pokud Sprint 13 dodá speedup, Sprint 14 = stateful BPTT na GPU.

**Otevřené otázky:**
- **Memory layout.** Row-major vs column-major matic v GPU paměti — určuje paměťové hits při matvec/matmul. Default row-major (jako CPU); pokud benchmark ukáže problém, transpose před uploadem.
- **Bind group reuse.** Vytvořit bind groups jednou per `GatedRecurrentSnnGpu` instance, ne per `forward_backward_batch_gpu` volání. Drobnost ale významná pro overhead při malém batch size.
- **Backward fusion vs. unfusion.** Naivní implementace: 3-4 separate compute passes per timestep (gate gradient, h_v_pre gradient, weight grad accumulation). Fused: jeden pass přes hidden dim. Začneme unfused (čitelnost + ladění); fusion jako optimization v Sprintu 14.
- **wgpu version churn.** wgpu major versions break často. Pinout exact verze v Cargo.toml + plan na update jednou ročně.
- **Tooling.** RUST_LOG=wgpu_core=info pro debug. Není GPU profiler v ekosystému stejně dobrý jako Nsight; spoléháme na wall-clock benchmarks.

**Risks (a mitigation):**
- **R1: wgpu setup fail na uživatelově HW.** Mitigation: detekce při `GpuContext::new`, jasný error message, fallback na CPU implementaci v `char_lm_gated_gpu`.
- **R2: numerická divergence shaderů.** Mitigation: parity testy po každém kroku, ne až na konci. WGSL `f32` IEEE 754 dělá rounding stejně jako Rust f32 jen reordering ovlivňuje sumy.
- **R3: BPTT memory.** Pro chunk T=64, hidden=512, batch=16: forward cache = 4 × 64 × 16 × 512 × 4 bytes = 8.4 MB per layer. OK, ale hidden=1024 batch=64 = 67 MB → blízko limitů integrated GPU. Watch.
- **R4: scope creep.** GPU port může fungovat jako "rabbit hole" (mixed precision, fusion, multi-stream, …). Strict scope: forward + backward + Adam, žádné optimalizace nad rámec exit kritéria.

**Realistický odhad rozsahu:** ~1500 LOC, rozprostřeno na 4-5 dní soustředěné práce. Z toho ~30 % je WGSL kernely, ~30 % je Rust GPU wrapper, ~20 % testy a benchmarky, ~20 % Cargo configuration / wgpu-specific debugging.

**Status (foundation hotová):**
- ✅ Stateful TBPTT: `forward_lm_bytes_stateful` čte/píše `GatedState` přes chunk boundaries, gradient detached. `forward_backward_batch_stateful` paralelně přes batch dim s per-stream stavem. Identity test ověřuje bit-paritu se stateless cestou; smoke test ověřuje, že `cache.initial_h_*` skutečně přenáší netriviální stav. Oba `char_lm_gated*` examples používají parallel-stream stateful TBPTT (každý batch slot má vlastní contiguous slice korpu).
- ✅ wgpu foundation: `Cargo.toml` s `gpu` feature flagem (deps `wgpu = "23"`, `bytemuck`, `pollster`). `src/gpu/{mod,context,buffers,kernels}.rs` — `GpuContext` init, typed `GpuBuffer` s upload/download přes staging buffer, první WGSL kernel `sparse_matvec_onehot`.
- ✅ GPU parity test (`tests/gpu_parity.rs`, gated by `--features gpu`): naměřeno **bit-identical** výstup `sparse_matvec_onehot` vs CPU column-read na NVIDIA GTX 1650 (Vulkan backend).

**Sprint 13 followups (next session):**
- WGSL `sparse_matvec_binary` kernel (`out[h] = sum over c where mask[c]==1 of W[h*N+c]`)
- WGSL fused gated cell forward (gate + membrane + spike + reset v jednom passu)
- WGSL fused gated cell backward s surrogate gradient
- WGSL Adam step kernel
- `GatedRecurrentSnnGpu` mirror struktura, `from_cpu`/`to_cpu` weight transfer
- `forward_backward_batch_gpu` end-to-end na GPU
- `examples/char_lm_gated_gpu.rs` + benchmark měření chunks/sec speedup

---

## Sprint 14 — GPU forward end-to-end

**Sprint 14 status (forward port hotový):**
- ✅ WGSL `dense_matvec` kernel s volitelným bias add — generický `out = W @ v (+ bias)`. Použito pro `W_rec`, `W_gate_rec`, `W_out`.
- ✅ WGSL `gated_cell_step` kernel — fused element-wise per timestep: gate sigmoid, membrane integration s adaptivním decay, spike threshold, subtraction reset. Outputy packed do single bufferu `[gate, h_v_pre, h_spike, h_v_after]` aby se vešlo do limitu 8 storage bindings.
- ✅ `GatedRecurrentSnnGpu` mirror struct s GPU-resident weights + cached compute pipelines. `from_cpu(net, ctx)` uploadne všech 8 parameter buffers + zkompiluje 3 kernely.
- ✅ `forward_lm_bytes_gpu` — multi-timestep dispatch loop:
  1. `sparse_matvec_onehot` pro `W_in[:, byte]` a `W_gate_in[:, byte]`
  2. `dense_matvec` pro `W_rec @ h_spike_prev` a `W_gate_rec @ h_spike_prev`
  3. CPU pre-sum gate_pre = `b_gate + W_gate_in_contrib + W_gate_rec_contrib` (3 KB roundtrip per step — fused-add WGSL kernel je followup)
  4. `gated_cell_step` kernel
  5. `copy_buffer_to_buffer` extrahuje h_spike a h_v_after z packed bufferu
  6. `dense_matvec` pro `W_out @ h_spike + b_o` → logits
- ✅ Forward parity test (`gated_forward_lm_matches_cpu_within_tolerance`): GPU output matches CPU within **1e-3** tolerance per logit per timestep, max diff naměřena ~1e-7. Test pouští se hidden=32, T=16 byte sequence na NVIDIA GTX 1650.

**Sprint 14 ZÁMĚRNĚ NEŘEŠÍ:**
- Backward port (Sprint 15+).
- Fused-add WGSL kernel pro `gate_pre` (Sprint 14.1 micro-optim).
- Sparse W_rec / W_out matvec (binární spike skipping na GPU s prefix-sum); aktuální dense matvec je dostatečný a clean.
- GPU end-to-end training. Až po backward portu.
- `char_lm_gated_gpu.rs` example. Až po training port.

**Sprint 15+ followups:**
- WGSL `gated_cell_step_backward` kernel s surrogate gradient (per-timestep gradient inputs/outputs, dependence na cache)
- WGSL `outer_product_accum` kernel pro `grad_w += d_out ⊗ in` (W_out, W_rec, W_gate_rec gradient updates)
- WGSL `column_accum` kernel pro `grad_w[:, byte_idx] += d_h_v_pre` (W_in, W_gate_in sparse gradient)
- WGSL `adam_step` kernel — Adam update přes parameter buffer
- `forward_backward_batch_gpu` end-to-end
- `examples/char_lm_gated_gpu.rs` s `--gpu` flag
- `benches/lm_gpu_speedup.rs` měření chunks/sec speedup

---

## Sprint 14 a dál — možné větve (decision point po 13)

**Vstupy do rozhodnutí:**
- Sprint 13 GPU speedup
- BPB strop vanilla gated SNN
- Realistický next milestone (10 MB corpus? Hybrid? Recurrent topologie změny?)

**Možné větve:**
- **Stateful BPTT** — carry hidden state s detach mezi chunks. Long-range dependencies. ~50 LOC.
- **Větší corpus + delší trénink** — 10-100 MB text, multi-hour GPU trénink. Test scaling.
- **Hybrid surrogate→STDP** — pretrained LM, STDP fine-tune. Biological plausibility experiment.
- **CIFAR-10 přes gated SNN** — image classification benchmark, srovnání s SOTA SNN.
- **f16 mixed precision** — další 2× speedup na GPU (na NVIDIA s tensor cores 4×).
- **Multi-layer gated SNN** — 2-3 hidden vrstvy místo 1, depth scaling experiment.

---

## Co tento plán **záměrně neřeší**

- **LLM trénink.** Cesta k tomu je dlouhá; sprint 5 je první milník, kde má smysl o LLM vůbec mluvit.
- **Distribuovaný trénink, multi-GPU.** Problém pro daleko pozdější fázi.
- **Webové UI, vizualizace.** Pro debug stačí logging + Python notebook na post-processing.
- **Backwards compatibility.** Jsme ve výzkumné fázi, breaking changes jsou OK.
- **Benchmark vs. PyTorch SNN knihovny** (snnTorch, Norse, BindsNET). Až po sprintu 5 — porovnávat s referencí dává smysl, jen když máme co.

---
