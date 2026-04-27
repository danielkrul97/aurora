# Aurora

Custom LLM postavený na neurobiologicky inspirované architektuře spiking
neural network. Implementace v Rustu, jen lokální plasticita (STDP +
neuromodulace), žádný backprop.

> Status: rané sprinty hotové — primitivy SNN postavené a testované.
> Plný MNIST/LLM benchmark zatím nedosahuje cílů (viz `docs/10_implementacni_plan.md`).

## Co je hotové

- LIF neuron s dendritickými kompartmenty + homeostatická adaptace prahu
- STDP (pair-based) a three-factor R-STDP s eligibility trace + neuromodulátorem
- Discrete-time `Network` i event-driven `EventNetwork` (~24× rychlejší při řídké aktivitě)
- MNIST pipeline (encoding, topology builder, readout, IDX loader, train/eval examples)
- WTA primitivy: weight clamp, all-to-all lateral inhibition helper, plastic flag per synapse
- Lazy eligibility commit + rayon paralelizace pro discrete `Network::step`

## Spuštění

```bash
cargo test                                    # všech 14 integračních testů
cargo bench --bench sparsity_scaling          # event vs. discrete throughput
bash scripts/get_mnist.sh                     # stáhne MNIST do data/mnist/
cargo run --release --example mnist_train     # supervised R-STDP training
cargo run --release --example mnist_eval      # accuracy na test setu
```

## Dokumentace

- [`docs/`](./docs/README.md) — průzkum SNN/LLM (8 tematických dokumentů)
- [`docs/09_aurora_neuron.md`](./docs/09_aurora_neuron.md) — design jednoho neuronu
- [`docs/10_implementacni_plan.md`](./docs/10_implementacni_plan.md) — sprinty + dosažené stavy
