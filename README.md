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
- Per-neuron modulátory pro per-output reward attribution
- Surrogate gradient SNN s hand-coded BPTT (alternativní learning path k STDP)
- Recurrent surrogate SNN s W_rec hidden→hidden + gradient clipping pro Sequential MNIST
- Adam optimizer reusable přes oba surrogate stacks
- Byte-level char-level LM (next-byte prediction, BPB metric, autoregressive generation)
- Gated recurrent SNN (LSTM-style per-neuron forget gate, řeší vanishing gradient)
- Mini-batched parallel BPTT pro gated cell (rayon přes batch dim, externalizovaný grad buffer)
- Stateful TBPTT s detach na chunk hranicích (`forward_lm_bytes_stateful`, parallel streams)
- GPU foundation přes wgpu (feature `gpu`, NVIDIA/Vulkan/Metal/DX12); první kernel + parity test
- GPU forward end-to-end pro gated SNN (`GatedRecurrentSnnGpu`, multi-timestep dispatch, CPU↔GPU parity test)

## Spuštění

```bash
cargo test                                    # všech 14 integračních testů
cargo bench --bench sparsity_scaling          # event vs. discrete throughput
bash scripts/get_mnist.sh                     # stáhne MNIST do data/mnist/
cargo run --release --example mnist_train     # supervised R-STDP training (Sprints 5-7)
cargo run --release --example mnist_eval      # accuracy na test setu
cargo run --release --example mnist_surrogate # surrogate-gradient BPTT (Sprint 8)
cargo run --release --example seq_mnist        # recurrent surrogate, Sequential MNIST (Sprint 9)
cargo run --release --example char_lm -- data/tiny.txt  # byte-level char LM (Sprint 10)
cargo run --release --example char_lm_gated -- data/tiny.txt  # gated cell variant (Sprint 11)
cargo run --release --example char_lm_gated_cz -- data/cesky.txt  # CZ char-LM (UTF-8)
```

## Dokumentace

- [`docs/`](./docs/README.md) — průzkum SNN/LLM (8 tematických dokumentů)
- [`docs/09_aurora_neuron.md`](./docs/09_aurora_neuron.md) — design jednoho neuronu
- [`docs/10_implementacni_plan.md`](./docs/10_implementacni_plan.md) — sprinty + dosažené stavy
