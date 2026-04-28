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
- GPU backward + Adam end-to-end pro gated SNN (BPTT na GPU, parity vs CPU < 5e-3 přes všech 8 grad bufferů, `forward_backward_lm_bytes_gpu`, `step_adam_gpu`, `clip_grads_gpu`, `to_cpu`)
- Fused `gate_pre` + workgroup-shared `softmax_cross_entropy` WGSL kernely — eliminace per-timestep CPU↔GPU roundtripů (4.5× rychlejší GPU forward+backward vs Sprint 15)
- Encoder batching + push constants (1 `queue.submit` per phase místo ~6400, push constants místo per-dispatch uniform bufferu) — 6.8× cumulative speedup od Sprint 14
- Forward megakernel (single dispatch covers entire chunk, workgroup-shared state mezi timesteps) + packed cache buffery + cached bind groups — 7.3× cumulative speedup od Sprint 14 při hidden ≤ 256
- Multi-sample batch fusion ve forward + softmax_ce (forward dispatchuje B workgroups, softmax 2D grid B×T) — `forward_backward_batch_gpu_fused`, **11.6× cumulative speedup** od Sprint 14 při batch=4
- Backward batch fusion (5 backward kernelů rozšířených o batch dim, fused `gpu_backward_batched` pokrývá všechny B samples per timestep paralelně) — odemyká vyšší batch sizes; B=16 50KB workload **3.4× rychlejší** než B=4
- Sparse spike compaction ve forward megakernelu (workgroup-shared atomic counter buildne `active_indices`, matvec inner loops skipují non-firing neurony) — replikuje CPU `active_indices` trick na GPU; +22 % bench při hidden=512
- Sparse skip v backward `outer_product_accum` (zero-check na `h_spike` operand uvnitř batch loopu) — univerzální +21-26 % gain napříč všemi hidden sizes
- Multi-layer gated SNN (CPU) — `MultiLayerGatedSnn` s `Vec<GatedLayer>`, cross-layer BPTT, konfigurovatelný `hidden_dims: &[usize]` per vrstva. GPU stays single-layer (Sprint 24+ port).
- Multi-layer streaming inference (`step_lm_byte`) + rayon-paralelní stateful batch trainer + `examples/char_lm_gated_multi` s `--n-layers` a `--hidden-per-layer` CLI argumenty
- GPU 2-layer port (`MultiLayerGatedSnnGpu`, single-shot forward megakernel chains 2 vrstvy in-shader, backward dispatches per-layer top-down) — parita vs CPU multi-layer < 5e-2 přes 14 grad bufferů

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
cargo run --release --features gpu --example char_lm_gated_gpu -- data/tiny.txt  # plný GPU trénink
cargo run --release --example char_lm_gated_multi -- data/tiny.txt 3 64 0.002 2 256  # multi-layer CPU (n_layers, hidden)
cargo run --release --features gpu --example char_lm_gated_multi_gpu -- data/tiny.txt 3 64 0.002 256  # 2-layer GPU
cargo bench  --features gpu --bench lm_gpu_speedup        # CPU vs GPU throughput
```

## Dokumentace

- [`docs/`](./docs/README.md) — průzkum SNN/LLM (8 tematických dokumentů)
- [`docs/09_aurora_neuron.md`](./docs/09_aurora_neuron.md) — design jednoho neuronu
- [`docs/10_implementacni_plan.md`](./docs/10_implementacni_plan.md) — sprinty + dosažené stavy
