# Implementační plán Aurora — v3 (LNN-only era)

> Pokračování `11_implementacni_plan_v2.md`. Tento soubor zachycuje
> Sprint 51+. V2 zamrzlý — nové sprinty piš sem.

---

## Stav po Sprintu 51 (LNN-only refactor)

**Zúžení projektu na CfC LNN.** Sprint 50 byl 2. statisticky významný win
nad SSM baseline (1. byl Sprint 46 diagonal SSM), s podstatně lepší
mechanistickou interpretací (smooth aktivace, bounded state, per-neuron
input-dependent decay). Po S50 jsem rozhodl o úplném refactoru:

- **Smazáno:** všechny SNN moduly (`snn_*`), SSM, attention, SDSA,
  apoptosis, biologické primitives (LIF, dendrite, neuron, synapse,
  network, engine, scheduler, eligibility, neuromodulator, homeostasis,
  stdp, topology, readout, encoding, mnist), surrogate, types, linalg.
  Plus všechny GPU SNN kernely (`forward_mega*.rs`, `kernels.rs`,
  `snn_gated_gpu*.rs`).
- **Přejmenováno:** `snn_loss.rs` → `loss.rs`, `aurora_lnn_train.rs` →
  `aurora_train.rs`. Verze crate `0.1.0` → `0.2.0`.
- **Smazáno:** 24 testů (zůstaly jen `lnn_smoke`), 4 benche, 6 examples
  (zůstal `aurora_train` + `train_bpe`), `09_aurora_neuron.md`,
  `scripts/get_mnist.sh` + `get_cesky.sh`.

**Nové GPU kernely** (`src/gpu/kernels_lnn.rs`, ~700 řádků WGSL):
- `lnn_forward_layer_zero` (sparse one-hot input), `lnn_forward_layer_deep`
  (dense input z předchozí vrstvy) — workgroup per batch element,
  256 threadů přes neurony, T-loop uvnitř kernelu, workgroup-shared
  `x_prev` + `x_new`.
- `lnn_output_proj` (dense GEMV pro logits), `softmax_cross_entropy`
  (workgroup-shared softmax).
- `lnn_topproj_backward` (W_out grad + b_o grad + d_x_top), `lnn_backward_layer_zero/deep`
  (7 grad termů per neuron per timestep, reverse-time T-loop, per-batch
  weight grad accumulation s d_x_carry workgroup-shared).
- `reduce_per_batch` (sčítá B-osu grad bufferů), `adam_step`, `clip_scale`,
  `zero_buffer`, `clamp_log_tau`, `l2_norm_squared`.

**API parity.** `MultiLayerLnnSnnGpu` (`src/gpu/lnn_gpu.rs`) zrcadlí CPU
`MultiLayerLnnSnn`: `from_cpu`, `to_cpu`, `forward_lm_tokens_stateful_batch`,
`forward_backward_batch_tokens_stateful`, `zero_grad`, `clip_grads`,
`step_adam`. Adam state, grad buffery, aktivační cache (5× `[B, T, H]`
per layer), per-batch grady — vše rezident na GPU.

**Nové testy** (`tests/lnn_gpu_parity.rs`):
| test | tolerance | co měří |
|---|---|---|
| `forward_parity_single_layer` | 1e-3 | logits CPU↔GPU pro 1L |
| `forward_parity_two_layer` | 1e-3 | logits + final state pro 2L |
| `backward_parity_single_step` | 1e-3 | weights po 1 Adam stepu |
| `multi_step_drift_bounded` | 5e-3 | weights po 10 Adam stepech |

**Smoke run** (tiny_200k, h=64, 2L, vocab=512, 1 epocha):
- val BPB **3.026** (start 4.3, end 3.47 train).
- elapsed 2.4 s na GTX 1650 (Vulkan).
- Auto-resume z checkpointu projde, val BPB klesá s každou další epochou.

**Co tím odpovídá na původní otázku „má smysl zúžit Auroru na LNN":**
Ano. Cell má jasnější gradient flow než SNN (no surrogate, no LIF
reset), bounded state z konstrukce, per-neuron input-dependent decay
mu dává schopnost selektivně paměťovat (Mamba-like), a originál C.
elegans paper je biologicky validní motivace pro CLAUDE.md cíl.

**Soubory:**
- `src/lnn.rs` (33 KB, beze změny — Sprint 50 cell zachován)
- `src/loss.rs` (rename), `src/checkpoint.rs` (kompletní přepis,
  `AURORA-LNN01` magic)
- `src/gpu/{lnn_gpu.rs, kernels_lnn.rs}` (nové, ~1500 řádků)
- `examples/aurora_train.rs` (rebuild s GPU+CPU backendem)
- `tests/lnn_gpu_parity.rs` (nové)
- `analysis/diagnostics.py` (update fields: rates → activity, alphas → alpha_means+devs+tau_means)
- `Cargo.toml` (verze 0.2.0, smazány staré example/bench bloky)

---

## Otevřené směry

### S52+ — Validace na enwik8_10m

Smoke run je jen na 200k bajtech. Klíčové ověření:
- enwik8_10m, h=512, 2L, vocab=4096, 25 epoch (overnight target).
- Cíl: **val BPB ≤ 2.36** (Sprint 48 SSM baseline). Lepší = LNN je
  konkurenční nad SSM ve skutečném scale.
- Sledovat: τ evolution per layer (jestli učí různé timescales),
  α saturace (mělo by být blízko 0.5 v ranných epochách).

### S53+ — Hyperparam sweep

CfC má méně knobů než SNN (žádný surrogate width, žádný spike rate
target, žádný membrane reset). Co zbývá:
- `--tau-init` (log_tau init): default 0.0 → τ=1.0. Sweepnout
  {-1.0, 0.0, +1.0} → τ ∈ {0.37, 1.0, 2.7}.
- `--lr`: 0.001 / 0.005 / 0.01.
- `--hidden`: 128 / 256 / 512 / 1024 (poslední vyžaduje 2GB VRAM
  pro per-batch grady).
- `--layers`: 2 / 3 / 4.

### S54+ — Architektonické rozšíření (po validaci)

Pouze pokud S52 pass. Možnosti:
- **Multi-timescale LNN:** per-layer log_tau init scaling (vrstva 0 fast,
  vrstva n slow). Inspirace od Hierarchical RNNs (Chung 2016).
- **Sparse LNN:** některé `W_*[i,j]` nuly přes magnitude pruning po
  warm-up. Throughput speedup pokud sparse compaction.
- **GPU optimalizace:** fused 2-layer megakernel (jako bývalo pro SNN),
  parallel reduction místo sequential w_f_grad accum.
- **Inference path:** GPU `step_lm_token` kernel (currently downloads
  weights to CPU per epoch for sample generation).

### S55+ — Otázky k zvážení

1. **Bigger corpus:** Sprint 49 ukázal že u SSM continuous mode
   overfittuje na malém korpusu. LNN je continuous-state také —
   chovat se bude podobně? Test na enwik8_10m (16× větší) je důležitý.
2. **Continuous SSM A/B znovu na LNN bázi:** Nahradit `tanh` v `g`/`h`
   za `linear` → rejmenován input-dependent decay → ekvivalence s
   selective SSM (Mamba)?

---

## Co tento plán **záměrně neřeší**

- **LLM trénink scale** (GPT-X velikosti). Aurora není generický LLM;
  cíl je biologicky-flavored architektura competitive na enwik8.
- **Distribuovaný trénink, multi-GPU.** Single GTX 1650 je hardware
  constraint.
- **Tensor-core specifické optimalizace** (FP16 compute s tensor cores)
  — TU116 je nemá.
- **Návrat k SNN cestě.** Sprint 49 a 50 dohromady ukázaly že hybrid
  SSM+spike je lokální optimum a CfC bez spike vyhrává empiricky i
  mechanisticky. Žádný spike-based directioning.

---

## Open questions kde literatura mlčí (zděděno z v2)

- **GTX 1650 (no tensor cores) jako serious LNN training target** —
  všechny SOTA benchmarky jsou A100/H100. Aurora's GTX 1650 výsledky
  na enwik8 by byly originální data.
- **CfC bez parallel scan na MAX_HIDDEN ≤ 1024 workgroup memory budget** —
  trade-off `n_hidden` vs. throughput nestudovaný.
- **BPE × LNN-CfC LM** — málo prior work; SpikeGPT používá byte-level.
