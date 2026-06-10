# Projekt

Aurora je decoder-only transformer language model, from-scratch v Rustu.
Pre-norm GPT-style stack (token + positional embed, N transformer bloků
s causal multi-head attention a GELU FFN, tied LM head), AdamW + cosine
LR schedule, CPU + GPU backend (wgpu / WGSL).

Projekt byl v Sprint 0–50 biologicky inspirovaná SNN (LIF + surrogate
gradient + gated recurrent cell). Sprint 51 demolice + Sprint 52–60
rebuild ho překlopily na klasický transformer; SNN linie zůstává jen
v archivních docs/01_*–docs/11_*. Cíl je realistické BPB < 1.5 na enwik8
v overnight runu (nano-GPT scale ~10M params, GTX 1650).

## Hlavní vstupní body

- **Hlavní training example:** `examples/aurora_train.rs`. CPU default,
  `--gpu` opt-in (vyžaduje `--features gpu`), `--viz` opt-in (vyžaduje
  `--features viz` — Bevy 0.18 okno s real-time pipeline vizualizací,
  implementace v `src/viz/`). CLI flags: `--d-model --n-layer --n-head
  --d-ff --seq-len --batch-size --epochs --lr --warmup-steps
  --weight-decay --grad-clip --checkpoint --bpe --metrics --seed
  --sample-len --sample-temp --gpu --viz --rope --muon-lr
  --muon-momentum --muon-wd`. Spouštěj přes
  `scripts/overnight.sh` nebo přímo `cargo run --release
  [--features gpu,viz] --example aurora_train -- ...`.
- **Anti-regrese před commitem:**
  - `tests/transformer_primitives.rs` — numerické gradient checks (CPU)
  - `tests/transformer_smoke.rs` — copy task convergence (CPU)
  - `tests/lm_quality_regression.rs` — BPB threshold + checkpoint round-trip
  - `tests/stability.rs` — grad health, sampling diversity, LR schedule
  - `tests/gpu_parity.rs` — CPU↔GPU drift < 5e-2, 13 testů (`--features gpu`)
  - `tests/kv_cache.rs` — inference parity vs full-forward
- **Diagnostické grafy:** `python3 analysis/diagnostics.py logs/<run>.jsonl`
  po skončení tréninku.

## Konvence

- **Aktivní plán = `docs/12_refactoring_plan_transformer.md`.**
  `docs/10_*` (Sprint 0–25) a `docs/11_*` (Sprint 26–50) jsou zamrzlé
  historické záznamy SNN éry. Nový sprint dokumentuj v 12; pokud 12
  přeroste ~50 KB, založ 13.
- **JSONL metriky** chodí default do `logs/<ckpt_stem>_<unix_ts>.jsonl`.
  Schema: `run` / `step` / `epoch_done` / `checkpoint` events. Per-event
  flush — `tail -f` má sub-second lag.
- **Checkpoint format `AURORA04`** (`src/checkpoint.rs`) — přidal `flags`
  u32 (bit 0 = rope). Žádný backward compat se staršími formáty
  (AURORA01–03 jsou pryč). Bumpy magic na AURORA05 při schema change.
- **Korpusy** se stahují přes `scripts/get_corpus.sh <variant>` do
  `data/`. `data/` a `logs/` jsou v gitignore.
- **GPU performance stav (po Sprint 64):** GEMM i MHA kernely jsou tiled,
  trénink běží 1 submit/batch, norm readback jen na report stepy, eval
  batchovaný přes `eval_step_batch` (kapacita = `max_batch`). Per-step je
  compute-bound (GTX 1650) — další zrychlení = menší práce (f16, flash
  attention), ne dispatch. CPU eval je rayon-paralelní; CPU linear/tied-head
  smyčky jsou pozičně blokované (bit-exact, neměnit pořadí akumulace).
- **Live viz** (`--features viz --viz`) je čistě additivní — kompiluje
  Bevy 0.18 jako optional dep, default build/CI nezasahuje. Viz thread
  čte `Arc<Mutex<TrainingSnapshot>>` přes `try_lock`, trénink píše
  jednou per micro-step. Bary v UI jsou per-step relativní normalizace
  grad norem (max layer = 1.0, floor 0.18) — víc viditelné než absolutní
  scaling. Mesh2d + `bevy::sprite_render::ColorMaterial` všude pro
  kreslení (Sprite::from_color v 0.18 renderuje nic).

# Code style

- Screenshoty VŽDY ukládej do složky screenshots/

## Commit messages

- Conventional Commits (`fix:`, `feat:`, `chore:`, …)
- Subject ≤ 50 znaků, bez tečky na konci
- Body jen když "proč" není zřejmé z diffů — max 2–3 řádky
- Žádné bullet-listy změněných souborů ani vyčerpávající popisy — kdo chce detail, přečte si diff

## Komentáře

- **Default: žádný komentář.** Dobře pojmenovaná proměnná, funkce a typ popisují *co* kód dělá — komentář to jen duplikuje a zastarává.
- Komentář piš **jen když vysvětluje WHY** — skrytý constraint, netriviální invariant, workaround pro konkrétní bug, překvapivé chování pro čtenáře.
- Pokud by odstranění komentáře nikoho nezmátlo, nepiš ho.
- **Stručně**: typicky 1–3 řádky. Odstavce v komentářích jsou varovný signál (kód by měl být čitelnější, ne okomentovanější).
- **Nepiš komentáře u self-explanatory kódu** — getter, mapovací funkce, zřejmá validace, pojmenovaný boolean výraz, typická CRUD operace. I krátký komentář tady jen přidává šum.
- **Nereferencuj kontext, který komentář přežije**: žádné „used by X", „added for the Y flow", „handles case from issue #123". To patří do PR description / commit message, ne do souboru.
- Komentáře piš anglicky.
