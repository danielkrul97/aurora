# Projekt

Cílem projektu Aurora je vytvořit custom LLM, přičemž využije neurobiologicky inspirovanou architekturu
umělé neuronové sítě. Projekt je napsán v Rustu. Cílem však NENÍ vytvořit další obecnou LLM.

**Architektura:** Closed-form continuous-time (CfC) Liquid Neural Network cell
(Hasani et al., Nature MI 2022, arXiv:2106.13898). Inspirovaná C. elegans nervovou soustavou,
smooth aktivace (sigmoid, tanh, exp) — žádný surrogate gradient, stav `x ∈ [-1, 1]^H` z konstrukce.

## Hlavní vstupní body

- **Hlavní training example:** `examples/aurora_train.rs` (GPU default + BPE + checkpointing + auto-resume; `--cpu` fallback).
  Spouštěj přes `scripts/overnight.sh` nebo přímo `cargo run --release --features gpu --example aurora_train -- ...`.
- **Anti-regrese:** `tests/lnn_smoke.rs` (CPU forward + finite-diff grad checks),
  `tests/lnn_gpu_parity.rs` (CPU↔GPU forward+backward+Adam drift < 1e-3). Před commitem oba zkontroluj.
- **Diagnostické grafy:** `python3 analysis/diagnostics.py logs/<run>.jsonl` po skončení tréninku.

## Konvence

- **Aktivní plán = `docs/12_implementacni_plan_v3.md`.** `docs/11_implementacni_plan_v2.md` je zamrzlý
  historický záznam (Sprint 26–50, SNN+SSM era), do něj nepiš. `10_implementacni_plan.md` je
  hluboká historie (Sprint 0–25). Nový sprint dokumentuj v 12; pokud 12 přeroste ~50 KB, založ 13.
- **JSONL metriky** chodí default do `logs/<corpus_stem>_<unix_ts>.jsonl`. Schema: `run` / `chunk` /
  `epoch_done` / `checkpoint` events. Per-event flush — `tail -f` má sub-second lag.
  Per-event fields pro CfC: `activity` (mean |x|), `alpha_means`/`alpha_devs` (gating), `tau_means` (timescale).
- **Checkpoint format `AURORA-LNN01`** (`src/checkpoint.rs`). Bumpy magic při schema change.
  Žádný backward compat na předchozí SNN formáty (`AURORA01/02`) — jiná architektura, jiný layout.
- **Korpusy** se stahují přes `scripts/get_corpus.sh <variant>` do `data/`. `data/` a `logs/` jsou v gitignore.

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
