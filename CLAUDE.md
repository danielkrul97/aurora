# Projekt

Cílem projektu Aurora je vytvořit custom LLM, přičemž využije neurobiologicky inspirovanou architekturu 
umělé neuronové sítě. Projekt je napsán v Rustu. Cílem však NENÍ vytvořit další obecnou LLM.

## Hlavní vstupní body

- **Hlavní training example:** `examples/aurora_train.rs` (GPU 2-layer + BPE + checkpointing + auto-resume).
  Spouštěj přes `scripts/overnight.sh` nebo přímo `cargo run --release --features gpu --example aurora_train -- ...`.
- **Anti-regrese:** `tests/lm_quality_regression.rs` (CPU byte + BPE BPB threshold + checkpoint round-trip),
  `tests/gpu_parity.rs` (CPU↔GPU drift < 5e-2). Před commitem oba zkontroluj.
- **Diagnostické grafy:** `python3 analysis/diagnostics.py logs/<run>.jsonl` po skončení tréninku.

## Konvence

- **Aktivní plán = `docs/11_implementacni_plan_v2.md`.** `docs/10_implementacni_plan.md` je zamrzlý
  historický záznam (Sprint 0–25, 92 KB), do něj nepiš. Nový sprint dokumentuj v 11; pokud 11 přeroste
  ~50 KB, založ 12.
- **JSONL metriky** chodí default do `logs/<ckpt_stem>_<unix_ts>.jsonl`. Schema: `run` / `chunk` /
  `epoch_done` / `checkpoint` events. Per-event flush — `tail -f` má sub-second lag.
- **Checkpoint format `AURORA02`** (`src/checkpoint.rs`). Bumpy magic při schema change, drž backward compat
  pro `AURORA01`.
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
