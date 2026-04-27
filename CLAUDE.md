# Projekt

Cílem projektu Aurora je vytvořit custom LLM, přičemž využije neurobiologicky inspirovanou architekturu 
umělé neuronové sítě. Projekt je napsán v Rustu. Cílem však NENÍ vytvořit další obecnou LLM.

## Architektura

| Vrstva       | Technologie |
| ------------ |-------------|
| Tests        |             |

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