#!/usr/bin/env bash
# Build a Czech char-LM corpus from public-domain Project Gutenberg books.
# Output: data/cesky.txt — UTF-8 text with Gutenberg license boilerplate
# stripped, books separated by blank lines.
#
# IDs were taken from https://www.gutenberg.org/browse/languages/cs on
# 2026-04-28. Includes Karel Čapek's R.U.R., five novels by František
# Omelka, two Dostoyevsky translations into Czech, plus a couple short
# pieces. Together ~3-5 MB of native Czech text.

set -euo pipefail

mkdir -p data data/cz_raw
out=data/cesky.txt

# Gutenberg numeric IDs for Czech ebooks.
ids=(
  13083  # Čapek, Karel — R.U.R.
  47754  # Omelka, František — Blesky nad Beskydami
  27960  # Omelka, František — Hore dědinú
  29592  # Omelka, František — Pasáček Ali
  29648  # Omelka, František — Štafeta
  27974  # Omelka, František — Vlci proti Mustangům
  37525  # Dostoyevsky — Dvojník (Czech translation)
  34225  # Dostoyevsky — Zápisky z mrtvého domu
  68828  # Batěk — Sto jisker ethických
  59765  # Lepar et al. — Čítanka pro školy obecné
)

# Download each book's UTF-8 plain text. Try several URL patterns —
# Gutenberg's layout varies by upload date.
download() {
  local id="$1"
  local dst="data/cz_raw/${id}.txt"
  if [ -s "$dst" ]; then
    echo "  cached: ${id}"
    return 0
  fi
  local urls=(
    "https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
    "https://www.gutenberg.org/files/${id}/${id}-0.txt"
    "https://www.gutenberg.org/files/${id}/${id}.txt"
  )
  for url in "${urls[@]}"; do
    if curl -sfL --output "$dst" "$url" && [ -s "$dst" ]; then
      echo "  ok: ${id} ($(wc -c < "$dst") bytes)"
      return 0
    fi
  done
  echo "  FAILED: ${id}" >&2
  rm -f "$dst"
  return 1
}

# Strip Gutenberg boilerplate (everything before "*** START OF" and after
# "*** END OF"). Outputs the body text only.
strip_boilerplate() {
  awk '
    /^\*\*\* START OF/ { in_body = 1; next }
    /^\*\*\* END OF/   { exit }
    in_body == 1       { print }
  '
}

echo "downloading Czech books from Project Gutenberg…"
for id in "${ids[@]}"; do
  download "$id" || true
done

echo "stripping boilerplate and concatenating → $out"
: > "$out"
for id in "${ids[@]}"; do
  src="data/cz_raw/${id}.txt"
  if [ ! -s "$src" ]; then
    continue
  fi
  strip_boilerplate < "$src" >> "$out"
  printf '\n\n' >> "$out"
done

bytes=$(wc -c < "$out")
echo "done: $out = $bytes bytes"
