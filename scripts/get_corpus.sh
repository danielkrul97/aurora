#!/usr/bin/env bash
# Download text corpora for char-LM training.
# Usage: bash scripts/get_corpus.sh [tiny|shakespeare-complete|wikitext2|enwik8|enwik8-10m|enwik9]
# Default: tiny (1.1 MB tiny-shakespeare from karpathy/char-rnn).
#
# Wikipedia variants (Matt Mahoney compression-benchmark dumps):
#   enwik8     — 100 MB, first 1e8 bytes of enwiki-20060303 dump
#                (canonical char-LM benchmark, used in Hutter Prize)
#   enwik8-10m — first 10 MB of enwik8 (quick experiments)
#   enwik9     — 1 GB, first 1e9 bytes (heavy training)

set -euo pipefail

mkdir -p data
which="${1:-tiny}"

case "$which" in
  tiny)
    out=data/tiny.txt
    url=https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    direct=yes
    ;;
  shakespeare-complete)
    out=data/shakespeare.txt
    url=https://www.gutenberg.org/cache/epub/100/pg100.txt
    direct=yes
    ;;
  wikitext2)
    out=data/wikitext2.txt
    url=https://raw.githubusercontent.com/awni/transformer/master/data/wikitext-2/wiki.train.tokens
    direct=yes
    ;;
  enwik8)
    out=data/enwik8.txt
    url=https://mattmahoney.net/dc/enwik8.zip
    direct=no
    archive=data/enwik8.zip
    member=enwik8
    ;;
  enwik8-10m)
    out=data/enwik8_10m.txt
    url=https://mattmahoney.net/dc/enwik8.zip
    direct=no
    archive=data/enwik8.zip
    member=enwik8
    truncate_bytes=10000000
    ;;
  enwik9)
    out=data/enwik9.txt
    url=https://mattmahoney.net/dc/enwik9.zip
    direct=no
    archive=data/enwik9.zip
    member=enwik9
    ;;
  *)
    echo "unknown corpus: $which" >&2
    echo "  options: tiny, shakespeare-complete, wikitext2, enwik8, enwik8-10m, enwik9" >&2
    exit 1
    ;;
esac

if [ -f "$out" ]; then
  echo "$out already present, skipping"
else
  if [ "$direct" = "yes" ]; then
    echo "downloading $url → $out"
    curl -fL --output "$out" "$url"
  else
    if [ ! -f "$archive" ]; then
      echo "downloading $url → $archive"
      curl -fL --output "$archive" "$url"
    fi
    if ! command -v unzip >/dev/null 2>&1; then
      echo "unzip command not found — install it (apt install unzip / brew install unzip)" >&2
      exit 1
    fi
    echo "extracting $member from $archive"
    unzip -o -p "$archive" "$member" > "$out.full"
    if [ -n "${truncate_bytes:-}" ]; then
      echo "truncating to first $truncate_bytes bytes"
      head -c "$truncate_bytes" "$out.full" > "$out"
      rm "$out.full"
    else
      mv "$out.full" "$out"
    fi
  fi
fi

bytes=$(wc -c < "$out")
echo "$out: $bytes bytes"
