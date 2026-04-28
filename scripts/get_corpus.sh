#!/usr/bin/env bash
# Download text corpora for char-LM training.
# Usage: bash scripts/get_corpus.sh [tiny|shakespeare-complete|wikitext2]
# Default: tiny (1.1 MB tiny-shakespeare from karpathy/char-rnn).

set -euo pipefail

mkdir -p data
which="${1:-tiny}"

case "$which" in
  tiny)
    out=data/tiny.txt
    url=https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    ;;
  shakespeare-complete)
    out=data/shakespeare.txt
    url=https://www.gutenberg.org/cache/epub/100/pg100.txt
    ;;
  wikitext2)
    out=data/wikitext2.txt
    url=https://raw.githubusercontent.com/awni/transformer/master/data/wikitext-2/wiki.train.tokens
    ;;
  *)
    echo "unknown corpus: $which (try: tiny, shakespeare-complete, wikitext2)" >&2
    exit 1
    ;;
esac

if [ -f "$out" ]; then
  echo "$out already present, skipping"
else
  echo "downloading $url → $out"
  curl -fL --output "$out" "$url"
fi

bytes=$(wc -c < "$out")
echo "$out: $bytes bytes"
