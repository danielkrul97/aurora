#!/usr/bin/env bash
# Download MNIST IDX files into data/mnist/ for the mnist_train / mnist_eval
# examples. Files are ~50 MB total.

set -euo pipefail

DEST="data/mnist"
mkdir -p "$DEST"

URLS=(
  "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
  "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
)

for url in "${URLS[@]}"; do
  fname=$(basename "$url")
  unpacked="${fname%.gz}"
  if [ -f "$DEST/$unpacked" ]; then
    echo "$unpacked already present, skipping"
    continue
  fi
  echo "downloading $fname"
  curl -fL --output "$DEST/$fname" "$url"
  gunzip "$DEST/$fname"
done

echo "MNIST ready in $DEST/"
