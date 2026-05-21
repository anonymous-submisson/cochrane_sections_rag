#!/bin/bash

set -euo pipefail

DEST="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )/data/cochrane-sections"
TMP="$(mktemp -d)"

git clone --depth=1 https://github.com/JanB100/cochrane-sections.git "$TMP"

mkdir -p "$DEST"
cp "$TMP/data/cochrane/data.json" "$DEST/"
cp -r "$TMP/alignments" "$DEST/"

rm -rf "$TMP"

