#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"

DATA_PATH="$REPO_ROOT/data/cochrane-sections/data.json"
ALIGNMENTS_DIR="$REPO_ROOT/data/cochrane-sections/alignments"
PREPROCESS="$REPO_ROOT/src/data_preprocess/cochrane-sections/preprocess_cochrane_sections.py"

echo ""
echo "DATA_PATH: $DATA_PATH"
echo "ALIGNMENTS_DIR: $ALIGNMENTS_DIR"
echo ""

echo "building cochrane-sections (auto and manual)"
python $PREPROCESS \
    --data_path $DATA_PATH \
    --alignments_dir $ALIGNMENTS_DIR \
    --output_dir $REPO_ROOT/data/cochrane-sections/cochrane-sections

echo ""
echo "building cochrane-sections-manual (manual)"
python $PREPROCESS \
    --data_path $DATA_PATH \
    --alignments_dir $ALIGNMENTS_DIR \
    --output_dir $REPO_ROOT/data/cochrane-sections/cochrane-sections-manual \
    --manual_only
