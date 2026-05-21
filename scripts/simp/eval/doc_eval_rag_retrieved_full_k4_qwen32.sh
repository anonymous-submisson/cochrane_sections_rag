#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"

python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

MODEL_TYPE="${MODEL_TYPE:-qwen-32b}"

case $MODEL_TYPE in
    qwen-32b) MODEL_SHORT_NAME="RAG-retrieved-full-Qwen2.5-32B" ;;
    gemma-3-27b) MODEL_SHORT_NAME="RAG-retrieved-full-Gemma-3-27B" ;;
    llama-70b) MODEL_SHORT_NAME="RAG-retrieved-full-Llama-3.1-70B" ;;
    *) echo "Unknown model type: $MODEL_TYPE"; exit 1 ;;
esac

RUNS=(1 2 3)
K=4

BASE_OUTPUT_DIR="$REPO_ROOT/data/outputs/generations"
DOCS_CSV="$REPO_ROOT/data/cochrane-sections/cochrane-sections/cochrane_sections_docs_test.csv"
MODEL_NAME="${MODEL_SHORT_NAME}-k${K}-cochrane-sections"
OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_NAME"

echo ""
echo "MODEL_NAME: $MODEL_NAME"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "RUNS: ${RUNS[*]}"
echo ""

START_TIME=$(date +%s)

for RUN in "${RUNS[@]}"; do
    PRED_FILE="$OUTPUT_DIR/${MODEL_NAME}_run${RUN}.csv"
    DOC_FILE="$OUTPUT_DIR/${MODEL_NAME}_run${RUN}_doc.csv"
    METRICS_FILE="$OUTPUT_DIR/${MODEL_NAME}_run${RUN}_metrics.csv"
    PER_SECTION_FILE="$OUTPUT_DIR/${MODEL_NAME}_run${RUN}_per_section_sari.json"

    echo ""
    echo "run $RUN"
    echo "PRED_FILE: $PRED_FILE"

    echo ""
    echo "aggregating sentences into documents"
    python $REPO_ROOT/src/eval/utils/aggregate.py \
        --input_csv $PRED_FILE \
        --output_csv $DOC_FILE \
        --doc_id_col pair_id

    echo ""
    echo "merging simple_full reference from docs csv"
    python $REPO_ROOT/src/eval/utils/merge_simple_ref.py \
        --doc_file $DOC_FILE \
        --docs_csv $DOCS_CSV

    echo ""
    echo "evaluating"
    python $REPO_ROOT/src/eval/eval_simp.py \
        --input_data $DOC_FILE \
        --x_col complex \
        --y_col pred \
        --r_col simple_full \
        --metrics '["smart", "fkgl", "sari"]' \
        --prepro True \
        --out_file $METRICS_FILE \
        --skip_baseline True \
        --seed 42 \
        --deterministic

    echo ""
    echo "computing per-section SARI"
    python $REPO_ROOT/src/eval/per_section_sari.py \
        --input_csvs $PRED_FILE \
        --labels $MODEL_NAME \
        --output_json $PER_SECTION_FILE
done

echo ""
echo "aggregating across runs"

RUNS_CSV=$(IFS=,; echo "${RUNS[*]}")
METRICS_INPUTS=""
PER_SECTION_INPUTS=""
for RUN in "${RUNS[@]}"; do
    [ -n "$METRICS_INPUTS" ] && METRICS_INPUTS="${METRICS_INPUTS},"
    METRICS_INPUTS="${METRICS_INPUTS}${OUTPUT_DIR}/${MODEL_NAME}_run${RUN}_metrics_summary.json"
    [ -n "$PER_SECTION_INPUTS" ] && PER_SECTION_INPUTS="${PER_SECTION_INPUTS},"
    PER_SECTION_INPUTS="${PER_SECTION_INPUTS}${OUTPUT_DIR}/${MODEL_NAME}_run${RUN}_per_section_sari.json"
done

python $REPO_ROOT/src/eval/utils/aggregate_run_stats.py \
    --inputs "$METRICS_INPUTS" \
    --runs "$RUNS_CSV" \
    --output_json "$OUTPUT_DIR/${MODEL_NAME}_metrics_stats.json"

python $REPO_ROOT/src/eval/utils/aggregate_run_stats.py \
    --inputs "$PER_SECTION_INPUTS" \
    --runs "$RUNS_CSV" \
    --output_json "$OUTPUT_DIR/${MODEL_NAME}_per_section_sari_stats.json"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "finished in $((DURATION / 3600))h $(((DURATION % 3600) / 60))m $((DURATION % 60))s"

