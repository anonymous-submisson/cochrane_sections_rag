#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"

MODEL_TYPE="${MODEL_TYPE:-qwen-32b}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
EXTRA_ARGS=""
if [ -n "$MAX_MODEL_LEN" ]; then
    EXTRA_ARGS="--max_model_len $MAX_MODEL_LEN"
fi

case $MODEL_TYPE in
    qwen-32b) MODEL_ALIAS="qwen-32b"; MODEL_SHORT_NAME="LLM-Qwen2.5-32B" ;;
    gemma-3-27b) MODEL_ALIAS="gemma-3-27b"; MODEL_SHORT_NAME="LLM-Gemma-3-27B" ;;
    llama-70b) MODEL_ALIAS="llama-70b"; MODEL_SHORT_NAME="LLM-Llama-3.1-70B" ;;
    *)
        echo "Unknown model type: $MODEL_TYPE"
        exit 1
        ;;
esac

RUNS=(1 2 3)

DATA_DIR="$REPO_ROOT/data/cochrane-sections/cochrane-sections"
BASE_OUTPUT_DIR="$REPO_ROOT/data/outputs/generations"
MODEL_NAME_FULL="${MODEL_SHORT_NAME}-cochrane-sections"
OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_NAME_FULL"
mkdir -p $OUTPUT_DIR

TEST_FILE="$DATA_DIR/cochrane_sections_sents_test.csv"

echo ""
echo "MODEL_TYPE: $MODEL_TYPE"
echo "TEST_FILE: $TEST_FILE"
echo "RUNS: ${RUNS[*]}"
echo ""

START_TIME=$(date +%s)

for RUN in "${RUNS[@]}"; do
    PRED_FILE="$OUTPUT_DIR/${MODEL_NAME_FULL}_run${RUN}.csv"
    echo ""
    SEED=$((41 + RUN))
    echo "run $RUN  seed=$SEED -> $PRED_FILE"

    python $REPO_ROOT/src/llm_simp/simplifier_sections/generate.py \
        --test_file "$TEST_FILE" \
        --out_file "$PRED_FILE" \
        --model_name $MODEL_ALIAS \
        --tensor_parallel_size $TP_SIZE \
        $EXTRA_ARGS \
        --seed $SEED \
        --deterministic
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "finished in $((DURATION / 3600))h $(((DURATION % 3600) / 60))m $((DURATION % 60))s"

