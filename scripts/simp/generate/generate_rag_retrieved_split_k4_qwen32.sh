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
    qwen-32b) MODEL_ALIAS="qwen-32b"; MODEL_SHORT_NAME="RAG-retrieved-split-Qwen2.5-32B" ;;
    gemma-3-27b) MODEL_ALIAS="gemma-3-27b"; MODEL_SHORT_NAME="RAG-retrieved-split-Gemma-3-27B" ;;
    llama-70b) MODEL_ALIAS="llama-70b"; MODEL_SHORT_NAME="RAG-retrieved-split-Llama-3.1-70B" ;;
    *)
        echo "unknown model type: $MODEL_TYPE"
        exit 1
        ;;
esac

RUNS=(1 2 3)
K=4
INDEX_SLUG="medcpt_title_auto"

DATA_DIR="$REPO_ROOT/data/cochrane-sections/cochrane-sections"
DATA_JSON="$REPO_ROOT/data/cochrane-sections/data.json"
BASE_OUTPUT_DIR="$REPO_ROOT/data/outputs/generations"
MODEL_NAME_FULL="${MODEL_SHORT_NAME}-k${K}-cochrane-sections"
OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_NAME_FULL"
mkdir -p "$OUTPUT_DIR"

POOL_FILE="$DATA_DIR/cochrane_sections_sents_train.csv"
TEST_FILE="$DATA_DIR/cochrane_sections_sents_test.csv"

INDEX_DIR="$REPO_ROOT/data/intermediate/rag_index"
mkdir -p "$INDEX_DIR"
INDEX_FILE="$INDEX_DIR/rag_index_cochrane_sections_${INDEX_SLUG}.pkl"

echo ""
echo "MODEL_TYPE: $MODEL_TYPE"
echo "K: $K  POOL: cochrane-sections train RETRIEVAL: retrieved EXAMPLE: split"
echo "INDEX_FILE: $INDEX_FILE"
echo "RUNS: ${RUNS[*]}"
echo ""

if [ ! -f "$INDEX_FILE" ]; then
    echo "index missing, building doc-level index"
    python $REPO_ROOT/src/llm_simp/simplifier_rag/build_index.py \
        --pool_file "$POOL_FILE" \
        --out_file "$INDEX_FILE" \
        --data_json "$DATA_JSON" \
        --article_encoder "ncbi/MedCPT-Article-Encoder" \
        --query_encoder "ncbi/MedCPT-Query-Encoder" \
        --max_length 64 \
        --batch_size 128 \
        --seed 42 \
        --deterministic
else
    echo "index found, reusing"
fi

START_TIME=$(date +%s)

for RUN in "${RUNS[@]}"; do
    PRED_FILE="$OUTPUT_DIR/${MODEL_NAME_FULL}_run${RUN}.csv"
    echo ""
    SEED=$((41 + RUN))
    echo "run $RUN  seed=$SEED -> $PRED_FILE"

    python $REPO_ROOT/src/llm_simp/simplifier_rag/generate_rag_pls.py \
        --test_file "$TEST_FILE" \
        --out_file "$PRED_FILE" \
        --index_file "$INDEX_FILE" \
        --data_json "$DATA_JSON" \
        --k $K \
        --tensor_parallel_size $TP_SIZE \
        --model_name $MODEL_ALIAS \
        $EXTRA_ARGS \
        --seed $SEED \
        --deterministic
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "finished in $((DURATION / 3600))h $(((DURATION % 3600) / 60))m $((DURATION % 60))s"

