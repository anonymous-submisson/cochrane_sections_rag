#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MODEL_TYPE="gemma-3-27b"

bash "$SCRIPT_DIR/doc_eval_rag_retrieved_full_k4_qwen32.sh"
