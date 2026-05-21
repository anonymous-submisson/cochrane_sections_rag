#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MODEL_TYPE="llama-70b"

bash "$SCRIPT_DIR/doc_eval_rag_rand_full_k4_qwen32.sh"
