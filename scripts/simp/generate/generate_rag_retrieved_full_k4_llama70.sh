#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MODEL_TYPE="llama-70b"
export TP_SIZE="2"
export MAX_MODEL_LEN="8192"

bash "$SCRIPT_DIR/generate_rag_retrieved_full_k4_qwen32.sh"
