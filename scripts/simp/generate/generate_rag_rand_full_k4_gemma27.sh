#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MODEL_TYPE="gemma-3-27b"
export TP_SIZE="1"


bash "$SCRIPT_DIR/generate_rag_rand_full_k4_qwen32.sh"
