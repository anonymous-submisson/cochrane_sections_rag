#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MODEL_TYPE="gemma-3-27b"

bash "$SCRIPT_DIR/doc_eval_llm_qwen32.sh"
