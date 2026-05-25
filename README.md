# Retrieval-Augmented Simplification of Sectioned Biomedical Abstracts

Code for the paper *Retrieval-Augmented Simplification of Sectioned Biomedical Abstracts*.

For convenience we provide bash scripts for all experiments in this paper. For specific parameters refer to the [scripts](scripts/) directory.

## Requirements

```bash
conda create -n rag_simp python=3.10 -y
pip install -r requirements_rag_simp.txt
```

```bash
export HF_TOKEN="YOURTOKENHERE"
```

## Data

Either download manually from [COCHRANE-SECTIONS](https://github.com/JanB100/cochrane-sections), and copy `alignments/` and `data/cochrane/data.json` into `data/cochrane-sections/`; or run:

```bash
bash scripts/preprocessing/download_cochrane-sections.sh
```

## Data Preprocessing

To preprocess the Cochrane-sections data, run:

```bash
bash scripts/preprocessing/process_cochrane-sections.sh
```

## Generation

Each of the five systems in the paper has one script per model family (Qwen2.5-32B, Gemma-3-27B, Llama-3.1-70B). Outputs are written to `data/outputs/generations/`, and index is cached at `data/intermediate/rag_index/`

*note: the generation scripts default to 1 GPU for Qwen and Gemma, and 2 GPUs for Llama. To run with a different number of GPUs, override the TP_SIZE variable.*

To generate simplifications with our main system, run:

```bash
bash scripts/simp/generate/generate_rag_retrieved_split_k4_{qwen32|gemma27|llama70}.sh
```

To generate simplifications with the four baselines, run:

```bash
bash scripts/simp/generate/generate_llm_{qwen32|gemma27|llama70}.sh
bash scripts/simp/generate/generate_rag_rand_full_k4_{qwen32|gemma27|llama70}.sh
bash scripts/simp/generate/generate_rag_rand_split_k4_{qwen32|gemma27|llama70}.sh
bash scripts/simp/generate/generate_rag_retrieved_full_k4_{qwen32|gemma27|llama70}.sh
```

## Evaluation

To evaluate the simplifications of our main system, run:

```bash
bash scripts/simp/eval/doc_eval_rag_retrieved_split_k4_{qwen32|gemma27|llama70}.sh
```

To evaluate the simplifications of the four baselines, run:

```bash
bash scripts/simp/eval/doc_eval_llm_{qwen32|gemma27|llama70}.sh
bash scripts/simp/eval/doc_eval_rag_rand_full_k4_{qwen32|gemma27|llama70}.sh
bash scripts/simp/eval/doc_eval_rag_rand_split_k4_{qwen32|gemma27|llama70}.sh
bash scripts/simp/eval/doc_eval_rag_retrieved_full_k4_{qwen32|gemma27|llama70}.sh
```

## Acknowledgements

We adapt and build upon code from the following repositories:

- Anonymous
- Anonymous
- Anonymous
