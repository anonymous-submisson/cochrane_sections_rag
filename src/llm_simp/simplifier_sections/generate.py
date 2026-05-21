import argparse
import logging
import time
from datetime import datetime

import pandas as pd

from llm_simp.simplifier_sections.models import (
    get_model,
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
)
from llm_simp.simplifier_sections.prompts import SYSTEM_PROMPT
from llm_simp.simplifier_sections.utils.prompt_utils import build_user_prompt
from settings.reproducibility import set_seed

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def group_by_doc_and_section(df):
    groups = []
    for pair_id, doc_group in df.groupby("pair_id", sort=False):
        for section_id, sec_group in doc_group.groupby("section_id", sort=False):
            sec_group = sec_group.sort_values("sent_id")

            rows = []
            for df_idx, r in zip(sec_group.index, sec_group.to_dict(orient="records")):
                complex_text = r.get("complex", "")
                if pd.isna(complex_text):
                    complex_text = ""
                rows.append({"df_index": df_idx, "complex": complex_text})

            abs_sents = [r["complex"] for r in rows]
            groups.append(
                {
                    "pair_id": pair_id,
                    "section_id": section_id,
                    "rows": rows,
                    "abs_sents": abs_sents,
                }
            )
    return groups


def build_all_prompts(groups):
    prompts = []
    index_map = []
    for g_idx, group in enumerate(groups):
        abs_sents = group["abs_sents"]
        for r_idx in range(len(group["rows"])):
            user = build_user_prompt(abs_sents, r_idx)
            prompts.append((SYSTEM_PROMPT, user))
            index_map.append({"g_idx": g_idx, "r_idx": r_idx})
    return prompts, index_map


def process_single_file(model, test_file, out_file):
    file_start = time.time()
    logger.info(f"loading: {test_file}")
    df = pd.read_csv(test_file)

    logger.info(f"loaded {len(df)} sentences from {df['pair_id'].nunique()} documents")

    groups = group_by_doc_and_section(df)
    logger.info(f"grouped into {len(groups)} (doc, section) groups")

    prompts, index_map = build_all_prompts(groups)
    logger.info(f"built {len(prompts)} prompts")

    predictions = model.simplify_batch(prompts)

    df["pred"] = ""
    for entry, pred in zip(index_map, predictions):
        row_idx = groups[entry["g_idx"]]["rows"][entry["r_idx"]]["df_index"]
        df.at[row_idx, "pred"] = pred

    df.to_csv(out_file, index=False)
    logger.info(f"saved to: {out_file}")

    file_elapsed = time.time() - file_start
    logger.info(f"completed in {int(file_elapsed // 60)}m {int(file_elapsed % 60)}s")


def main(args):
    set_seed(args.seed, deterministic=args.deterministic)

    start_time = time.time()
    logger.info(f"start time: {datetime.now()}")

    logger.info(f"loading model: {args.model_name} (seed={args.seed})")
    model = get_model(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        max_model_len=args.max_model_len,
    )

    process_single_file(model=model, test_file=args.test_file, out_file=args.out_file)

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"end time: {datetime.now()}")
    logger.info(f"total time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="section-level simplification for cochrane-sections"
    )

    parser.add_argument("--test_file", required=True)
    parser.add_argument("--out_file", required=True)

    model_choices = list(SUPPORTED_MODELS.keys())
    parser.add_argument(
        "--model_name", default=DEFAULT_MODEL, help=f"model alias {model_choices}"
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--max_model_len", type=int, default=None, help="cap vLLM context window"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    args = parser.parse_args()
    main(args)
