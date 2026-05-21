import argparse
import gc
import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from llm_simp.simplifier_rag.models import DEFAULT_MODEL, get_model
from llm_simp.simplifier_rag.embedder import Embedder, load_review_titles
from llm_simp.simplifier_rag.utils.prompt_utils import (
    build_system_prompt,
    build_user_prompt,
)
from settings.reproducibility import set_seed

section_names = [
    "Background",
    "Objectives",
    "Search methods",
    "Selection criteria",
    "Data collection and analysis",
    "Main results",
    "Authors' conclusions",
]

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENCODE_BATCH_SIZE = 32


def load_index(index_file):
    with open(index_file, "rb") as f:
        index = pickle.load(f)
    logger.info(
        f"loaded index: {index['n_docs']} docs, dim={index['embed_dim']}, "
        f"query_encoder={index['query_encoder']}, pooling={index['pooling']}"
    )
    return index


def build_test_doc_texts(df, titles):
    texts = {}
    fallback_count = 0
    for pair_id, doc_group in df.groupby("pair_id", sort=False):
        title = titles.get(str(pair_id).upper(), "")
        if title:
            texts[pair_id] = title
        else:
            doc_group = doc_group.sort_values(["section_id", "sent_id"])
            sents = [
                str(s)
                for s in doc_group["complex"].tolist()
                if isinstance(s, str) or not pd.isna(s)
            ]
            texts[pair_id] = " ".join(sents).strip()
            fallback_count += 1
    if fallback_count:
        logger.warning(
            f"{fallback_count}/{len(texts)} test docs had no title; using concat of complex sentences"
        )
    return texts


def _copy_example_fields(sec):
    return {
        "section_name": sec["section_name"],
        "complex": sec["complex"],
        "pls": sec["pls"],
        "complex_sents": sec.get("complex_sents"),
        "labels": sec.get("labels"),
        "pls_rewrites": sec.get("pls_rewrites"),
        "pls_inserts": sec.get("pls_inserts"),
    }


def retrieve_topk(query_emb, index, section_id, exclude_pair_id, k):
    if k <= 0:
        return []
    pair_ids = index["pair_ids"]
    doc_embs = index["doc_embeddings"]
    records = index["records"]

    candidates = []
    for i, pid in enumerate(pair_ids):
        if pid == exclude_pair_id:
            continue
        sec = records[pid]["sections"].get(section_id)
        if sec is None:
            continue
        if not sec["pls"]:
            continue
        candidates.append((i, pid, sec))

    if not candidates:
        return []

    cand_idx = np.array([c[0] for c in candidates])
    cand_embs = doc_embs[cand_idx]
    cos_sims = cand_embs @ query_emb

    order = np.argsort(-cos_sims)
    selected = []
    for j in order:
        sim = float(cos_sims[j])
        _, pid, sec = candidates[j]
        ex = _copy_example_fields(sec)
        ex["pair_id"] = pid
        ex["similarity"] = sim
        selected.append(ex)
        if len(selected) >= k:
            break
    return selected


def retrieve_random(index, section_id, exclude_pair_id, k, rng):
    if k <= 0:
        return []
    pair_ids = index["pair_ids"]
    records = index["records"]

    candidates = []
    for pid in pair_ids:
        if pid == exclude_pair_id:
            continue
        sec = records[pid]["sections"].get(section_id)
        if sec is None or not sec["pls"]:
            continue
        candidates.append((pid, sec))

    if not candidates:
        return []

    n = min(k, len(candidates))
    chosen_idx = rng.choice(len(candidates), size=n, replace=False)
    selected = []
    for j in chosen_idx:
        pid, sec = candidates[int(j)]
        ex = _copy_example_fields(sec)
        ex["pair_id"] = pid
        ex["similarity"] = None
        selected.append(ex)
    return selected


def group_by_doc_and_section(df):
    groups = []
    for pair_id, doc_group in df.groupby("pair_id", sort=False):
        for section_id, sec_group in doc_group.groupby("section_id", sort=False):
            sec_group = sec_group.sort_values("sent_id")
            sents = [str(s) for s in sec_group["complex"].tolist()]
            indices = list(sec_group.index)
            sec_id = int(section_id)
            if "section_name" in sec_group.columns:
                csv_names = sec_group["section_name"].dropna().astype(str).unique()
            else:
                csv_names = []
            sec_name = csv_names[0] if len(csv_names) > 0 else section_names[sec_id]
            groups.append(
                {
                    "pair_id": pair_id,
                    "section_id": sec_id,
                    "section_name": sec_name,
                    "sents": sents,
                    "indices": indices,
                }
            )
    return groups


def build_all_prompts(groups, retrieval, example_format="split"):
    prompts = []
    index_map = []
    for g_idx, group in enumerate(groups):
        key = (group["pair_id"], group["section_id"])
        examples = retrieval.get(key, [])
        system_prompt = build_system_prompt(
            group["section_name"],
            examples,
            example_format=example_format,
        )
        for s_idx in range(len(group["sents"])):
            user = build_user_prompt(group["section_name"], group["sents"], s_idx)
            prompts.append((system_prompt, user))
            index_map.append((g_idx, s_idx))
    return prompts, index_map


def summarise_retrieval(retrieval):
    sims = [
        ex["similarity"]
        for v in retrieval.values()
        for ex in v
        if ex["similarity"] is not None
    ]
    counts = [len(v) for v in retrieval.values()]
    summary = {
        "n_queries": len(retrieval),
        "mean_examples_per_query": float(np.mean(counts)) if counts else 0.0,
        "min_examples_per_query": int(np.min(counts)) if counts else 0,
        "max_examples_per_query": int(np.max(counts)) if counts else 0,
        "n_similarities": len(sims),
    }
    if sims:
        summary.update(
            {
                "similarity_mean": float(np.mean(sims)),
                "similarity_std": float(np.std(sims)),
                "similarity_min": float(np.min(sims)),
                "similarity_max": float(np.max(sims)),
            }
        )
    return summary


def save_retrieval_debug(retrieval, path):
    serialisable = {}
    for (pair_id, sec_id), examples in retrieval.items():
        key = f"{pair_id}__{sec_id}"
        serialisable[key] = [
            {
                "pair_id": ex["pair_id"],
                "section_name": ex["section_name"],
                "similarity": ex["similarity"],
                "pls_preview": (
                    (ex["pls"][:300] + "...") if len(ex["pls"]) > 300 else ex["pls"]
                ),
            }
            for ex in examples
        ]
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)


def main(args):
    set_seed(args.seed, deterministic=args.deterministic)

    start_time = time.time()
    logger.info(f"start time: {datetime.now()}")

    index = load_index(args.index_file)

    logger.info(f"loading test file: {args.test_file}")
    df = pd.read_csv(args.test_file)

    logger.info(f"loaded {len(df)} sentences from {df['pair_id'].nunique()} documents")

    retrieval = {}

    if args.retrieval == "random":
        logger.info(f"random retrieval mode (no encoder, seed={args.seed})")
        rng = np.random.default_rng(args.seed)
        test_pair_ids = list(df["pair_id"].drop_duplicates())
        for pid in test_pair_ids:
            for section_id in range(len(section_names)):
                retrieval[(pid, section_id)] = retrieve_random(
                    index=index,
                    section_id=section_id,
                    exclude_pair_id=pid,
                    k=args.k,
                    rng=rng,
                )
        embedder = None
    else:
        logger.info(f"loading query encoder: {index['query_encoder']}")
        embedder = Embedder(
            model_name=index["query_encoder"],
            max_length=index.get("max_length", args.max_length),
            pooling=index["pooling"],
        )

        titles = load_review_titles(args.data_json)
        test_doc_texts = build_test_doc_texts(df, titles)
        test_pair_ids = list(test_doc_texts.keys())
        logger.info(f"encoding {len(test_pair_ids)} test queries (title)")
        test_embs = embedder.encode(
            [test_doc_texts[pid] for pid in test_pair_ids],
            batch_size=ENCODE_BATCH_SIZE,
            normalize=True,
        )

        for pid, emb in zip(test_pair_ids, test_embs):
            for section_id in range(len(section_names)):
                retrieval[(pid, section_id)] = retrieve_topk(
                    query_emb=emb,
                    index=index,
                    section_id=section_id,
                    exclude_pair_id=pid,
                    k=args.k,
                )

    if embedder is not None:
        del embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("released query encoder from gpu")

    summary = summarise_retrieval(retrieval)
    logger.info(f"retrieval summary: {summary}")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path = out_path.with_suffix(".retrieval.json")
    save_retrieval_debug(retrieval, debug_path)
    logger.info(f"saved retrieval debug to: {debug_path}")

    groups = group_by_doc_and_section(df)
    logger.info(f"grouped into {len(groups)} (doc, section) groups")

    prompts, index_map = build_all_prompts(
        groups,
        retrieval,
        example_format=args.example_format,
    )
    logger.info(f"built {len(prompts)} prompts")

    logger.info(f"loading simplifier model: {args.model_name} (seed={args.seed})")
    model = get_model(
        model_name=args.model_name,
        max_new_tokens=512,
        temperature=0.2,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        seed=args.seed,
        max_model_len=args.max_model_len,
    )

    logger.info("generating simplifications")
    predictions = model.simplify_batch(prompts)

    df["pred"] = ""
    for (g_idx, s_idx), pred in zip(index_map, predictions):
        row_idx = groups[g_idx]["indices"][s_idx]
        df.at[row_idx, "pred"] = pred

    df.to_csv(out_path, index=False)
    logger.info(f"saved predictions to: {out_path}")

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"end time: {datetime.now()}")
    logger.info(f"total time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="retrieval-augmented plain-language summary simplification (rag-pls)"
    )

    parser.add_argument("--test_file", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--index_file", required=True)
    parser.add_argument("--data_json", required=True)

    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--retrieval", choices=["topk", "random"], default="topk")
    parser.add_argument("--example_format", choices=["split", "whole"], default="split")

    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument(
        "--max_model_len", type=int, default=None, help="cap vLLM context window."
    )

    args = parser.parse_args()
    main(args)
