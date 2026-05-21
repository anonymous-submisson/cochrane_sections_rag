import argparse
import ast
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from llm_simp.simplifier_rag.embedder import Embedder, load_review_titles
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


def parse_simple_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s or s == "[]":
        return []
    return ast.literal_eval(s)


def _collapse_label(raw_label):
    if not isinstance(raw_label, str):
        return "KEEP"
    return "DELETE" if raw_label.strip().lower() == "delete" else "KEEP"


def _dedup_preserve(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _collect_pls_sents(simple_cells):
    out = []
    for simple_cell in simple_cells:
        for it in parse_simple_list(simple_cell):
            if isinstance(it, str) and it.strip():
                out.append(it.strip())
    return out


def build_doc_records(df):
    has_label = "label" in df.columns
    records = {}
    for pair_id, doc_group in df.groupby("pair_id", sort=False):
        sections = {}
        complex_all = []
        for section_id, sec_group in doc_group.groupby("section_id", sort=False):
            sec_group = sec_group.sort_values("sent_id")

            if has_label:
                is_insert_row = sec_group["label"].astype(str).str.lower() == "insert"
            else:
                is_insert_row = pd.Series(False, index=sec_group.index)
            abs_rows = sec_group[~is_insert_row]
            ins_rows = sec_group[is_insert_row]

            complex_sents_raw = abs_rows["complex"].tolist()
            complex_sents = [
                str(s)
                for s in complex_sents_raw
                if isinstance(s, str) or not pd.isna(s)
            ]
            if has_label:
                label_sents_raw = abs_rows["label"].tolist()
                labels = [
                    _collapse_label(lab)
                    for lab, s in zip(label_sents_raw, complex_sents_raw)
                    if isinstance(s, str) or not pd.isna(s)
                ]
            else:
                labels = ["KEEP"] * len(complex_sents)

            pls_rewrites = _dedup_preserve(
                _collect_pls_sents(abs_rows["simple"].tolist())
            )
            pls_inserts = _dedup_preserve(
                _collect_pls_sents(ins_rows["simple"].tolist())
            )
            pls_combined = _dedup_preserve(pls_rewrites + pls_inserts)

            sec_id = int(section_id)
            csv_names = (
                sec_group["section_name"].dropna().astype(str).unique()
                if "section_name" in sec_group.columns
                else []
            )
            sec_name = csv_names[0] if len(csv_names) > 0 else section_names[sec_id]
            sections[sec_id] = {
                "section_name": sec_name,
                "complex": " ".join(complex_sents).strip(),
                "pls": " ".join(pls_combined).strip(),
                "pls_rewrites": pls_rewrites,
                "pls_inserts": pls_inserts,
                "complex_sents": complex_sents,
                "labels": labels,
            }
            complex_all.extend(complex_sents)
        records[pair_id] = {
            "complex_all": " ".join(complex_all).strip(),
            "sections": sections,
        }
    return records


def main(args):
    set_seed(args.seed, deterministic=args.deterministic)

    start_time = time.time()
    logger.info(f"start time: {datetime.now()}")
    logger.info(f"loading pool: {args.pool_file}")

    df = pd.read_csv(args.pool_file)
    logger.info(f"loaded {len(df)} rows from {df['pair_id'].nunique()} pool documents")

    records = build_doc_records(df)
    pair_ids = list(records.keys())
    logger.info(f"built records for {len(pair_ids)} documents")

    per_section_counts = {i: 0 for i in range(len(section_names))}
    for rec in records.values():
        for sec_id, sec in rec["sections"].items():
            if sec["pls"]:
                per_section_counts[sec_id] += 1
    for sec_id, name in enumerate(section_names):
        logger.info(
            f"pool examples with non-empty pls for section {sec_id} ({name}): {per_section_counts[sec_id]}"
        )

    titles = load_review_titles(args.data_json)
    n_titled = 0
    for pid in pair_ids:
        title = titles.get(pid.upper(), "")
        records[pid]["title"] = title
        if title:
            n_titled += 1
    logger.info(f"matched titles for {n_titled}/{len(pair_ids)} pool docs")

    logger.info(
        f"loading article encoder: {args.article_encoder} (pooling={args.pooling})"
    )
    article_embedder = Embedder(
        model_name=args.article_encoder,
        max_length=args.max_length,
        pooling=args.pooling,
    )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    index = {
        "article_encoder": args.article_encoder,
        "query_encoder": args.query_encoder,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "n_docs": len(pair_ids),
        "pair_ids": pair_ids,
        "records": records,
        "source_file": str(args.pool_file),
        "data_json": str(args.data_json),
    }

    pool_texts = []
    for pid in pair_ids:
        title = records[pid].get("title", "")
        complex_all = records[pid].get("complex_all", "")
        pool_texts.append(title if title else complex_all)
    logger.info(
        f"pool text chars: "
        f"min={min(len(t) for t in pool_texts)}, max={max(len(t) for t in pool_texts)}, "
        f"mean={sum(len(t) for t in pool_texts)/len(pool_texts):.0f}"
    )
    logger.info(f"encoding {len(pool_texts)} pool titles/fallbacks")
    doc_embeddings = article_embedder.encode(
        pool_texts,
        batch_size=args.batch_size,
        normalize=True,
    )
    logger.info(f"doc embeddings shape: {doc_embeddings.shape}")
    index["embed_dim"] = int(doc_embeddings.shape[1])
    index["doc_embeddings"] = doc_embeddings

    with open(out_path, "wb") as f:
        pickle.dump(index, f)
    logger.info(f"saved index to: {out_path}")

    elapsed = time.time() - start_time
    logger.info(f"end time: {datetime.now()}")
    logger.info(f"total time: {int(elapsed // 60)}m {int(elapsed % 60)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="build retrieval index for rag-pls over cochrane-sections train pool"
    )

    parser.add_argument("--pool_file", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--article_encoder", default="ncbi/MedCPT-Article-Encoder")
    parser.add_argument("--query_encoder", default="ncbi/MedCPT-Query-Encoder")
    parser.add_argument("--pooling", default="cls", choices=["cls", "mean", "last"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    args = parser.parse_args()
    main(args)
