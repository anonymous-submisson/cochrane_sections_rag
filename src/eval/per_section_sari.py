import argparse
import ast
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import nltk

from easse.fkgl import corpus_fkgl

from eval.easse_sari import get_corpus_sari_operation_scores

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

section_names = [
    "Background",
    "Objectives",
    "Search methods",
    "Selection criteria",
    "Data collection and analysis",
    "Main results",
    "Authors' conclusions",
]


def parse_simple(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return [s for s in parsed if isinstance(s, str) and s.strip()]
    except (ValueError, SyntaxError):
        pass
    return []


def build_section_triples(df):
    triples = []
    for (pair_id, section_id), group in df.groupby(
        ["pair_id", "section_id"], sort=False
    ):
        group = group.sort_values("sent_id")
        complex_str = " ".join(str(s) for s in group["complex"] if pd.notna(s))
        pred_str = " ".join(
            str(s) for s in group["pred"].fillna("").astype(str) if str(s).strip()
        )
        simple_sents = []
        for v in group["simple"]:
            simple_sents.extend(parse_simple(v))
        simple_str = " ".join(simple_sents)
        triples.append(
            {
                "pair_id": pair_id,
                "section_id": int(section_id),
                "complex": complex_str,
                "pred": pred_str,
                "simple": simple_str,
                "n_sents": len(group),
            }
        )
    return triples


def build_doc_triples(df):
    triples = []
    for pair_id, group in df.groupby("pair_id", sort=False):
        group = group.sort_values(["section_id", "sent_id"])
        complex_str = " ".join(str(s) for s in group["complex"] if pd.notna(s))
        pred_str = " ".join(
            str(s) for s in group["pred"].fillna("").astype(str) if str(s).strip()
        )
        simple_sents = []
        for v in group["simple"]:
            simple_sents.extend(parse_simple(v))
        simple_str = " ".join(simple_sents)
        triples.append(
            {
                "pair_id": pair_id,
                "complex": complex_str,
                "pred": pred_str,
                "simple": simple_str,
            }
        )
    return triples


def compute_whole_doc(triples):
    kept = [t for t in triples if t["simple"].strip()]
    if not kept:
        return {
            "sari": np.nan,
            "sari_a": np.nan,
            "sari_k": np.nan,
            "sari_d": np.nan,
            "fkgl_pred": np.nan,
            "pred_tokens": np.nan,
            "n_docs": 0,
        }
    saris = np.zeros((len(kept), 4))
    token_counts = []
    all_pred_sents = []
    for i, t in enumerate(kept):
        sari = get_corpus_sari_operation_scores(
            [t["complex"]], [t["pred"]], [[t["simple"]]]
        )
        saris[i] = [np.mean(sari), *sari]
        token_counts.append(len(t["pred"].split()))
        all_pred_sents.extend(
            nltk.sent_tokenize(t["pred"]) if t["pred"].strip() else [""]
        )
    fkgl = corpus_fkgl(all_pred_sents) if all_pred_sents else np.nan
    return {
        "n_docs": len(kept),
        "sari": round(float(saris[:, 0].mean()), 2),
        "sari_a": round(float(saris[:, 1].mean()), 2),
        "sari_k": round(float(saris[:, 2].mean()), 2),
        "sari_d": round(float(saris[:, 3].mean()), 2),
        "fkgl_pred": round(float(fkgl), 2),
        "pred_tokens": round(float(np.mean(token_counts)), 1),
    }


def compute_per_section(triples):
    rows = []
    for section_id in range(7):
        sec_triples = [t for t in triples if t["section_id"] == section_id]
        kept = [t for t in sec_triples if t["simple"].strip()]
        n_docs = len(sec_triples)
        n_with_ref = len(kept)
        if not kept:
            rows.append(
                {
                    "section_id": section_id,
                    "section_name": section_names[section_id],
                    "n_docs": n_docs,
                    "n_with_ref": 0,
                    "sari": np.nan,
                    "sari_a": np.nan,
                    "sari_k": np.nan,
                    "sari_d": np.nan,
                    "fkgl_pred": np.nan,
                    "pred_tokens": np.nan,
                }
            )
            continue

        saris = np.zeros((len(kept), 4))
        pred_token_counts = []
        pred_sents_all = []
        for i, t in enumerate(kept):
            sari = get_corpus_sari_operation_scores(
                [t["complex"]], [t["pred"]], [[t["simple"]]]
            )
            saris[i] = [np.mean(sari), *sari]
            pred_token_counts.append(len(t["pred"].split()))
            pred_sents = nltk.sent_tokenize(t["pred"]) if t["pred"].strip() else [""]
            pred_sents_all.extend(pred_sents)

        fkgl = corpus_fkgl(pred_sents_all) if pred_sents_all else np.nan

        rows.append(
            {
                "section_id": section_id,
                "section_name": section_names[section_id],
                "n_docs": n_docs,
                "n_with_ref": n_with_ref,
                "sari": round(float(saris[:, 0].mean()), 2),
                "sari_a": round(float(saris[:, 1].mean()), 2),
                "sari_k": round(float(saris[:, 2].mean()), 2),
                "sari_d": round(float(saris[:, 3].mean()), 2),
                "fkgl_pred": round(float(fkgl), 2),
                "pred_tokens": round(float(np.mean(pred_token_counts)), 1),
            }
        )
    return rows


def format_cell(val, width=14):
    if val is None or pd.isna(val):
        return f" {'-':>{width}}"
    return f" {val:>{width}.2f}"


def log_metric_table(metric, labels, per_section_results, whole_doc_results):
    logger.info("")
    logger.info(f"{metric} per section (vs aligned-only simple)")
    header = f"{'section':<32}"
    for label in labels:
        header += f" {label[:14]:>14}"
    logger.info(header)
    for section_id in range(7):
        row = f"{section_names[section_id]:<32}"
        for label in labels:
            rows = per_section_results[label]
            r = next((r for r in rows if r["section_id"] == section_id), None)
            val = r[metric] if r else np.nan
            row += format_cell(val)
        logger.info(row)
    row = f"{'whole-doc (aligned only)':<32}"
    for label in labels:
        row += format_cell(whole_doc_results[label].get(metric))
    logger.info(row)


def main(args):
    nltk.download("punkt_tab", quiet=True)

    pred_paths = [Path(p.strip()) for p in args.input_csvs.split(",") if p.strip()]
    labels = (
        [l.strip() for l in args.labels.split(",")]
        if args.labels
        else [p.stem for p in pred_paths]
    )
    if len(labels) != len(pred_paths):
        raise ValueError("number of labels must match number of input csvs")

    per_section_results = {}
    whole_doc_results = {}
    for path, label in zip(pred_paths, labels):
        logger.info(f"processing: {label} ({path})")
        df = pd.read_csv(path)
        required = {"pair_id", "section_id", "sent_id", "complex", "pred", "simple"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        per_section_results[label] = compute_per_section(build_section_triples(df))
        whole_doc_results[label] = compute_whole_doc(build_doc_triples(df))

    metrics_to_show = ["sari", "sari_k", "sari_d", "pred_tokens", "fkgl_pred"]
    for metric in metrics_to_show:
        log_metric_table(metric, labels, per_section_results, whole_doc_results)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "per_section": per_section_results,
                    "whole_doc_aligned": whole_doc_results,
                },
                f,
                indent=2,
            )
        logger.info(f"saved per-section results to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="per-section SARI analysis for cochrane-sections"
    )
    parser.add_argument(
        "--input_csvs",
        required=True,
        help="comma-separated list of sentence-level prediction csv paths",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="comma-separated list of display labels (one per input csv)",
    )
    parser.add_argument("--output_json", default=None, help="optional json output path")
    args = parser.parse_args()
    main(args)
