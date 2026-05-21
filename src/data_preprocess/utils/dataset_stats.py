import argparse
import ast
import json
import logging
import os

import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

dataset_configs = {
    "cochrane-sections": {
        "sent_files": [
            "data/cochrane-sections/cochrane-sections/cochrane_sections_sents_train.csv",
            "data/cochrane-sections/cochrane-sections/cochrane_sections_sents_val.csv",
            "data/cochrane-sections/cochrane-sections/cochrane_sections_sents_test.csv",
        ],
        "id_col": "pair_id",
    },
    "cochrane-sections-manual": {
        "sent_files": [
            "data/cochrane-sections/cochrane-sections-manual/cochrane_sections_sents_train.csv",
            "data/cochrane-sections/cochrane-sections-manual/cochrane_sections_sents_val.csv",
            "data/cochrane-sections/cochrane-sections-manual/cochrane_sections_sents_test.csv",
        ],
        "id_col": "pair_id",
    },
}


def parse_simple(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return []


def compute_stats(sent_files, id_col, repo_root):
    dfs = []
    for f in sent_files:
        path = os.path.join(repo_root, f)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
            logger.info(f"  loaded {f}: {len(dfs[-1])} rows")
        else:
            logger.warning(f"  missing: {f}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    n_sent_pairs = len(df)

    n_doc_pairs = df[id_col].nunique()

    has_para = "para_id" in df.columns
    n_para_pairs = df.groupby([id_col, "para_id"]).ngroups if has_para else None

    avg_ci = df["complex"].apply(lambda x: len(str(x).split())).mean()

    df["_simple_parsed"] = df["simple"].apply(parse_simple)
    simple_lengths = []
    for sents in df["_simple_parsed"]:
        for s in sents:
            if isinstance(s, str) and s.strip():
                simple_lengths.append(len(s.split()))
    avg_si = sum(simple_lengths) / len(simple_lengths) if simple_lengths else 0

    doc_n = df.groupby(id_col).size()
    avg_n = doc_n.mean()

    doc_k = df.groupby(id_col)["_simple_parsed"].apply(
        lambda col: sum(len(sents) for sents in col)
    )
    avg_k = doc_k.mean()

    avg_p = None
    if has_para:
        para_sizes = df.groupby([id_col, "para_id"]).size()
        avg_p = para_sizes.mean()

    labels = df["label"].copy()
    labels = labels.replace("none", "merge")
    label_counts = labels.value_counts()
    total = label_counts.sum()
    label_dist = {
        label: round(100 * count / total, 2) for label, count in label_counts.items()
    }

    stats = {
        "n_doc_pairs": int(n_doc_pairs),
        "n_sent_pairs": int(n_sent_pairs),
        "avg_ci": round(avg_ci, 2),
        "avg_si": round(avg_si, 2),
        "avg_n": round(avg_n, 2),
        "avg_k": round(avg_k, 2),
        "label_distribution": label_dist,
    }
    if n_para_pairs is not None:
        stats["n_para_pairs"] = int(n_para_pairs)
        stats["avg_p"] = round(avg_p, 2)
    return stats


def main(args):
    out_dir = os.path.join(args.repo_root, "data", "dataset_stats")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"output dir: {out_dir}")

    summary = {}
    for dataset, config in dataset_configs.items():
        logger.info(f"{dataset}")
        stats = compute_stats(config["sent_files"], config["id_col"], args.repo_root)
        if stats is None:
            logger.info(f"{dataset}: no files found, skipping")
            continue

        summary[dataset] = stats

        extras = f"  avg |c_i|={stats['avg_ci']}, avg |s_i|={stats['avg_si']}, avg n={stats['avg_n']}, avg k={stats['avg_k']}"
        if "avg_p" in stats:
            extras += f", avg p={stats['avg_p']}"
        logger.info(
            f"{dataset}: {stats['n_doc_pairs']} docs, {stats['n_sent_pairs']} sents"
        )
        logger.info(extras)
        logger.info(f"  labels: {stats['label_distribution']}")

        out_path = os.path.join(out_dir, f"{dataset}.json")
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"  saved -> {out_path}")

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"summary ({len(summary)} datasets) saved -> {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="compute preprocessed dataset statistics"
    )
    parser.add_argument("--repo_root", required=True, help="path to repo root")
    args = parser.parse_args()
    main(args)
