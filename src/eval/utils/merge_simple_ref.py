import argparse
import logging

import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def merge_simple_ref(doc_file, docs_csv, ref_col="simple_full", doc_id_col="pair_id"):
    doc = pd.read_csv(doc_file)
    ref = pd.read_csv(docs_csv)[[doc_id_col, ref_col]]
    doc = doc.merge(ref, on=doc_id_col, how="left")
    doc.to_csv(doc_file, index=False)
    logger.info(f"merged {ref_col} into {len(doc)} documents -> {doc_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="merge a reference column from the docs csv into an aggregated doc-level csv"
    )
    parser.add_argument(
        "--doc_file",
        required=True,
        help="aggregated doc csv; rewritten in place with the ref column added",
    )
    parser.add_argument(
        "--docs_csv",
        required=True,
        help="docs_test.csv containing doc_id_col and the ref column",
    )
    parser.add_argument("--ref_col", default="simple_full")
    parser.add_argument("--doc_id_col", default="pair_id")
    args = parser.parse_args()

    merge_simple_ref(args.doc_file, args.docs_csv, args.ref_col, args.doc_id_col)
