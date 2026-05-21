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


def aggregate_paragraphs_to_documents(input_csv, output_csv, doc_id_col="pair_id"):
    df = pd.read_csv(input_csv)

    def _join_nonempty(x):
        return " ".join(s for s in x.dropna().astype(str) if s.strip())

    doc_df = (
        df.groupby(doc_id_col, sort=False)
        .agg(
            {
                "complex": _join_nonempty,
                "simple": _join_nonempty,
                "pred": _join_nonempty,
            }
        )
        .reset_index()
    )

    doc_df.to_csv(output_csv, index=False)
    logger.info(f"aggregated {len(df)} paragraphs into {len(doc_df)} documents")
    logger.info(f"saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="aggregate paragraph-level predictions into document-level data"
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--doc_id_col", default="pair_id")
    args = parser.parse_args()

    aggregate_paragraphs_to_documents(args.input_csv, args.output_csv, args.doc_id_col)
