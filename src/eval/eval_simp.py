import json
import logging
import re

import fire
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

from easse.fkgl import corpus_fkgl
from transformers import BartTokenizer

from eval.easse_sari import get_corpus_sari_operation_scores
from eval.smart_eval import matching_functions, scorer
from settings import set_seed

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_METRICS = ["sari", "fkgl", "smart"]


def read_file(filename):
    return [d.strip() for d in open(filename).readlines()]


def clean_sequences(input_seqs, output_seqs, ref_seqs=None):
    if ref_seqs is None:
        ref_seqs = []

    output_seqs = [re.sub(r"\[PLAN\].*\[SIMPLIFICATION\]", "", t) for t in output_seqs]
    output_seqs = [
        re.sub(r"(\<COPY\>|\<REPHRASE\>|\<SPLIT\>|\<DELETE\>)", "", t)
        for t in output_seqs
    ]
    input_seqs = [re.sub(r"\<\\?/?s\>", "", t) for t in input_seqs]
    input_seqs = [re.sub(r"\<\SEP\>", "", t) for t in input_seqs]
    ref_seqs = [[re.sub(r"\<\\?/?s\>", "", t[0])] for t in ref_seqs]
    ref_seqs = [[re.sub(r"\<SEP\>", "", t[0])] for t in ref_seqs]
    output_seqs = [re.sub(r"\<\\?/?s\>", "", t) for t in output_seqs]
    output_seqs = [re.sub(r"\<pad\>", "", t) for t in output_seqs]
    input_seqs = [re.sub(r" +", " ", t) for t in input_seqs]
    output_seqs = [re.sub(r" +", " ", t) for t in output_seqs]
    ref_seqs = [[re.sub(r" +", " ", t[0])] for t in ref_seqs]

    return input_seqs, output_seqs, ref_seqs


def calculate_sari(in_doc, out_doc, ref_docs):
    sari = get_corpus_sari_operation_scores(
        [in_doc], [out_doc], [[x] for x in ref_docs]
    )
    return [np.mean(sari), *sari]


def evaluate(
    input_data,
    x_col=None,
    y_col=None,
    r_col=None,
    prepro=False,
    metrics=DEFAULT_METRICS,
    out_file=None,
    skip_baseline=False,
    seed=42,
    deterministic=False,
):
    set_seed(seed, deterministic=deterministic)

    if isinstance(input_data, str):
        input_data = pd.read_csv(input_data)

    input_data[y_col] = input_data[y_col].fillna("")
    input_seqs = list(input_data[x_col])
    output_seqs = list(input_data[y_col])
    ref_seqs = [[d] for d in input_data[r_col]]

    input_seqs, output_seqs, ref_seqs = clean_sequences(
        input_seqs, output_seqs, ref_seqs
    )

    tokenizer = BartTokenizer.from_pretrained(
        "facebook/bart-base", add_prefix_space=False
    )

    n = len(input_seqs)
    lens = np.zeros(n)
    nsents = np.zeros(n)
    results = {
        "sari": np.zeros((n, 4)),
        "fkgl": np.zeros(n),
        "smart": np.zeros((n, 3)),
    }

    out_docs = []
    ref_docss = []
    for i in tqdm(range(n), desc="evaluating", mininterval=30):
        in_doc = input_seqs[i]
        out_doc = output_seqs[i]
        ref_docs = ref_seqs[i]

        if prepro:
            in_doc = tokenizer.decode(
                tokenizer(in_doc)["input_ids"], skip_special_tokens=True
            )
            out_doc = tokenizer.decode(
                tokenizer(out_doc)["input_ids"], skip_special_tokens=True
            )
            for j in range(len(ref_docs)):
                ref_docs[j] = tokenizer.decode(
                    tokenizer(ref_docs[j])["input_ids"], skip_special_tokens=True
                )

        if "sari" in metrics:
            results["sari"][i] = calculate_sari(in_doc, out_doc, ref_docs)

        out_doc_sents = nltk.sent_tokenize(out_doc)
        ref_docs_sents = [nltk.sent_tokenize(ref_doc) for ref_doc in ref_docs]

        if "fkgl" in metrics:
            results["fkgl"][i] = corpus_fkgl(out_doc_sents)

        if "smart" in metrics:
            matcher = matching_functions.chrf_matcher
            smart_scorer = scorer.SmartScorer(matching_fn=matcher)

            if out_doc_sents == []:
                out_doc_sents = [""]

            smarts = smart_scorer.smart_score(ref_docs_sents[0], out_doc_sents)[
                "smartL"
            ]
            results["smart"][i] = np.array(
                [smarts["precision"], smarts["recall"], smarts["fmeasure"]]
            )

        lens[i] = len(tokenizer(out_doc)["input_ids"])
        nsents[i] = len(out_doc_sents)

        out_docs.append(out_doc)
        ref_docss.append(ref_docs)

    input_data["sari"] = results["sari"][:, 0]
    input_data["sari_a"] = results["sari"][:, 1]
    input_data["sari_k"] = results["sari"][:, 2]
    input_data["sari_d"] = results["sari"][:, 3]
    input_data["fkgl"] = results["fkgl"]
    input_data["pred_len"] = lens
    input_data["pred_num_sents"] = nsents
    if "smart" in metrics:
        input_data["smart_p"] = results["smart"][:, 0]
        input_data["smart_r"] = results["smart"][:, 1]
        input_data["smart_f1"] = results["smart"][:, 2]

    input_data["pred"] = out_docs

    if "fkgl" in metrics:
        logger.info(f"FKGL: {input_data['fkgl'].mean():.4f}")
    if "sari" in metrics:
        logger.info(f"SARI: {input_data['sari'].mean():.4f}")
        for o, key in zip("akd", ["sari_a", "sari_k", "sari_d"]):
            logger.info(f"  {o}: {input_data[key].mean():.4f}")
    if "smart" in metrics:
        logger.info(
            f"SMART P={input_data['smart_p'].mean():.4f} "
            f"R={input_data['smart_r'].mean():.4f} "
            f"F1={input_data['smart_f1'].mean():.4f}"
        )
    logger.info(
        f"avg len: {input_data['pred_len'].mean():.2f} tokens, "
        f"{input_data['pred_num_sents'].mean():.2f} sentences"
    )

    baseline_metrics = {"input": {}, "reference": {}}

    if not skip_baseline:
        logger.info("computing baseline metrics (input and reference)")
        matcher = matching_functions.chrf_matcher
        smart_scorer = scorer.SmartScorer(matching_fn=matcher)

        input_smart = np.zeros((n, 3))
        input_fkgl = np.zeros(n)
        input_sari = np.zeros(n)
        input_lens = np.zeros(n)
        input_nsents = np.zeros(n)

        ref_smart = np.zeros((n, 3))
        ref_fkgl = np.zeros(n)
        ref_sari = np.zeros(n)
        ref_lens = np.zeros(n)
        ref_nsents = np.zeros(n)

        for i in tqdm(range(n), desc="baseline metrics", mininterval=30):
            in_doc = input_seqs[i]
            ref_doc = ref_seqs[i][0]

            if prepro:
                in_doc = tokenizer.decode(
                    tokenizer(in_doc)["input_ids"], skip_special_tokens=True
                )
                ref_doc = tokenizer.decode(
                    tokenizer(ref_doc)["input_ids"], skip_special_tokens=True
                )

            in_doc_sents = nltk.sent_tokenize(in_doc)
            ref_doc_sents = nltk.sent_tokenize(ref_doc)

            if in_doc_sents == []:
                in_doc_sents = [""]
            input_smarts = smart_scorer.smart_score(ref_doc_sents, in_doc_sents)[
                "smartL"
            ]
            input_smart[i] = np.array(
                [
                    input_smarts["precision"],
                    input_smarts["recall"],
                    input_smarts["fmeasure"],
                ]
            )
            input_fkgl[i] = corpus_fkgl(in_doc_sents)
            input_sari[i] = np.mean(
                get_corpus_sari_operation_scores([in_doc], [in_doc], [[ref_doc]])
            )
            input_lens[i] = len(tokenizer(in_doc)["input_ids"])
            input_nsents[i] = len(in_doc_sents)

            if ref_doc_sents == []:
                ref_doc_sents = [""]
            ref_smarts = smart_scorer.smart_score(ref_doc_sents, ref_doc_sents)[
                "smartL"
            ]
            ref_smart[i] = np.array(
                [ref_smarts["precision"], ref_smarts["recall"], ref_smarts["fmeasure"]]
            )
            ref_fkgl[i] = corpus_fkgl(ref_doc_sents)
            ref_sari[i] = np.mean(
                get_corpus_sari_operation_scores([in_doc], [ref_doc], [[ref_doc]])
            )
            ref_lens[i] = len(tokenizer(ref_doc)["input_ids"])
            ref_nsents[i] = len(ref_doc_sents)

        baseline_metrics["input"] = {
            "smart_p": float(input_smart[:, 0].mean()),
            "smart_r": float(input_smart[:, 1].mean()),
            "smart_f1": float(input_smart[:, 2].mean()),
            "fkgl": float(input_fkgl.mean()),
            "sari": float(input_sari.mean()),
            "avg_len_tokens": float(input_lens.mean()),
            "avg_len_sentences": float(input_nsents.mean()),
        }
        baseline_metrics["reference"] = {
            "smart_p": float(ref_smart[:, 0].mean()),
            "smart_r": float(ref_smart[:, 1].mean()),
            "smart_f1": float(ref_smart[:, 2].mean()),
            "fkgl": float(ref_fkgl.mean()),
            "sari": float(ref_sari.mean()),
            "avg_len_tokens": float(ref_lens.mean()),
            "avg_len_sentences": float(ref_nsents.mean()),
        }

    if out_file is not None:
        input_data.to_csv(out_file, index=False)
        logger.info(f"per-document results saved to {out_file}")

        summary = {
            "num_documents": len(input_data),
            "prediction": {
                "smart_p": (
                    float(input_data["smart_p"].mean()) if "smart" in metrics else None
                ),
                "smart_r": (
                    float(input_data["smart_r"].mean()) if "smart" in metrics else None
                ),
                "smart_f1": (
                    float(input_data["smart_f1"].mean()) if "smart" in metrics else None
                ),
                "fkgl": float(input_data["fkgl"].mean()) if "fkgl" in metrics else None,
                "sari": float(input_data["sari"].mean()) if "sari" in metrics else None,
                "avg_len_tokens": float(input_data["pred_len"].mean()),
                "avg_len_sentences": float(input_data["pred_num_sents"].mean()),
            },
        }

        if baseline_metrics["input"]:
            summary["input"] = baseline_metrics["input"]
            summary["reference"] = baseline_metrics["reference"]

        if "sari" in metrics:
            summary["prediction"]["sari_a"] = float(input_data["sari_a"].mean())
            summary["prediction"]["sari_k"] = float(input_data["sari_k"].mean())
            summary["prediction"]["sari_d"] = float(input_data["sari_d"].mean())

        summary_file = (
            out_file.replace(".csv", "_summary.json")
            if out_file.endswith(".csv")
            else out_file + "_summary.json"
        )
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"summary saved to {summary_file}")

    return input_data


if __name__ == "__main__":

    def cli(**kwargs):
        evaluate(**kwargs)

    fire.Fire(cli)
