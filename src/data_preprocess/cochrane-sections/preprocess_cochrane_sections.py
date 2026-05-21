import argparse
import ast
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict

import Levenshtein
import nltk
import pandas as pd

_junk_stopwords = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "although",
    "among",
    "and",
    "another",
    "any",
    "are",
    "around",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "can",
    "cannot",
    "could",
    "did",
    "does",
    "doing",
    "done",
    "down",
    "during",
    "each",
    "either",
    "ever",
    "every",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "here",
    "how",
    "however",
    "including",
    "into",
    "its",
    "itself",
    "just",
    "less",
    "like",
    "many",
    "may",
    "might",
    "more",
    "most",
    "much",
    "must",
    "near",
    "nearly",
    "neither",
    "never",
    "nor",
    "not",
    "now",
    "often",
    "once",
    "only",
    "other",
    "others",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "should",
    "since",
    "some",
    "somewhat",
    "still",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "thus",
    "too",
    "under",
    "until",
    "upon",
    "very",
    "was",
    "were",
    "what",
    "when",
    "where",
    "whether",
    "which",
    "while",
    "who",
    "whom",
    "whose",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "yet",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def _content_words(text):
    return [
        w
        for w in re.findall(r"[a-z][a-z\-]+", text.lower())
        if len(w) > 3 and w not in _junk_stopwords
    ]


def is_alignment_junk(insert_text):
    t = insert_text.strip()
    if not t:
        return True
    if t[0] in ")]}.,":
        return True
    if t.count(")") > t.count("("):
        return True
    if t.count("]") > t.count("["):
        return True
    if len(_content_words(t)) < 2:
        return True
    return False


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

copy_threshold = 0.92


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return {d["doi"]: d for d in data}


def load_alignments(path):
    with open(path) as f:
        return json.load(f)


def sent_tokenize_with_headings(sections):
    full_list = []
    content_indices = []
    section_ids = []
    para_ids = []
    current_para = 0

    for sect_id, sect in enumerate(sections):
        full_list.append(sect["heading"])

        text = sect["text"]
        sents = nltk.sent_tokenize(text)

        prev_end = 0
        for k, s in enumerate(sents):
            content_indices.append(len(full_list))
            section_ids.append(sect_id)

            start = text.find(s, prev_end)
            if k > 0 and "\n" in text[prev_end:start]:
                current_para += 1
            para_ids.append(current_para)

            prev_end = start + len(s)
            full_list.append(s)

        current_para += 1

    return full_list, content_indices, section_ids, para_ids


def sent_tokenize_pls(item):
    if item["pls_type"] == "long":
        sents = nltk.sent_tokenize(item["pls"])
        return sents, list(range(len(sents)))

    full_list = []
    content_indices = []
    for sect in item["pls"]:
        full_list.append(sect["heading"])
        for s in nltk.sent_tokenize(sect["text"]):
            content_indices.append(len(full_list))
            full_list.append(s)

    return full_list, content_indices


def build_alignment_maps(triplets, abs_content_indices, pls_content_indices):
    abs_full_to_content = {v: i for i, v in enumerate(abs_content_indices)}
    pls_full_to_content = {v: i for i, v in enumerate(pls_content_indices)}

    abs_to_pls = defaultdict(set)
    pls_to_abs = defaultdict(set)

    for pls_idx, abs_idx, _ in triplets:
        abs_c = abs_full_to_content.get(abs_idx)
        pls_c = pls_full_to_content.get(pls_idx)
        if abs_c is None or pls_c is None:
            continue
        abs_to_pls[abs_c].add(pls_c)
        pls_to_abs[pls_c].add(abs_c)

    return abs_to_pls, pls_to_abs


def get_content_sentences(full_list, content_indices):
    return [full_list[i] for i in content_indices]


def assign_label(complex_sent, aligned_simple_sents):
    if not aligned_simple_sents:
        return "delete"
    if len(aligned_simple_sents) > 1:
        return "split"
    if Levenshtein.ratio(complex_sent, aligned_simple_sents[0]) >= copy_threshold:
        return "ignore"
    return "rephrase"


def apply_merge_labels(rows):
    n = len(rows)
    merged = set()

    for i in range(n):
        if i in merged:
            continue
        if rows[i]["label"] in ("delete", "split"):
            continue

        pls_ids = rows[i]["_aligned_pls_ids"]
        if len(pls_ids) != 1:
            continue
        pls_id = next(iter(pls_ids))

        group = [i]
        j = i + 1
        while j < n and rows[j]["_aligned_pls_ids"] == {pls_id}:
            group.append(j)
            j += 1

        if len(group) < 2:
            continue

        rows[group[0]]["label"] = "merge"
        rows[group[1]]["label"] = "none"
        for idx in group[2:]:
            rows[idx]["label"] = "delete"
            rows[idx]["simple"] = "[]"
            rows[idx]["_aligned_pls_ids"] = set()
        merged.update(group)


def assign_simp_sent_ids(doc_rows):
    cur = 0
    for r in doc_rows:
        r["simp_sent_id"] = cur
        label = r["label"]
        if label in ("rephrase", "ignore"):
            cur += 1
        elif label == "split":
            cur += max(len(r["_aligned_pls_ids"]), 1)
        elif label == "none":
            cur += 1


def assign_insert_section(k, pls_to_abs, abs_section_ids, abs_para_ids, n_abs):
    aligned = sorted(pls_to_abs.keys())
    if not aligned or n_abs == 0:
        return 0, 0

    best = aligned[0]
    best_dist = abs(best - k)
    for a in aligned[1:]:
        d = abs(a - k)
        if d < best_dist or (d == best_dist and a < best):
            best = a
            best_dist = d

    anchor_j = min(pls_to_abs[best])
    if anchor_j >= n_abs:
        anchor_j = n_abs - 1
    return abs_section_ids[anchor_j], abs_para_ids[anchor_j]


def synthesize_insert_rows(
    pair_id,
    pls_to_abs,
    pls_sents,
    abs_sents,
    abs_section_ids,
    abs_para_ids,
    n_abs,
):
    n_pls = len(pls_sents)
    aligned = set(pls_to_abs.keys())
    unaligned = [k for k in range(n_pls) if k not in aligned]
    if not unaligned:
        return []

    rows = []
    n_junk = 0
    for k in unaligned:
        insert_text = pls_sents[k]
        if is_alignment_junk(insert_text):
            n_junk += 1
            continue

        sec_id, para_id = assign_insert_section(
            k, pls_to_abs, abs_section_ids, abs_para_ids, n_abs
        )
        doc_pos = (k + 1) / max(n_pls, 1)

        rows.append(
            {
                "pair_id": pair_id,
                "section_id": sec_id,
                "section_name": section_names[sec_id],
                "para_id": para_id,
                "sent_id": 1000 + len(rows),
                "complex": "",
                "label": "insert",
                "simple": str([insert_text]),
                "doc_pos": doc_pos,
                "doc_quint": max(1, math.ceil(doc_pos / 0.2)),
                "doc_len": n_abs,
                "_aligned_pls_ids": set(),
                "simp_sent_id": pd.NA,
            }
        )
    return rows


def build_sentence_level_dataset(data_by_doi, alignment_dict):
    rows = []
    n_skipped_sections = 0

    for doi, triplets in alignment_dict.items():
        item = data_by_doi.get(doi)
        if item is None:
            continue

        if len(item["abstract"]) != 7:
            n_skipped_sections += 1
            continue

        abs_full, abs_content_indices, section_ids, para_ids = (
            sent_tokenize_with_headings(item["abstract"])
        )
        pls_full, pls_content_indices = sent_tokenize_pls(item)

        abs_to_pls, pls_to_abs = build_alignment_maps(
            triplets, abs_content_indices, pls_content_indices
        )

        abs_sents = get_content_sentences(abs_full, abs_content_indices)
        pls_sents = get_content_sentences(pls_full, pls_content_indices)
        n_abs = len(abs_sents)

        pair_id = doi.split(".")[2]
        doc_rows = []
        para_start = 0

        for j in range(n_abs):
            doc_pos = (j + 1) / n_abs
            aligned_pls_ids = sorted(abs_to_pls.get(j, set()))
            aligned_sents = [pls_sents[k] for k in aligned_pls_ids]

            doc_rows.append(
                {
                    "pair_id": pair_id,
                    "section_id": section_ids[j],
                    "section_name": section_names[section_ids[j]],
                    "para_id": para_ids[j],
                    "sent_id": j,
                    "complex": abs_sents[j],
                    "label": assign_label(abs_sents[j], aligned_sents),
                    "simple": str(aligned_sents),
                    "doc_pos": doc_pos,
                    "doc_quint": math.ceil(doc_pos / 0.2),
                    "doc_len": n_abs,
                    "_aligned_pls_ids": set(aligned_pls_ids),
                }
            )

            is_last = j + 1 == n_abs
            para_ends = is_last or para_ids[j + 1] > para_ids[j]

            if para_ends:
                apply_merge_labels(doc_rows[para_start:])
                para_start = j + 1

        assign_simp_sent_ids(doc_rows)

        insert_rows = synthesize_insert_rows(
            pair_id,
            pls_to_abs,
            pls_sents,
            abs_sents,
            section_ids,
            para_ids,
            n_abs,
        )
        doc_rows.extend(insert_rows)

        rows.extend(doc_rows)

    for r in rows:
        del r["_aligned_pls_ids"]

    n_insert = sum(1 for r in rows if r["label"] == "insert")
    logger.info(f"skipped {n_skipped_sections} docs (non-standard sections)")
    logger.info(f"sentence-level dataset: {len(rows)} sentences ({n_insert} insert)")
    return pd.DataFrame(rows)


def build_paragraph_level_dataset(sent_df):
    para_rows = []

    for (pair_id, para_id), group in sent_df.groupby(
        ["pair_id", "para_id"], sort=False
    ):
        simple_sents = [s for y in group.simple for s in eval(y)]
        row = {
            "pair_id": pair_id,
            "para_id": para_id,
            "complex": " <s> ".join(group.complex),
            "simple": " ".join(simple_sents),
        }
        for col in group.columns:
            if col not in row:
                row[col] = list(group[col])
        para_rows.append(row)

    logger.info(f"paragraph-level dataset: {len(para_rows)} paragraphs")
    return pd.DataFrame(para_rows)


def build_document_level_dataset(sent_df, data_by_doi=None):
    doc_rows = []

    for pair_id, group in sent_df.groupby("pair_id", sort=False):
        simple_sents = [s for y in group.simple for s in eval(y)]
        row = {
            "pair_id": pair_id,
            "complex": " <s> ".join(group.complex),
            "simple": " ".join(simple_sents),
        }
        for col in group.columns:
            if col not in row:
                row[col] = list(group[col])

        if data_by_doi is not None:
            doi_matches = [d for d in data_by_doi if d.split(".")[2] == pair_id]
            if doi_matches:
                item = data_by_doi[doi_matches[0]]
                if item["pls_type"] == "long":
                    row["simple_full"] = item["pls"]
                else:
                    pls_texts = [
                        s
                        for sect in item["pls"]
                        for s in nltk.sent_tokenize(sect["text"])
                    ]
                    row["simple_full"] = " ".join(pls_texts)

        doc_rows.append(row)

    logger.info(f"document-level dataset: {len(doc_rows)} documents")
    return pd.DataFrame(doc_rows)


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


def compute_label_distribution(labels):
    labels = labels.replace("none", "merge")
    counts = labels.value_counts()
    total = counts.sum()
    return {
        label: {"count": int(count), "pct": round(100 * count / total, 2)}
        for label, count in counts.items()
    }


def compute_stats(sent_df):
    n_doc_pairs = sent_df["pair_id"].nunique()
    n_para_pairs = sent_df.groupby(["pair_id", "para_id"]).ngroups
    n_sent_pairs = len(sent_df)

    avg_ci = sent_df["complex"].apply(lambda x: len(str(x).split())).mean()

    sent_df["_simple_parsed"] = sent_df["simple"].apply(parse_simple)
    simple_lengths = []
    for sents in sent_df["_simple_parsed"]:
        for s in sents:
            if isinstance(s, str) and s.strip():
                simple_lengths.append(len(s.split()))
    avg_si = sum(simple_lengths) / len(simple_lengths) if simple_lengths else 0

    doc_n = sent_df.groupby("pair_id").size()
    avg_n = doc_n.mean()

    doc_k = sent_df.groupby("pair_id")["_simple_parsed"].apply(
        lambda col: sum(len(sents) for sents in col)
    )
    avg_k = doc_k.mean()

    para_sizes = sent_df.groupby(["pair_id", "para_id"]).size()
    avg_p = para_sizes.mean()

    sent_df.drop(columns=["_simple_parsed"], inplace=True)

    section_label_dist = {}
    for sec_name, group in sent_df.groupby("section_name", sort=False):
        section_label_dist[sec_name] = compute_label_distribution(group["label"])

    return {
        "n_doc_pairs": int(n_doc_pairs),
        "n_para_pairs": int(n_para_pairs),
        "n_sent_pairs": int(n_sent_pairs),
        "avg_ci": round(avg_ci, 2),
        "avg_si": round(avg_si, 2),
        "avg_n": round(avg_n, 2),
        "avg_k": round(avg_k, 2),
        "avg_p": round(avg_p, 2),
        "label_distribution": compute_label_distribution(sent_df["label"]),
        "label_distribution_per_section": section_label_dist,
    }


def main(args):
    data_by_doi = load_data(args.data_path)
    logger.info(f"loaded {len(data_by_doi)} documents from {args.data_path}")

    if args.manual_only:
        splits = {
            "train": ["train.json"],
            "val": ["val.json"],
            "test": ["test_manual.json"],
        }
    else:
        splits = {
            "train": ["auto.json", "train.json"],
            "val": ["val.json"],
            "test": ["test_manual.json"],
        }

    all_sent_dfs = []
    all_stats = {}

    for split_name, alignment_files in splits.items():
        logger.info(f"processing {split_name} split")

        alignment_dict = {}
        for fname in alignment_files:
            path = os.path.join(args.alignments_dir, fname)
            alignment_dict.update(load_alignments(path))

        logger.info(f"loaded {len(alignment_dict)} alignment entries for {split_name}")

        sent_df = build_sentence_level_dataset(data_by_doi, alignment_dict)
        para_df = build_paragraph_level_dataset(sent_df)
        doc_df = build_document_level_dataset(sent_df, data_by_doi)

        os.makedirs(args.output_dir, exist_ok=True)

        sent_path = os.path.join(
            args.output_dir, f"cochrane_sections_sents_{split_name}.csv"
        )
        para_path = os.path.join(
            args.output_dir, f"cochrane_sections_para_{split_name}.csv"
        )
        doc_path = os.path.join(
            args.output_dir, f"cochrane_sections_docs_{split_name}.csv"
        )

        sent_df.to_csv(sent_path, index=False)
        para_df.to_csv(para_path, index=False)
        doc_df.to_csv(doc_path, index=False)

        logger.info(f"saved {split_name}: {sent_path}, {para_path}, {doc_path}")

        split_stats = compute_stats(sent_df)
        all_stats[split_name] = split_stats
        all_sent_dfs.append(sent_df)

        logger.info(
            f"{split_name}: {split_stats['n_doc_pairs']} docs, "
            f"{split_stats['n_sent_pairs']} sents, "
            f"avg |c_i|={split_stats['avg_ci']}, avg |s_i|={split_stats['avg_si']}"
        )

    combined_df = pd.concat(all_sent_dfs, ignore_index=True)
    all_stats["all"] = compute_stats(combined_df)

    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"saved stats to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="preprocess cochrane-sections into sentence, paragraph and document-level datasets"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/cochrane-sections/data.json",
        help="path to data.json",
    )
    parser.add_argument(
        "--alignments_dir",
        type=str,
        default="data/cochrane-sections/alignments",
        help="path to directory containing alignment json files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cochrane-sections/cochrane-sections",
        help="directory to write output csv files",
    )
    parser.add_argument(
        "--manual_only",
        action="store_true",
        help="use only manual alignments for train (train.json only, no auto.json)",
    )
    args = parser.parse_args()
    main(args)
