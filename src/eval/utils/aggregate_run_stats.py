import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_finite(x):
    return is_numeric(x) and not (isinstance(x, float) and np.isnan(x))


def aggregate_trees(trees, runs):
    ref = next((t for t in trees if t is not None), None)

    if isinstance(ref, dict):
        keys = []
        seen = set()
        for t in trees:
            if isinstance(t, dict):
                for k in t.keys():
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
        return {
            k: aggregate_trees(
                [t.get(k) if isinstance(t, dict) else None for t in trees], runs
            )
            for k in keys
        }

    if isinstance(ref, list):
        max_len = max((len(t) for t in trees if isinstance(t, list)), default=0)
        return [
            aggregate_trees(
                [t[i] if isinstance(t, list) and i < len(t) else None for t in trees],
                runs,
            )
            for i in range(max_len)
        ]

    if all(t is None or is_numeric(t) for t in trees):
        finite = [float(t) for t in trees if is_finite(t)]
        values = [float(t) if is_numeric(t) else None for t in trees]
        if not finite:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "values": values,
                "runs": runs,
            }
        arr = np.asarray(finite, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "values": values,
            "runs": runs,
        }

    return ref


def main(args):
    runs = [int(s.strip()) for s in args.runs.split(",") if s.strip()]
    input_paths = [Path(p.strip()) for p in args.inputs.split(",") if p.strip()]

    if len(runs) != len(input_paths):
        raise ValueError(
            f"number of runs ({len(runs)}) must match number of inputs ({len(input_paths)})"
        )

    trees = []
    for path in input_paths:
        logger.info(f"loading {path}")
        with open(path) as f:
            trees.append(json.load(f))

    stats = aggregate_trees(trees, runs)

    output = {
        "runs": runs,
        "inputs": [str(p) for p in input_paths],
        "stats": stats,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"wrote aggregated stats to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="aggregate per-run json metric files into mean/std/min/max"
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="comma-separated list of per-run json result files",
    )
    parser.add_argument(
        "--runs",
        required=True,
        help="comma-separated list of run ids, one per input file",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="destination json with aggregated stats",
    )
    args = parser.parse_args()
    main(args)
