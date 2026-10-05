# Adapted from PAPO-Eval (https://github.com/xhguo7/PAPO-Eval). Licensed under the Apache License, Version 2.0.
"""Score benchmark predictions: accuracy of the final \\boxed{} answer, averaged over rollouts."""

import argparse
import json
import os
import re
from collections import defaultdict
from statistics import mean

from mathruler.grader import extract_boxed_content, grade_answer

BENCHMARKS = [
    ("hiyouga_geometry3k", "Geo3K"),
    ("AI4Math_MathVista", "MathVista"),
    ("We-Math_We-Math", "We-Math"),
    ("PAPO_MMK12", "MMK12"),
    ("AI4Math_MathVerse", "MathVerse"),
    ("lscpku_LogicVista", "LogicVista"),
    ("MMMU_MMMU_Pro", "MMMU-Pro"),
]


def boxed_accuracy(predict: str, ground_truth: str) -> float:
    predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
    return 1.0 if grade_answer(extract_boxed_content(predict), ground_truth) else 0.0


def score_file(path: str, n_rollouts: int) -> float:
    """Mean accuracy over questions, where each question's accuracy is averaged over its rollouts."""
    accs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            key = (ex.get("id", ""), ex.get("prompt", ""), ex.get("label", ""))
            accs[key].append(boxed_accuracy(ex.get("predict", ""), ex.get("label", "")))

    per_question = []
    for values in accs.values():
        assert len(values) % n_rollouts == 0, f"Expected a multiple of {n_rollouts} rollouts per question"
        per_question.extend([mean(values)] * (len(values) // n_rollouts))
    return mean(per_question) if per_question else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True, help="Directory with <benchmark>.jsonl prediction files")
    parser.add_argument("--n-rollouts", type=int, default=1)
    args = parser.parse_args()

    results = {}
    for name, display in BENCHMARKS:
        path = os.path.join(args.pred_dir, f"{name}.jsonl")
        if os.path.exists(path):
            results[display] = 100 * score_file(path, args.n_rollouts)
        else:
            print(f"Skipping {display}: {path} not found")
    if not results:
        raise FileNotFoundError(f"No prediction files found in {args.pred_dir}")
    if len(results) == len(BENCHMARKS):
        results["Avg."] = mean(results.values())

    width = max(len(k) for k in results)
    print(f"\nAccuracy (%) @ {args.n_rollouts} rollout(s): {args.pred_dir}")
    for k, v in results.items():
        print(f"  {k:<{width}}  {v:6.2f}")

    out_path = os.path.join(args.pred_dir, "scores.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
