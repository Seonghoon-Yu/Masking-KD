"""Keep only the teacher rollouts whose final answer is correct."""

import argparse
import os
from glob import glob

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="./data/rollouts/qwen3_vl_8b_thinking")
    parser.add_argument("--output-dir", default="./data/rollouts/qwen3_vl_8b_thinking_correct")
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    total_before = total_after = 0
    for f in files:
        df = pd.read_parquet(f)
        df_correct = df[df["is_correct"]].reset_index(drop=True)
        df_correct.to_parquet(os.path.join(args.output_dir, os.path.basename(f)))

        total_before += len(df)
        total_after += len(df_correct)
        print(f"  {os.path.basename(f)}: {len(df)} -> {len(df_correct)}")

    print(f"Kept {total_after} / {total_before} rollouts ({total_after / total_before:.1%}) -> {args.output_dir}")


if __name__ == "__main__":
    main()
