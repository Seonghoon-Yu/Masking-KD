"""Download the evaluation benchmarks from PAPO-Galaxy/PAPO_eval (https://github.com/xhguo7/PAPO-Eval).

Writes data/papo/<benchmark>.json and data/images/<benchmark>/..., the layout expected by
data/dataset_info.json and evaluation/infer.py.
"""

import argparse
import json
import os
import zipfile

from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO_ID = "PAPO-Galaxy/PAPO_eval"

# local name -> (split name in PAPO_eval, image archive)
BENCHMARKS = {
    "hiyouga_geometry3k": ("hiyouga_geometry3k", "hiyouga_geometry3k_images.zip"),
    "AI4Math_MathVista": ("AI4Math_MathVista", "AI4Math_MathVista_images.zip"),
    "We-Math_We-Math": ("We_Math", "We-Math_We-Math_images.zip"),
    "PAPO_MMK12": ("PAPO_MMK12", "PAPO_MMK12_test_images.zip"),
    "AI4Math_MathVerse": ("AI4Math_MathVerse", "AI4Math_MathVerse_images.zip"),
    "lscpku_LogicVista": ("lscpku_LogicVista", "lscpku_LogicVista_images.zip"),
    "MMMU_MMMU_Pro": ("MMMU_MMMU_Pro", "MMMU_MMMU_Pro_images.zip"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="./data")
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    args = parser.parse_args()

    json_dir = os.path.join(args.output_dir, "papo")
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    for name in args.benchmarks:
        split, archive = BENCHMARKS[name]
        print(f"[{name}] downloading split '{split}'")

        examples = [{k: v for k, v in ex.items() if k != "id"} for ex in load_dataset(REPO_ID, split=split)]
        json_path = os.path.join(json_dir, f"{name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"[{name}] saved {len(examples)} examples to {json_path}")

        zip_path = hf_hub_download(repo_id=REPO_ID, filename=archive, repo_type="dataset")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(image_dir)
        print(f"[{name}] extracted images to {image_dir}")


if __name__ == "__main__":
    main()
