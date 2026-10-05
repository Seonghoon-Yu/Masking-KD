"""Generate greedy teacher rollouts with vLLM.

Results are written as parquet shards; rerunning the script resumes after the last shard.
"""

import argparse
import os
from glob import glob

# Run the vLLM engine in-process; a forked engine process fails once CUDA is initialized.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from masking_kd.data import PromptBuilder, load_images, load_records
from masking_kd.utils import check_correctness


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--data-path", default="PAPOGalaxy/PAPO_ViRL39K_train")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="./data/rollouts/qwen3_vl_8b_thinking")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches-per-shard", type=int, default=100)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-images-per-prompt", type=int, default=10)
    return parser.parse_args()


def save_shard(rows, output_dir, shard_id):
    path = os.path.join(output_dir, f"results-{shard_id:05d}.parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"Saved {len(rows)} rollouts to {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.max_images_per_prompt},
        disable_log_stats=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, n=1)

    records = load_records(args.data_path, args.split)
    prompt_builder = PromptBuilder()

    # Resume after the samples already stored in existing shards.
    shards = sorted(glob(os.path.join(args.output_dir, "results-*.parquet")))
    shard_id = len(shards)
    start = sum(pq.ParquetFile(f).metadata.num_rows for f in shards)
    if start:
        print(f"Resuming from sample {start} ({shard_id} shards found)")

    rows, n_batches = [], 0
    for batch_start in tqdm(range(start, len(records), args.batch_size), desc="Teacher rollouts"):
        batch = records[batch_start : batch_start + args.batch_size]

        vllm_inputs = []
        for example in batch:
            prompt = processor.apply_chat_template(
                prompt_builder.build_messages(example), tokenize=False, add_generation_prompt=True
            )
            images = load_images(example)
            entry = {"prompt": prompt}
            if images:
                entry["multi_modal_data"] = {"image": images if len(images) > 1 else images[0]}
            vllm_inputs.append(entry)

        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

        for example, output in zip(batch, outputs, strict=True):
            text = output.outputs[0].text
            is_correct, parsed_answer = check_correctness(text, example["answer"])
            rows.append({
                "images": example["images"],
                "problem": example["problem"],
                "answer": example["answer"],
                "response": text,
                "is_correct": is_correct,
                "parsed_answer": parsed_answer,
            })

        n_batches += 1
        if n_batches % args.batches_per_shard == 0:
            save_shard(rows, args.output_dir, shard_id)
            rows, shard_id = [], shard_id + 1

    if rows:
        save_shard(rows, args.output_dir, shard_id)
    print("Done.")


if __name__ == "__main__":
    main()
