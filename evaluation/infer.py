# Adapted from LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory) and PAPO-Eval
# (https://github.com/xhguo7/PAPO-Eval). Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.
"""Run vLLM inference on a benchmark registered in data/dataset_info.json and save a JSONL file."""

import argparse
import json
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from transformers import AutoProcessor, Seq2SeqTrainingArguments
from vllm import LLM, SamplingParams

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_infer_args

SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, help="Benchmark name, e.g. hiyouga_geometry3k")
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--template", default="qwen2_vl")
    parser.add_argument("--cutoff-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--n-rollouts", type=int, default=1, help="Number of generations per question")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=20480)
    return parser.parse_args()


def load_benchmark(args, processor):
    """Load the raw benchmark JSON and the LLaMA-Factory dataset (which resolves the images)."""
    model_args, data_args, _, _ = get_infer_args(
        dict(
            model_name_or_path=args.model,
            dataset=args.dataset,
            dataset_dir=args.dataset_dir,
            template=args.template,
            cutoff_len=args.cutoff_len,
            trust_remote_code=True,
        )
    )
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    lf_dataset = get_dataset(
        template, model_args, data_args, Seq2SeqTrainingArguments(output_dir="dummy_dir"), "ppo",
        tokenizer=tokenizer, processor=processor,
    )["train_dataset"]

    with open(os.path.join(args.dataset_dir, "papo", f"{args.dataset}.json"), encoding="utf-8") as f:
        raw_dataset = json.load(f)

    assert len(lf_dataset) == len(raw_dataset), (
        f"LLaMA-Factory dataset ({len(lf_dataset)}) and raw JSON ({len(raw_dataset)}) are misaligned"
    )
    return raw_dataset, lf_dataset


def build_request(idx, item, sample, processor):
    messages = item.get("messages", [])
    user_content = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    if isinstance(user_content, list):
        prompt_text = "".join(part.get("text", "") for part in user_content if part.get("type") == "text")
    else:
        prompt_text = user_content
    label = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")

    images = sample.get("images")
    has_image = images is not None
    if has_image and not isinstance(images, list):
        images = [images]

    # Each "<image>" placeholder in the question becomes an image slot.
    content = []
    if has_image and "<image>" in prompt_text:
        for j, part in enumerate(prompt_text.split("<image>")):
            if j != 0:
                content.append({"type": "image"})
            if part.strip():
                content.append({"type": "text", "text": part})
    elif has_image:
        content.append({"type": "image"})
        if prompt_text:
            content.append({"type": "text", "text": prompt_text})
    else:
        content.append({"type": "text", "text": prompt_text})

    chat = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    return {
        "idx": idx,
        "id": item.get("id", f"test_{idx}"),
        "prompt": prompt,
        "label": label,
        "images": images or None,
    }


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    raw_dataset, lf_dataset = load_benchmark(args, processor)
    requests = [build_request(i, raw_dataset[i], lf_dataset[i], processor) for i in range(len(raw_dataset))]
    print(f"{args.dataset}: {len(requests)} questions")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 10, "video": 2},
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_new_tokens
    )
    vllm_inputs = [
        {"prompt": r["prompt"], **({"multi_modal_data": {"image": r["images"]}} if r["images"] else {})}
        for r in requests
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    results = []
    for rollout in range(args.n_rollouts):
        print(f"Rollout {rollout + 1}/{args.n_rollouts}")
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        for r, output in zip(requests, outputs, strict=True):
            results.append({
                "idx": r["idx"],
                "id": r["id"],
                "prompt": r["prompt"],
                "predict": output.outputs[0].text,
                "label": r["label"],
            })
        with open(args.output, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
