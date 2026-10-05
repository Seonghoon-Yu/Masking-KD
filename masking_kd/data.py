"""Data loading shared by teacher rollout generation and distillation training.

Both stages must build prompts and preprocess images identically, so the prompt
template, the image resolution bounds, and the message builder all live here.
"""

import os
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, List

import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from jinja2 import Template
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .utils import process_image

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

PROMPT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "prompts", "math_perception.jinja")


def load_records(data_path: str, split: str = "train") -> List[Dict[str, Any]]:
    """Load parquet records from a local directory or a Hugging Face dataset repo."""
    if os.path.isdir(data_path):
        files = sorted(glob(os.path.join(data_path, "*.parquet")))
    else:
        files = _download_hf_parquet_files(data_path, split)

    if not files:
        raise ValueError(f"No parquet files found in {data_path}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} examples from {data_path}")
    return df.to_dict("records")


def _download_hf_parquet_files(repo_id: str, split: str) -> List[str]:
    repo_files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    parquet_files = sorted(f for f in repo_files if f.endswith(".parquet"))

    files = [f for f in parquet_files if f.startswith(f"{split}-") or f.startswith(f"data/{split}-")]
    if not files:
        files = [f for f in parquet_files if f.startswith("data/")] or parquet_files

    return [hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset") for f in files]


class PromptBuilder:
    """Turns a raw example into chat messages using the math reasoning prompt template."""

    def __init__(self, template_path: str = PROMPT_TEMPLATE_PATH):
        with open(template_path, encoding="utf-8") as f:
            self.template = Template(f.read().strip())

    def build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = self.template.render(content=example["problem"])

        if "images" not in example:
            return [{"role": "user", "content": prompt}]

        # Each "<image>" placeholder in the problem becomes an image slot.
        content = []
        for i, text in enumerate(prompt.split("<image>")):
            if i != 0:
                content.append({"type": "image"})
            if text:
                content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]


def load_images(example: Dict[str, Any]) -> list:
    return [process_image(image, MIN_PIXELS, MAX_PIXELS) for image in example.get("images", [])]


class RolloutDistillDataset(Dataset):
    """Teacher rollouts for off-policy distillation.

    Each item is the prompt (question + images) followed by the teacher response.
    Labels mask the prompt with -100 so that only response tokens are distilled.
    """

    def __init__(self, data_path: str, processor, split: str = "train"):
        self.processor = processor
        self.prompt_builder = PromptBuilder()
        self.records = load_records(data_path, split)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        example = self.records[index]
        messages = self.prompt_builder.build_messages(example)
        response = example["response"]

        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(
            messages + [{"role": "assistant", "content": response}], tokenize=False, add_generation_prompt=False
        )

        images = load_images(example)
        full_inputs = self.processor(images=images, text=[full_text], add_special_tokens=False, return_tensors="pt")
        prompt_inputs = self.processor(images=images, text=[prompt_text], add_special_tokens=False, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        input_ids = full_inputs["input_ids"][0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full_inputs["attention_mask"][0],
            "labels": labels,
            "pixel_values": full_inputs.get("pixel_values"),
            "image_grid_thw": full_inputs.get("image_grid_thw"),
        }


@dataclass
class Collator:
    """Right-pads token sequences and concatenates image patches across the batch."""

    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenizer = self.processor.tokenizer
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        batch = {
            "input_ids": pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0),
            "labels": pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100),
        }

        image_features = [f for f in features if f.get("pixel_values") is not None]
        if image_features:
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in image_features], dim=0)
            batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in image_features], dim=0)

        return batch
