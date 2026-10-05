"""Masking-KD training entrypoint. Launch with scripts/train.sh."""

import pathlib
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, HfArgumentParser, TrainingArguments

from masking_kd.data import Collator, RolloutDistillDataset
from masking_kd.trainer import MaskingKDArguments, MaskingKDTrainer


@dataclass
class ModelArguments:
    student_model_name_or_path: str = field(default="Qwen/Qwen3-VL-2B-Thinking")
    teacher_model_name_or_path: str = field(default="Qwen/Qwen3-VL-8B-Thinking")
    freeze_vision_tower: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(
        default="SeonghoonYu/Masking-KD-Rollouts",
        metadata={"help": "Hugging Face dataset repo or local directory of teacher rollout parquet files."},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, MaskingKDArguments))
    model_args, data_args, training_args, distill_args = parser.parse_args_into_dataclasses()

    # The student uses SDPA so that the custom 4D mask is applied; the teacher always sees the full context.
    student = AutoModelForImageTextToText.from_pretrained(
        model_args.student_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    teacher = AutoModelForImageTextToText.from_pretrained(
        model_args.teacher_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(model_args.student_model_name_or_path)

    if model_args.freeze_vision_tower:
        student.visual.requires_grad_(False)
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"Trainable params: {n_trainable} / {n_total} ({n_trainable / n_total:.2%})")

    if training_args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        student.config.use_cache = False

    trainer = MaskingKDTrainer(
        model=student,
        teacher_model=teacher,
        distill_args=distill_args,
        args=training_args,
        train_dataset=RolloutDistillDataset(data_args.data_path, processor),
        data_collator=Collator(processor),
        processing_class=processor,
    )

    resume = any(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
