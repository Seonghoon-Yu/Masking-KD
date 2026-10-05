# Hide-to-See: Reasoning-prefix Masking for Visual-anchored Thinking in VLM Distillation

Official implementation of **Masking-KD** (NeurIPS 2026), which distills Qwen3-VL-8B-Thinking into Qwen3-VL-2B-Thinking.

[📄 Paper](https://arxiv.org/abs/2605.11651) | [🤗 Model](https://huggingface.co/SeonghoonYu/Masking-KD) | [🤗 Training data](https://huggingface.co/datasets/SeonghoonYu/Masking-KD-Rollouts)

## Overview

In standard distillation, the student can predict the next reasoning token by leaning on the reasoning it has already written.
Masking-KD hides the parts of that **reasoning prefix** each token relies on most. To match the teacher, which still sees the full context, the student has to ground its prediction in the image and the question.

Each training step:

1. **Teacher** runs on the full context: image, question, and reasoning.
2. **Student attention**: for each response token, find the earlier response tokens it attends to most.
3. **KL-adaptive budget**: tokens the student already matches (low KL) get more masking, and harder tokens get less. The target attention mass to hide, τ, ranges from 0.3 to 0.5.
4. **Masked distillation**: mask those tokens with a 4D attention mask and distill the student toward the full-context teacher (reverse KL).

The image, the question, and the immediately preceding token are never masked.

## Installation

```bash
conda create -n masking_kd python=3.12 -y
conda activate masking_kd
pip install -r requirements.txt
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

If `import flash_attn` fails with `GLIBC_2.32 not found` (e.g. Ubuntu 20.04), build it from source instead. This requires CUDA 12.8 `nvcc`:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn==2.8.3 --no-build-isolation
```

## Data

**Training data.** Training uses greedy rollouts from the teacher, Qwen3-VL-8B-Thinking, on [ViRL39K](https://huggingface.co/datasets/PAPOGalaxy/PAPO_ViRL39K_train), keeping only rollouts with a correct final answer (19,387 samples). These are released as [SeonghoonYu/Masking-KD-Rollouts](https://huggingface.co/datasets/SeonghoonYu/Masking-KD-Rollouts), and `scripts/train.sh` downloads them automatically.

To regenerate them yourself (greedy decoding is not bit-exact across GPUs, so a few responses may differ from the released ones):

```bash
bash scripts/generate_rollouts.sh   # -> data/rollouts/qwen3_vl_8b_thinking_correct
```

**Evaluation data.** We use seven benchmarks from [PAPO-Eval](https://github.com/xhguo7/PAPO-Eval).

```bash
python evaluation/prepare_data.py   # -> data/papo/*.json, data/images/
```

## Training

```bash
bash scripts/train.sh
```

Our checkpoint was trained on 2× A100 80GB GPUs.

The script uses 2 GPUs by default. To change this, set `CUDA_VISIBLE_DEVICES`; gradient accumulation is adjusted to keep the global batch size at 512.

| Hyperparameter | Value | | Hyperparameter | Value |
|---|---|---|---|---|
| Epochs | 2 | | τ range (`tau_min`, `tau_max`) | 0.3, 0.5 |
| Global batch size | 512 | | Max mask ratio | 0.3 |
| Learning rate | 1e-6 | | KD temperature | 2.0 |
| Warmup ratio | 0.03 | | Max sequence length (longer samples are truncated) | 5000 |

## Evaluation

```bash
bash scripts/eval.sh SeonghoonYu/Masking-KD                    # released checkpoint
bash scripts/eval.sh checkpoints/masking_kd_qwen3_vl_2b_thinking  # your own run
```

The script runs greedy decoding (up to 4096 new tokens) on Geo3K, MathVista, We-Math, MMK12, MathVerse, LogicVista, and MMMU-Pro. It then prints the accuracy of the final `\boxed{}` answer for each benchmark and their average.

## Code structure

```
masking_kd/trainer.py    # Masking-KD training step (attention-based masking, KL-adaptive budget, loss)
masking_kd/data.py       # prompt template, image preprocessing, dataset, collator
train.py                 # training entrypoint
generate_rollouts.py     # teacher rollouts with vLLM
filter_rollouts.py       # keep correct rollouts
evaluation/              # benchmark download, vLLM inference, scoring
scripts/                 # rollout / train / eval launchers
```

## Acknowledgements

This code builds on [PAPO](https://github.com/MikeWangWZHL/PAPO) and [PAPO-Eval](https://github.com/xhguo7/PAPO-Eval) (training data and evaluation benchmarks), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [EasyR1](https://github.com/hiyouga/EasyR1), and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL).

## Citation

```bibtex
@inproceedings{yu2026hide,
  title     = {Hide to See: Reasoning-prefix Masking for Visual-anchored Thinking in VLM Distillation},
  author    = {Yu, Seonghoon and Nam, Dongjun and Lee, Byung-Kwan and Son, Jeany},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
