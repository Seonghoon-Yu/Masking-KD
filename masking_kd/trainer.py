"""Masking-KD trainer: reasoning-prefix masking with KL-adaptive mask budgets.

For every training sample (prompt + teacher response) one step runs:

  1. Teacher forward on the full context (no grad).
  2. Student forward with eager attention (no grad) to collect its attention map,
     averaged over heads and layers. Attention is accumulated layer by layer with
     forward hooks, so only one [seq, seq] matrix is kept in GPU memory.
  3. Per response token, the student-teacher KL sets a target attention ratio tau:
     tokens the student already matches (low KL) get a larger tau (more masking),
     tokens it struggles with (high KL) get a smaller tau.
  4. For each response query, the previous response tokens it attends to most are
     masked (top-k by attention) until their attention mass reaches tau, under a
     budget of max_mask_ratio. The immediately preceding token is never masked,
     and the prompt (image + question) is never masked.
  5. Student forward with this 4D mask (with grad) and a reverse-KL distillation
     loss against the full-context teacher.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Trainer


@dataclass
class MaskingKDArguments:
    distill_temperature: float = field(default=2.0, metadata={"help": "Softmax temperature for the KL loss."})
    tau_min: float = field(default=0.3, metadata={"help": "Target attention ratio for the highest-KL tokens."})
    tau_max: float = field(default=0.5, metadata={"help": "Target attention ratio for the lowest-KL tokens."})
    sigmoid_scale: float = field(default=1.0, metadata={"help": "Scale s in sigmoid((u - mean(u)) / s), u = -log KL."})
    max_mask_ratio: float = field(
        default=0.3, metadata={"help": "Maximum fraction of eligible response tokens masked per query."}
    )
    min_mask_k: int = field(default=1, metadata={"help": "Minimum number of tokens masked per query."})
    fallback_mask_ratio: float = field(
        default=0.1, metadata={"help": "Mask ratio used for queries whose attention to the response is low."}
    )
    resp_attn_min: float = field(
        default=0.10, metadata={"help": "Queries with response attention mass below this use fallback_mask_ratio."}
    )
    max_seq_len: int = field(default=5000, metadata={"help": "Sequences longer than this are truncated."})
    kl_chunk_size: int = field(default=512, metadata={"help": "Chunk size for KL computation over the vocabulary."})


class _AttentionAccumulator:
    """Forward hook that sums head-averaged attention maps over layers.

    The attention weights are dropped from the layer output right after they are
    accumulated, so at most one layer's attention lives in GPU memory.
    """

    def __init__(self):
        self.attn_sum = None
        self.n_layers = 0

    def hook_fn(self, module, inputs, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn = output[1][0].mean(dim=0).float()  # [batch, heads, seq, seq] -> [seq, seq]
            self.attn_sum = attn if self.attn_sum is None else self.attn_sum + attn
            self.n_layers += 1
            return (output[0], None) + output[2:]
        return output

    def get_mean(self):
        if self.n_layers == 0:
            raise ValueError("No attention layers were captured by the hooks.")
        return self.attn_sum / self.n_layers


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class MaskingKDTrainer(Trainer):
    def __init__(self, teacher_model, distill_args: MaskingKDArguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_args = distill_args
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _get_position_ids(model, inputs):
        """Multimodal RoPE position ids, computed once and shared by all forwards."""
        inner = getattr(_unwrap(model), "model", _unwrap(model))
        if hasattr(inner, "get_rope_index"):
            with torch.no_grad():
                position_ids, _ = inner.get_rope_index(
                    inputs["input_ids"],
                    inputs.get("image_grid_thw", None),
                    inputs.get("video_grid_thw", None),
                    inputs["attention_mask"],
                )
            return position_ids
        return (inputs["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)

    @staticmethod
    def _get_attention_modules(model):
        inner = getattr(model, "model", model)
        if hasattr(inner, "language_model"):
            layers = inner.language_model.layers
        elif hasattr(inner, "layers"):
            layers = inner.layers
        else:
            raise AttributeError("Cannot find transformer layers")
        attn_modules = [layer.self_attn for layer in layers if hasattr(layer, "self_attn")]
        if not attn_modules:
            raise AttributeError("Cannot find self_attn in transformer layers")
        return attn_modules

    @torch.no_grad()
    def _student_attention_map(self, model, inputs, position_ids):
        """Run the student with eager attention and return (logits, layer-averaged attention)."""
        model.eval()
        attn_modules = self._get_attention_modules(model)

        # SDPA does not return attention weights, so switch to eager attention for this forward only.
        prev_attn_impl = model.config._attn_implementation
        model.config._attn_implementation = "eager"

        accumulator = _AttentionAccumulator()
        hooks = [m.register_forward_hook(accumulator.hook_fn) for m in attn_modules]

        forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        if position_ids is not None:
            forward_inputs["position_ids"] = position_ids
        outputs = model(**forward_inputs, output_attentions=True, use_cache=False, return_dict=True)

        for h in hooks:
            h.remove()
        model.config._attn_implementation = prev_attn_impl
        model.train()

        return outputs.logits, accumulator.get_mean()

    @staticmethod
    def _compute_tau(s_logits, t_logits, T, tau_min, tau_max, sigmoid_scale, chunk_size):
        """Per-query target attention ratio from the per-token reverse KL.

        tau = tau_min + (tau_max - tau_min) * sigmoid((u - mean(u)) / s),  u = -log(KL)
        """
        R = s_logits.shape[0]
        if R <= 0:
            return torch.full((0,), (tau_min + tau_max) / 2, device=s_logits.device), {}

        per_token_kl = []
        for i in range(0, R, chunk_size):
            s_c = s_logits[i : i + chunk_size].float()
            t_c = t_logits[i : i + chunk_size].float()
            s_log = F.log_softmax(s_c / T, dim=-1)
            t_log = F.log_softmax(t_c / T, dim=-1)
            s_p = F.softmax(s_c / T, dim=-1)
            per_token_kl.append((s_p * (s_log - t_log)).sum(dim=-1) * (T**2))
            del s_c, t_c, s_log, t_log, s_p
        per_token_kl = torch.cat(per_token_kl).clamp(min=0.0)

        u = -torch.log(per_token_kl + 1e-6)
        tau = tau_min + (tau_max - tau_min) * torch.sigmoid((u - u.mean()) / sigmoid_scale)

        stats = {
            "kl": per_token_kl.mean().item(),
            "tau_mean": tau.mean().item(),
            "tau_std": tau.std().item(),
            "tau_min": tau.min().item(),
            "tau_max": tau.max().item(),
        }
        return tau, stats

    def _build_mask(self, attn_mean, seq_len, valid_len, prompt_len, resp_indices, tau, device, dtype):
        """Causal 4D mask with additional per-query masking of high-attention response tokens."""
        args = self.distill_args
        min_val = torch.finfo(dtype).min
        stats = {"k_mean": 0.0, "k_min": 0, "k_max": 0, "mask_ratio": 0.0, "attn_ratio": 0.0, "n_fallback": 0}

        mask_4d = torch.where(
            torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
            torch.tensor(min_val, device=device, dtype=dtype),
            torch.tensor(0.0, device=device, dtype=dtype),
        ).unsqueeze(0).unsqueeze(0)
        if valid_len < seq_len:
            mask_4d[0, 0, :, valid_len:] = min_val
            mask_4d[0, 0, valid_len:, :] = min_val

        R = valid_len - prompt_len
        if len(resp_indices) == 0 or R <= 0:
            return mask_4d, stats

        resp_t = torch.tensor(resp_indices, device=device, dtype=torch.long)
        q_pos = torch.arange(prompt_len, valid_len, device=device)

        # Eligible keys: earlier response tokens, excluding the immediately preceding one.
        eligible = resp_t.unsqueeze(0) < q_pos.unsqueeze(1)
        eligible &= ~(resp_t.unsqueeze(0) == (q_pos.unsqueeze(1) - 1))
        n_eligible = eligible.long().sum(dim=1)
        max_k_per_query = (n_eligible.float() * args.max_mask_ratio).int()

        attn_scores = attn_mean[q_pos.unsqueeze(1), resp_t.unsqueeze(0)]
        attn_scores_masked = attn_scores.clone()
        attn_scores_masked[~eligible] = -1.0
        resp_attn_sum = (attn_scores * eligible.float()).sum(dim=1).clamp(min=1e-8)

        sorted_scores, sorted_indices = attn_scores_masked.sort(dim=1, descending=True)
        max_k = max_k_per_query.max().item()
        if max_k == 0:
            return mask_4d, stats
        sorted_scores = sorted_scores[:, :max_k]
        sorted_indices = sorted_indices[:, :max_k]

        # Smallest k whose cumulative attention (relative to the query's response attention) reaches tau.
        k_range = torch.arange(max_k, device=device).unsqueeze(0)
        within_budget = k_range < max_k_per_query.unsqueeze(1)
        cumsum_ratio = sorted_scores.masked_fill(~within_budget, 0.0).cumsum(dim=1) / resp_attn_sum.unsqueeze(1)

        reached = cumsum_ratio >= tau.unsqueeze(1)
        k_per_query = reached.float().argmax(dim=1) + 1
        not_reached = ~reached.any(dim=1)
        k_per_query[not_reached] = max_k_per_query[not_reached].long()
        k_per_query = k_per_query.int().clamp(min=args.min_mask_k)

        # Queries that barely attend to the response use a fixed fallback ratio.
        low_resp = resp_attn_sum < args.resp_attn_min
        if low_resp.any():
            k_per_query[low_resp] = (n_eligible[low_resp].float() * args.fallback_mask_ratio).int().clamp(
                min=args.min_mask_k
            )
        k_per_query = torch.min(k_per_query, max_k_per_query)

        within_k = (k_range < k_per_query.unsqueeze(1)) & within_budget
        sel_q = q_pos.unsqueeze(1).expand_as(sorted_indices)[within_k]
        sel_k = resp_t[sorted_indices[within_k]]
        mask_4d[0, 0, sel_q, sel_k] = min_val

        valid = n_eligible > 0
        if valid.any():
            k = k_per_query.float()
            attn_ratio = cumsum_ratio[torch.arange(R, device=device), (k_per_query - 1).clamp(min=0)]
            stats = {
                "k_mean": k[valid].mean().item(),
                "k_min": int(k[valid].min().item()),
                "k_max": int(k[valid].max().item()),
                "mask_ratio": (k / n_eligible.float().clamp(min=1))[valid].mean().item(),
                "attn_ratio": attn_ratio[valid].mean().item(),
                "n_fallback": int(low_resp.sum().item()),
            }
        return mask_4d, stats

    def _truncate(self, inputs, position_ids, max_len):
        for key in ("input_ids", "attention_mask", "labels"):
            inputs[key] = inputs[key][:, :max_len]
        if position_ids is not None:
            position_ids = position_ids[..., :max_len]
        return inputs, position_ids

    def _dummy_step(self, model, inputs):
        """Zero-loss step for samples with no response tokens, keeping all ranks in sync."""
        dummy_len = min(16, inputs["input_ids"].shape[1])
        dummy_ids = inputs["input_ids"][:, :dummy_len].clone()
        config = _unwrap(model).config
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is not None:
            dummy_ids[dummy_ids == image_token_id] = getattr(config, "pad_token_id", 0) or 0
        dummy_inputs = {"input_ids": dummy_ids, "attention_mask": inputs["attention_mask"][:, :dummy_len]}

        with torch.no_grad():
            _unwrap(model)(**dummy_inputs, use_cache=False)
        loss = model(**dummy_inputs, use_cache=False).logits.sum() * 0.0
        self.accelerator.backward(loss)
        return loss.detach()

    # ── Training step ───────────────────────────────────────────────────────
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        device = next(model.parameters()).device
        if next(self.teacher_model.parameters()).device != device:
            self.teacher_model.to(device)

        args = self.distill_args
        T = args.distill_temperature
        dtype = torch.bfloat16
        position_ids = self._get_position_ids(model, inputs)

        valid_len = int(inputs["attention_mask"][0].long().sum().item())
        if valid_len > args.max_seq_len:
            inputs, position_ids = self._truncate(inputs, position_ids, args.max_seq_len)
            valid_len = args.max_seq_len
        seq_len = inputs["labels"].shape[1]

        resp_indices = (inputs["labels"][0] != -100).nonzero(as_tuple=True)[0].cpu().numpy()
        if len(resp_indices) == 0:
            return self._dummy_step(model, inputs)
        prompt_len = int(resp_indices[0])

        # Logits at position i predict token i + 1.
        resp_start, resp_end = max(prompt_len - 1, 0), valid_len - 1

        # 1. Teacher forward on the full context.
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        if position_ids is not None:
            teacher_inputs["position_ids"] = position_ids
        with torch.no_grad():
            t_logits = self.teacher_model(**teacher_inputs, use_cache=False).logits
        t_logits_query = t_logits[0, prompt_len:valid_len].to(dtype).contiguous()
        t_logits_target = t_logits[0, resp_start:resp_end].to(dtype).contiguous()
        del t_logits, teacher_inputs

        # 2. Student attention map (no grad).
        s_logits, attn_mean = self._student_attention_map(_unwrap(model), inputs, position_ids)

        # 3. KL-adaptive target attention ratio per response query.
        tau, kl_stats = self._compute_tau(
            s_logits[0, prompt_len:valid_len], t_logits_query,
            T, args.tau_min, args.tau_max, args.sigmoid_scale, args.kl_chunk_size,
        )
        del s_logits, t_logits_query
        torch.cuda.empty_cache()

        # 4. Mask the response tokens each query attends to most.
        mask_4d, mask_stats = self._build_mask(
            attn_mean, seq_len, valid_len, prompt_len, resp_indices, tau, device, dtype
        )
        del attn_mean, tau

        if self.is_world_process_zero() and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            print(
                f"  [step {self.state.global_step}] "
                f"k={mask_stats['k_mean']:.1f} [{mask_stats['k_min']}-{mask_stats['k_max']}], "
                f"mask_ratio={mask_stats['mask_ratio']:.4f}, attn_ratio={mask_stats['attn_ratio']:.4f}, "
                f"tau={kl_stats.get('tau_mean', 0):.4f}±{kl_stats.get('tau_std', 0):.4f} "
                f"[{kl_stats.get('tau_min', 0):.3f}-{kl_stats.get('tau_max', 0):.3f}], "
                f"kl={kl_stats.get('kl', 0):.4f}, fallback={mask_stats['n_fallback']}"
            )

        # 5. Masked student forward and distillation loss.
        student_inputs = {k: v for k, v in inputs.items() if k not in ("labels", "input_ids")}
        student_inputs["attention_mask"] = mask_4d
        if position_ids is not None:
            student_inputs["position_ids"] = position_ids
        # Pass embeddings instead of ids so gradient checkpointing sees an input that requires grad.
        student_inputs["inputs_embeds"] = _unwrap(model).get_input_embeddings()(inputs["input_ids"])

        with self.compute_loss_context_manager():
            # The 4D float mask rules out FlashAttention; let SDPA pick among the remaining kernels.
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]):
                s_logits_target = model(**student_inputs, use_cache=False).logits[0, resp_start:resp_end]
            del student_inputs, mask_4d

            n_tokens = s_logits_target.shape[0]
            if n_tokens > 0:
                kl_sum = torch.tensor(0.0, device=device)
                for i in range(0, n_tokens, args.kl_chunk_size):
                    s_c = s_logits_target[i : i + args.kl_chunk_size].float()
                    t_c = t_logits_target[i : i + args.kl_chunk_size].float()
                    s_log = F.log_softmax(s_c / T, dim=-1)
                    t_log = F.log_softmax(t_c / T, dim=-1)
                    s_p = F.softmax(s_c / T, dim=-1)
                    kl_sum = kl_sum + (s_p * (s_log - t_log)).sum(dim=-1).sum()
                    del s_c, t_c, s_log, t_log, s_p
                loss = kl_sum / n_tokens * (T**2)
            else:
                loss = s_logits_target.sum() * 0.0
            del t_logits_target, s_logits_target

        if self.is_world_process_zero() and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            print(f"  [step {self.state.global_step}] loss_kd={loss.item():.4f}")

        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps
