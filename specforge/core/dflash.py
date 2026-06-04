# coding=utf-8
"""DFlash Training Wrapper."""

from typing import Optional, TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from specforge.modeling.draft.dflash import DFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


def create_dflash_sdpa_mask(anchor_positions, block_keep_mask, S, block_size, device):
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)  # (1, 1, Q_LEN, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(
        1, 1, 1, -1
    )  # (1, 1, 1, KV_LEN)

    q_block_ids = q_indices // block_size

    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    final_mask = (mask_context | mask_draft) & valid_block
    return final_mask


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
):
    """Construct Flex Attention BlockMask for DFlash training.

    KV: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos).
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    return create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


def compute_topk_target_distillation_loss(
    student_logits: torch.Tensor,
    target_logits: torch.Tensor,
    target_logit_indices: torch.Tensor,
    weight_mask: torch.Tensor,
    top_k: int,
    temperature: float,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Top-k target-logit distillation for draft output logits.

    `target_logit_indices` points to target positions whose logits predict the
    draft labels. For causal LMs, token at position p is predicted by target
    logits at p - 1. If provided, `top_p` filters low-probability tokens inside
    the selected top-k candidate set and then renormalizes the target weights.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if student_logits.ndim != 3:
        raise ValueError(
            f"student_logits must have shape [batch, seq, vocab], got {tuple(student_logits.shape)}"
        )
    if target_logits.ndim != 3:
        raise ValueError(
            f"target_logits must have shape [batch, seq, vocab], got {tuple(target_logits.shape)}"
        )
    if target_logit_indices.shape != student_logits.shape[:2]:
        raise ValueError(
            "target_logit_indices must have shape matching student_logits[:2]"
        )
    if weight_mask.shape != student_logits.shape[:2]:
        raise ValueError("weight_mask must have shape matching student_logits[:2]")
    if student_logits.shape[0] != target_logits.shape[0]:
        raise ValueError("student and target batch sizes must match")
    if student_logits.shape[-1] != target_logits.shape[-1]:
        raise ValueError("student and target vocab sizes must match")

    batch_size, target_seq_len, vocab_size = target_logits.shape
    safe_indices = target_logit_indices.clamp(min=0, max=target_seq_len - 1)
    gathered_target_logits = torch.gather(
        target_logits.detach(),
        1,
        safe_indices.unsqueeze(-1).expand(batch_size, -1, vocab_size),
    )

    selected_k = min(top_k, vocab_size)
    target_top_logits, target_top_indices = torch.topk(
        gathered_target_logits.to(torch.float32) / temperature,
        selected_k,
        dim=-1,
    )
    target_top_probs = F.softmax(target_top_logits, dim=-1)
    if top_p is not None and top_p < 1.0:
        cumulative_probs = torch.cumsum(target_top_probs, dim=-1)
        top_p_mask = (cumulative_probs - target_top_probs) < top_p
        top_p_mask[..., 0] = True
        target_top_probs = target_top_probs * top_p_mask.to(target_top_probs.dtype)
        target_top_probs = target_top_probs / target_top_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

    student_log_probs = F.log_softmax(
        student_logits.to(torch.float32) / temperature, dim=-1
    )
    student_top_log_probs = torch.gather(student_log_probs, -1, target_top_indices)

    loss_per_token = -(target_top_probs * student_top_log_probs).sum(dim=-1)
    loss_per_token = loss_per_token * (temperature**2)
    weights = weight_mask.to(loss_per_token.dtype)
    return (loss_per_token * weights).sum() / (weights.sum() + 1e-6)


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with block-wise CE loss."""

    def __init__(
        self,
        draft_model: "DFlashDraftModel",
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        target_distillation_weight: float = 0.0,
        target_distillation_top_k: int = 3,
        target_distillation_top_p: Optional[float] = None,
        target_distillation_temperature: float = 1.0,
    ):
        super().__init__()
        if target_distillation_weight < 0:
            raise ValueError("target_distillation_weight must be non-negative")
        if target_distillation_top_k < 1:
            raise ValueError("target_distillation_top_k must be >= 1")
        if target_distillation_temperature <= 0:
            raise ValueError("target_distillation_temperature must be positive")
        if (
            target_distillation_top_p is not None
            and not 0.0 < target_distillation_top_p <= 1.0
        ):
            raise ValueError("target_distillation_top_p must be in (0, 1]")

        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.target_distillation_weight = target_distillation_weight
        self.target_distillation_top_k = target_distillation_top_k
        self.target_distillation_top_p = target_distillation_top_p
        self.target_distillation_temperature = target_distillation_temperature

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def prepare_noise_input(
        self, input_ids: torch.Tensor, block_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prepare noise input: first token of each block is real, rest are MASK."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if block_ids is not None:
            is_block_start = torch.ones(bsz, seq_len, dtype=torch.bool, device=device)
            is_block_start[:, 1:] = block_ids[:, 1:] != block_ids[:, :-1]
        else:
            positions = torch.arange(seq_len, device=device)
            is_block_start = (positions % self.block_size) == 0
            is_block_start = is_block_start.unsqueeze(0).expand(bsz, -1)

        noise_input_ids = torch.full_like(input_ids, self.mask_token_id)
        noise_input_ids[is_block_start] = input_ids[is_block_start]
        return noise_input_ids

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel block-wise training forward pass."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )

        logits = self.lm_head(output_hidden)

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        binary_eval_mask = weight_mask.view(-1)

        # --- Loss decay: exp(-(k-1)/γ) so k=1 (1st prediction) gets weight 1.0 ---
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -(k - 1).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        # --- Cross entropy ---
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        valid_token_count = flat_weights.sum() + 1e-6
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        if self.target_distillation_weight > 0:
            if target_logits is None:
                raise ValueError(
                    "target_logits must be provided when target_distillation_weight > 0"
                )
            target_logit_indices = (label_indices.view(bsz, -1) - 1).clamp(min=0)
            kd_loss = compute_topk_target_distillation_loss(
                student_logits=logits,
                target_logits=target_logits,
                target_logit_indices=target_logit_indices,
                weight_mask=weight_mask.view(bsz, -1),
                top_k=self.target_distillation_top_k,
                temperature=self.target_distillation_temperature,
                top_p=self.target_distillation_top_p,
            )
            loss = loss + self.target_distillation_weight * kd_loss

        # --- Accuracy ---
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy
