import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


_DFLASH_PATH = Path(__file__).resolve().parents[2] / "specforge/core/dflash.py"
_SPEC = importlib.util.spec_from_file_location("_dflash_core_for_test", _DFLASH_PATH)
_DFLASH_CORE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_DFLASH_CORE)
compute_topk_target_distillation_loss = (
    _DFLASH_CORE.compute_topk_target_distillation_loss
)


class TestDFlashTargetDistillation(unittest.TestCase):

    def test_topk_target_distribution_matches_manual_loss(self):
        student_logits = torch.tensor(
            [
                [
                    [2.0, 0.0, -1.0, 1.0],
                    [-2.0, 3.0, 0.5, 0.0],
                ]
            ],
            requires_grad=True,
        )
        target_logits = torch.tensor(
            [
                [
                    [0.0, 3.0, 1.0, -2.0],
                    [4.0, -1.0, 0.0, 2.0],
                    [-3.0, 0.0, 5.0, 1.0],
                ]
            ],
            requires_grad=True,
        )
        target_logit_indices = torch.tensor([[0, 2]])
        weight_mask = torch.tensor([[1.0, 0.0]])

        loss = compute_topk_target_distillation_loss(
            student_logits=student_logits,
            target_logits=target_logits,
            target_logit_indices=target_logit_indices,
            weight_mask=weight_mask,
            top_k=2,
            temperature=1.0,
        )

        top_values, top_indices = torch.topk(target_logits.detach()[0, 0], 2)
        target_probs = F.softmax(top_values, dim=-1)
        student_log_probs = F.log_softmax(student_logits[0, 0], dim=-1)
        expected = -(target_probs * student_log_probs[top_indices]).sum()
        expected = expected / (weight_mask.sum() + 1e-6)

        torch.testing.assert_close(loss, expected)

        loss.backward()
        self.assertIsNotNone(student_logits.grad)
        self.assertIsNone(target_logits.grad)

    def test_temperature_scales_loss(self):
        student_logits = torch.tensor([[[2.0, 0.0, -1.0]]])
        target_logits = torch.tensor([[[0.0, 4.0, 2.0]]])
        target_logit_indices = torch.tensor([[0]])
        weight_mask = torch.tensor([[1.0]])
        temperature = 2.0

        loss = compute_topk_target_distillation_loss(
            student_logits=student_logits,
            target_logits=target_logits,
            target_logit_indices=target_logit_indices,
            weight_mask=weight_mask,
            top_k=3,
            temperature=temperature,
        )

        target_probs = F.softmax(target_logits[0, 0] / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits[0, 0] / temperature, dim=-1)
        expected = -(target_probs * student_log_probs).sum()
        expected = expected * (temperature**2) / (weight_mask.sum() + 1e-6)

        torch.testing.assert_close(loss, expected)

    def test_top_p_filters_low_probability_tail(self):
        student_logits = torch.tensor([[[0.0, 1.0, 2.0, -5.0]]])
        target_logits = torch.tensor([[[3.0, 2.0, 0.0, -8.0]]])
        target_logit_indices = torch.tensor([[0]])
        weight_mask = torch.tensor([[1.0]])

        loss = compute_topk_target_distillation_loss(
            student_logits=student_logits,
            target_logits=target_logits,
            target_logit_indices=target_logit_indices,
            weight_mask=weight_mask,
            top_k=3,
            temperature=1.0,
            top_p=0.8,
        )

        top_values, top_indices = torch.topk(target_logits[0, 0], 3)
        target_probs = F.softmax(top_values, dim=-1)
        keep = torch.tensor([True, True, False])
        target_probs = target_probs * keep.to(target_probs.dtype)
        target_probs = target_probs / target_probs.sum()
        student_log_probs = F.log_softmax(student_logits[0, 0], dim=-1)
        expected = -(target_probs * student_log_probs[top_indices]).sum()
        expected = expected / (weight_mask.sum() + 1e-6)

        torch.testing.assert_close(loss, expected)


if __name__ == "__main__":
    unittest.main()
