import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel
from transformers.file_utils import ModelOutput
from typing import Dict, Optional, List, Union


class DistillationLoss(nn.Module):
    """KL divergence loss for knowledge distillation"""
    def forward(self, logits, teacher_probs, T=1.0):
        log_probs = F.log_softmax(logits / T, dim=-1)
        return -torch.mean(torch.sum(log_probs * teacher_probs, dim=-1))


class MyModel(PreTrainedModel):
    def __init__(self, base_model, tokenizer, T=1.0, kd_alpha=[0.2, 0.2, 0.2], ablation_exp_name=None):
        super().__init__(config=base_model.config)
        self.model = base_model
        self.tokenizer = tokenizer

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.03)
        self.kd_loss = DistillationLoss()
        self.cos_loss = nn.CosineEmbeddingLoss()

        self.T = T
        self.kd_alpha = kd_alpha
        self.ablation_exp_name = ablation_exp_name

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def encode(self, features):
        if features is None:
            return None
        outputs = self.model(
            input_ids=features['input_ids'],
            attention_mask=features['attention_mask'],
            position_ids=features.get('position_ids')
        )
        return outputs.logits

    def forward(self, text=None, label=None, teacher_score=None, data_type=None, **kwargs):
        logits = self.encode(text)

        # Apply ablation experiments in Stage 2
        if data_type is not None and data_type[0] is not None and self.ablation_exp_name and self.training:
            valid_mask = torch.tensor([dt != 'easy' for dt in data_type], dtype=torch.bool, device=logits.device)

            if valid_mask.sum() == 0:
                return ModelOutput(loss=logits.sum() * 0.0, logits=logits)

            loss = self._compute_ablation_loss(
                logits[valid_mask],
                label[valid_mask],
                [data_type[i] for i in range(len(data_type)) if valid_mask[i]],
                teacher_score[valid_mask] if teacher_score is not None else None
            )

        # Standard training (Stage 1)
        else:
            loss = self.ce_loss(logits, label)
            if teacher_score is not None:
                teacher_probs = F.softmax(teacher_score / self.T, dim=-1)
                kd_loss = self.kd_loss(logits, teacher_probs, self.T)
                cos_loss = self.cos_loss(
                    teacher_probs,
                    F.softmax(logits / self.T, dim=1),
                    torch.ones(logits.size(0), device=logits.device)
                )
                loss = self.kd_alpha[0] * loss + self.kd_alpha[1] * kd_loss + self.kd_alpha[2] * cos_loss

        return ModelOutput(loss=loss, logits=logits)

    def _compute_ablation_loss(self, logits, labels, data_types, teacher_score):
        """Adaptive loss based on sample difficulty"""
        if self.ablation_exp_name == 'exp5':  # Full method
            return self._exp5_loss(logits, labels, data_types, teacher_score)
        elif self.ablation_exp_name == 'exp4':
            return self._exp4_loss(logits, labels, data_types, teacher_score)
        elif self.ablation_exp_name == 'exp3':
            return self._exp3_loss(logits, labels, data_types, teacher_score)
        elif self.ablation_exp_name == 'exp2':
            return self._exp2_loss(logits, labels, data_types, teacher_score)
        else:  # exp1: CE only
            return self.ce_loss(logits, labels)

    def _exp5_loss(self, logits, labels, data_types, teacher_score):
        """
        NM_close: CE only
        NM_far: CE+KD+COS
        HH_close: KD+COS
        HH_far: CE+KD+COS
        """
        total_loss, count = 0.0, 0

        # NM_close: CE
        mask = torch.tensor([dt == 'NM_close' for dt in data_types], dtype=torch.bool, device=logits.device)
        if mask.sum() > 0:
            total_loss += self.ce_loss(logits[mask], labels[mask])
            count += 1

        # NM_far: CE+KD+COS
        mask = torch.tensor([dt == 'NM_far' for dt in data_types], dtype=torch.bool, device=logits.device)
        if mask.sum() > 0 and teacher_score is not None:
            total_loss += self._ce_kd_cos_loss(logits[mask], labels[mask], teacher_score[mask])
            count += 1

        # HH_close: KD+COS
        mask = torch.tensor([dt == 'HH_close' for dt in data_types], dtype=torch.bool, device=logits.device)
        if mask.sum() > 0 and teacher_score is not None:
            total_loss += self._kd_cos_loss(logits[mask], teacher_score[mask])
            count += 1

        # HH_far: CE+KD+COS
        mask = torch.tensor([dt == 'HH_far' for dt in data_types], dtype=torch.bool, device=logits.device)
        if mask.sum() > 0 and teacher_score is not None:
            total_loss += self._ce_kd_cos_loss(logits[mask], labels[mask], teacher_score[mask])
            count += 1

        return total_loss / count if count > 0 else logits.sum() * 0.0

    def _exp4_loss(self, logits, labels, data_types, teacher_score):
        """NM: CE+KD+COS, HH: KD+COS"""
        total_loss, count = 0.0, 0

        nm_mask = torch.tensor([dt in {'NM_close', 'NM_far'} for dt in data_types], dtype=torch.bool, device=logits.device)
        if nm_mask.sum() > 0 and teacher_score is not None:
            total_loss += self._ce_kd_cos_loss(logits[nm_mask], labels[nm_mask], teacher_score[nm_mask])
            count += 1

        hh_mask = torch.tensor([dt in {'HH_close', 'HH_far'} for dt in data_types], dtype=torch.bool, device=logits.device)
        if hh_mask.sum() > 0 and teacher_score is not None:
            total_loss += self._kd_cos_loss(logits[hh_mask], teacher_score[hh_mask])
            count += 1

        return total_loss / count if count > 0 else logits.sum() * 0.0

    def _exp3_loss(self, logits, labels, data_types, teacher_score):
        """NM: CE, HH: KD+COS"""
        total_loss, count = 0.0, 0

        nm_mask = torch.tensor([dt in {'NM_close', 'NM_far'} for dt in data_types], dtype=torch.bool, device=logits.device)
        if nm_mask.sum() > 0:
            total_loss += self.ce_loss(logits[nm_mask], labels[nm_mask])
            count += 1

        hh_mask = torch.tensor([dt in {'HH_close', 'HH_far'} for dt in data_types], dtype=torch.bool, device=logits.device)
        if hh_mask.sum() > 0 and teacher_score is not None:
            total_loss += self._kd_cos_loss(logits[hh_mask], teacher_score[hh_mask])
            count += 1

        return total_loss / count if count > 0 else logits.sum() * 0.0

    def _exp2_loss(self, logits, labels, data_types, teacher_score):
        """All selected samples: CE+KD+COS"""
        mask = torch.tensor([dt in {'NM_close', 'NM_far', 'HH_close', 'HH_far'} for dt in data_types],
                           dtype=torch.bool, device=logits.device)
        if mask.sum() > 0 and teacher_score is not None:
            return self._ce_kd_cos_loss(logits[mask], labels[mask], teacher_score[mask])
        return logits.sum() * 0.0

    def _ce_kd_cos_loss(self, logits, labels, teacher_score):
        """CE + KD + COS"""
        ce_loss = self.ce_loss(logits, labels)
        teacher_probs = F.softmax(teacher_score / self.T, dim=-1)
        kd_loss = self.kd_loss(logits, teacher_probs, self.T)
        cos_loss = self.cos_loss(
            teacher_probs,
            F.softmax(logits / self.T, dim=1),
            torch.ones(logits.size(0), device=logits.device)
        )
        return self.kd_alpha[0] * ce_loss + self.kd_alpha[1] * kd_loss + self.kd_alpha[2] * cos_loss

    def _kd_cos_loss(self, logits, teacher_score):
        """KD + COS"""
        teacher_probs = F.softmax(teacher_score / self.T, dim=-1)
        kd_loss = self.kd_loss(logits, teacher_probs, self.T)
        cos_loss = self.cos_loss(
            teacher_probs,
            F.softmax(logits / self.T, dim=1),
            torch.ones(logits.size(0), device=logits.device)
        )
        return 0.5 * kd_loss + 0.5 * cos_loss

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = {k: v.clone().cpu() for k, v in state_dict.items()}
        self.model.save_pretrained(output_dir, state_dict=state_dict)

    def save_pretrained(self, *args, **kwargs):
        self.tokenizer.save_pretrained(*args, **kwargs)
        return self.model.save_pretrained(*args, **kwargs)