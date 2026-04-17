from typing import List
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str
    config_name: str = None
    tokenizer_name: str = None
    cache_dir: str = None
    trust_remote_code: bool = False
    token: str = None

    # LoRA config
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ['v_proj', 'q_proj', 'k_proj', 'gate_proj', 'down_proj', 'o_proj', 'up_proj'])

    # Model config
    use_flash_attn: bool = False
    from_peft: str = None
    head_dim: int = 37


@dataclass
class DataArguments:
    data_fn: str = None
    cache_path: str = None
    max_len: int = 256
    fold: int = 0
    pad_to_multiple_of: int = 8

    add_correctness: bool = False
    knowledge_distillation: bool = False
    aug_data_training: bool = False


@dataclass
class MyTrainingArguments(TrainingArguments):
    kd_temperature: float = 1.0
    kd_alpha: List[float] = field(default_factory=lambda: [0.34, 0.33, 0.33])
    ablation_exp_name: str = None  # exp1/exp2/exp3/exp4/exp5
