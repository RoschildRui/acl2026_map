import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from .arguments import ModelArguments

def get_model(model_args: ModelArguments):
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        token=model_args.token,
        cache_dir=model_args.cache_dir
    )
    config.use_cache = False

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2" if model_args.use_flash_attn else None,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        config=config,
        trust_remote_code=model_args.trust_remote_code,
        device_map=None
    )

    model.score = nn.Linear(config.hidden_size, model_args.head_dim)

    if model_args.from_peft:
        model = PeftModel.from_pretrained(model, model_args.from_peft, is_trainable=True)
        model.print_trainable_parameters()
    elif model_args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=model_args.lora_rank,
            target_modules=model_args.target_modules,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model