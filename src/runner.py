import torch
from transformers import (
    AutoTokenizer, PreTrainedTokenizer, EvalPrediction
)

import os
import logging
from pathlib import Path
from typing import Tuple
from abc import ABC, abstractmethod
from transformers import set_seed, PreTrainedTokenizer

from .dataset import MyDataset, MyCollator

from .modeling import MyModel
from .arguments import ModelArguments, DataArguments, MyTrainingArguments
from .trainer import MyTrainer
from .load_model import get_model
from sklearn.metrics import log_loss, accuracy_score
import numpy as np

logger = logging.getLogger(__name__)


class MyRunner(ABC):
    def __init__(
            self,
            model_args: ModelArguments,
            data_args: DataArguments,
            training_args: MyTrainingArguments
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        if (
                os.path.exists(training_args.output_dir)
                and os.listdir(training_args.output_dir)
                and training_args.do_train
                and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

        set_seed(training_args.seed)

        self.tokenizer, self.model = self.load_tokenizer_and_model()
        
        if self.training_args.local_rank != -1:
            device = torch.device(f"cuda:{self.training_args.local_rank}")
            self.model = self.model.to(device)
            for param in self.model.parameters():
                if hasattr(param, 'to_tensor'):
                    param = param.to_tensor()

        self.train_dataset = MyDataset(
            args=self.data_args,
            tokenizer=self.tokenizer,
            is_train=True,
        )
        self.val_dataset = MyDataset(
            args=self.data_args,
            tokenizer=self.tokenizer,
            is_train=False,
        )
        self.data_collator = MyCollator(
            tokenizer=self.tokenizer,
            max_length=self.data_args.max_len,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            padding='longest',
            return_tensors="pt"
        )
        self.trainer = self.load_trainer()


    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, MyModel]:
        use_fast = True
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            token=self.model_args.token,
            cache_dir=self.model_args.cache_dir,
            use_fast=use_fast,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                if tokenizer.unk_token is None:
                    tokenizer.pad_token = '[PAD]'
                else:
                    tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        print("tokenizer.padding_side: ", tokenizer.padding_side)

        base_model = get_model(self.model_args)

        model = MyModel(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=self.training_args.per_device_train_batch_size,
            T=self.training_args.kd_temperature,
            kd_alpha=self.training_args.kd_alpha,
            ablation_exp_name=self.training_args.ablation_exp_name,
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        return tokenizer, model

    def compute_map3(self, eval_pred):
        """
        Computes multiple evaluation metrics including MAP@k, Accuracy, Recall@k, and Precision@k.
        """
        logits, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

        top10 = np.argsort(-probs, axis=1)[:, :10]
        n_samples = len(labels)
        map3, map5, map10 = 0.0, 0.0, 0.0
        recall3, recall5, recall10 = 0.0, 0.0, 0.0
        precision3, precision5, precision10 = 0.0, 0.0, 0.0
        acc = 0.0
        
        for i in range(n_samples):
            true_label = labels[i]
            
            # Accuracy: top-1 prediction equals true label
            if top10[i, 0] == true_label:
                acc += 1.0
            
            # Find position of true label in top-k predictions
            top3_preds = top10[i, :3]
            top5_preds = top10[i, :5]
            top10_preds = top10[i, :10]
            
            # MAP@k: 1/rank if true label in top-k, else 0
            if true_label in top3_preds:
                rank = np.where(top3_preds == true_label)[0][0] + 1
                map3 += 1.0 / rank
            
            if true_label in top5_preds:
                rank = np.where(top5_preds == true_label)[0][0] + 1
                map5 += 1.0 / rank
            
            if true_label in top10_preds:
                rank = np.where(top10_preds == true_label)[0][0] + 1
                map10 += 1.0 / rank
            
            # Recall@k: 1 if true label in top-k, else 0
            if true_label in top3_preds:
                recall3 += 1.0
            if true_label in top5_preds:
                recall5 += 1.0
            if true_label in top10_preds:
                recall10 += 1.0
            
            # Precision@k: number of correct predictions in top-k / k
            # For single-label classification, precision@k = 1/k if true label in top-k, else 0
            if true_label in top3_preds:
                precision3 += 1.0 / 3
            if true_label in top5_preds:
                precision5 += 1.0 / 5
            if true_label in top10_preds:
                precision10 += 1.0 / 10
        
        return {
            "map@3": map3 / n_samples,
            "map@5": map5 / n_samples,
            "map@10": map10 / n_samples,
            "accuracy": acc / n_samples,
            "recall@3": recall3 / n_samples,
            "recall@5": recall5 / n_samples,
            "recall@10": recall10 / n_samples,
            "precision@3": precision3 / n_samples,
            "precision@5": precision5 / n_samples,
            "precision@10": precision10 / n_samples,
        }

    def load_trainer(self) -> MyTrainer:
        self.training_args.metric_for_best_model="map@3"
        self.training_args.do_train = True
        self.training_args.do_eval = True
        self.training_args.eval_strategy = "steps"
        self.training_args.save_strategy = "steps"
        self.training_args.greater_is_better=True
        self.training_args.load_best_model_at_end = True

        if self.training_args.local_rank != -1:
            self.training_args.ddp_find_unused_parameters = False
            self.training_args.dataloader_pin_memory = False
        
        trainer = MyTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_map3,
        )
        return trainer

    def run(self):
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()