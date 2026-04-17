import os
import torch
from transformers.trainer import Trainer


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(self.model, 'save'):
            self.model.save(output_dir)
        else:
            super()._save(output_dir, state_dict)

        if self.tokenizer and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))