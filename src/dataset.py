import torch
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorForSeq2Seq
from .arguments import DataArguments

class MyDataset(Dataset):
    def __init__(self, args: DataArguments, tokenizer: PreTrainedTokenizer, is_train=True):
        self.args = args
        df = pd.read_csv(args.data_fn)

        if 'fold' in df.columns:
            self.df = df[df['fold'] != args.fold if is_train else df['fold'] == args.fold].reset_index(drop=True)

        self.df['text'] = self.df.apply(self._format, axis=1)

        if args.knowledge_distillation:
            for col in ['teacher_score']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(lambda x: [float(v) for v in x[1:-1].split(',')])

    def _format(self, row):
        text = f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\n"
        if self.args.add_correctness and 'is_correct' in row:
            text += f"Answer Correctness: {'CORRECT' if row['is_correct'] == 1 else 'INCORRECT'}\n"
        text += f"Student Explanation: {row['StudentExplanation']}\n"
        return text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            row['text'],
            row['label'],
            row.get('teacher_score') if self.args.knowledge_distillation else None,
            row.get('data_type') if self.args.aug_data_training else None
        )


@dataclass
class MyCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors='pt'):
        texts = [f[0] for f in features]
        labels = torch.LongTensor([f[1] for f in features])

        teacher_score = [f[2] for f in features]
        teacher_score = torch.FloatTensor(teacher_score) if teacher_score[0] is not None else None

        data_type = [f[3] for f in features]

        collated = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=self.padding,
            return_tensors=return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        return {
            "text": collated,
            "label": labels,
            "teacher_score": teacher_score,
            "data_type": data_type,
        }
