"""
Inference script for generating predictions on test data
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from transformers import AutoTokenizer, HfArgumentParser
from tqdm import tqdm

from .load_model import get_model
from .modeling import MyModel


@dataclass
class InferenceArguments:
    """Arguments for inference"""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    lora_path: str = field(
        default=None,
        metadata={"help": "Path to LoRA checkpoint"}
    )
    data_fn: str = field(
        metadata={"help": "Path to test data CSV file"}
    )
    save_fn: str = field(
        default="submission.csv",
        metadata={"help": "Path to save predictions"}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for inference"}
    )
    max_len: int = field(
        default=256,
        metadata={"help": "Maximum sequence length"}
    )
    add_correctness: bool = field(
        default=False,
        metadata={"help": "Whether to add answer correctness to prompt"}
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Cache directory for models"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code when loading model"}
    )
    token: str = field(
        default=None,
        metadata={"help": "HuggingFace token"}
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to run inference on"}
    )


def format_text(row, add_correctness=False):
    """Format input text from dataframe row"""
    text = f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\n"
    if add_correctness and 'is_correct' in row:
        text += f"Answer Correctness: {'CORRECT' if row['is_correct'] == 1 else 'INCORRECT'}\n"
    text += f"Student Explanation: {row['StudentExplanation']}\n"
    return text


def load_model_and_tokenizer(args: InferenceArguments):
    """Load model and tokenizer"""
    print(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        token=args.token,
        cache_dir=args.cache_dir,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
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

    print(f"Loading model from {args.model_name_or_path}...")

    from .arguments import ModelArguments
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
        from_peft=args.lora_path,
    )

    base_model = get_model(model_args)

    model = MyModel(
        base_model,
        tokenizer=tokenizer,
        train_batch_size=args.batch_size,
    )

    model = model.to(args.device)
    model.eval()

    print(f"Model loaded successfully on {args.device}")
    return model, tokenizer


def predict(model, tokenizer, texts, args: InferenceArguments):
    """Run inference on a batch of texts"""
    encoded = tokenizer(
        texts,
        max_length=args.max_len,
        truncation=True,
        padding='longest',
        return_tensors='pt'
    )

    encoded = {k: v.to(args.device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model.encode(encoded)

    probs = torch.nn.functional.softmax(logits, dim=-1)

    return probs.cpu().numpy()


def main():
    parser = HfArgumentParser(InferenceArguments)
    args = parser.parse_args_into_dataclasses()[0]

    print("=" * 60)
    print("Inference Configuration")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"LoRA path: {args.lora_path}")
    print(f"Test data: {args.data_fn}")
    print(f"Output file: {args.save_fn}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(args)

    print(f"\nLoading test data from {args.data_fn}...")
    df = pd.read_csv(args.data_fn)
    print(f"Loaded {len(df)} samples")

    print("Formatting texts...")
    df['text'] = df.apply(lambda row: format_text(row, args.add_correctness), axis=1)
    texts = df['text'].tolist()

    print(f"\nRunning inference with batch size {args.batch_size}...")
    all_probs = []

    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch_texts = texts[i:i + args.batch_size]
        probs = predict(model, tokenizer, batch_texts, args)
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    print(f"Predictions shape: {all_probs.shape}")

    top3_indices = np.argsort(-all_probs, axis=1)[:, :3]

    print(f"\nPreparing submission file...")
    if 'QuestionId' in df.columns:
        submission = pd.DataFrame({
            'QuestionId': df['QuestionId'],
            'MisconceptionId': [' '.join(map(str, row)) for row in top3_indices]
        })
    else:
        submission = pd.DataFrame({
            'MisconceptionId': [' '.join(map(str, row)) for row in top3_indices]
        })

    output_path = Path(args.save_fn)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.save_fn, index=False)
    print(f"Saved predictions to {args.save_fn}")
    print(f"Submission shape: {submission.shape}")
    print("\nDone!")


if __name__ == "__main__":
    main()
