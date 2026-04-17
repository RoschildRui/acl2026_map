# Cognitive-Uncertainty Guided Knowledge Distillation

Official implementation of "Cognitive-Uncertainty Guided Knowledge Distillation for Accurate Classification of Student Misconceptions"

## Core Method

Two-stage knowledge distillation framework with dual-layer marginal sample selection:

**Stage 1: Global Distillation**
- Standard knowledge distillation from teacher (Qwen-2.5-72B) to student (Qwen-3-4B)
- K-fold cross-validation to generate soft labels
- Loss: `L = α·CE + β·KD + γ·COS`

**Stage 2: Adaptive Refinement**
- **Sample Selection**: Identify high-value samples via cognitive uncertainty
  - **Near-miss (NM)**: Correct but uncertain (p⁽¹⁾-p⁽²⁾ ≤ δ) OR rank ∈ {2,3}
  - **Hard-hard (HH)**: Severely incorrect (rank > 3)
- **Difficulty Splitting**: Divide NM/HH into close/far by median distance
- **Adaptive Loss**: Dynamic CE/KD/COS weights based on sample difficulty

## Key Results

| Dataset | Metric | Value |
|---------|--------|-------|
| MAP-Charting | MAP@3 | **0.9585** (+17.8% vs baseline) |
| Algebra Misconception | Accuracy | **84.38%** (vs 67.73% SOTA LLM) |

## Code Structure

```
src/
├── __main__.py                    # Entry point for training
├── arguments.py                   # Training arguments and hyperparameters
├── modeling.py                    # Model with adaptive loss (exp2-exp5)
├── runner.py                      # Training pipeline and orchestration
├── trainer.py                     # Custom trainer implementation
├── load_model.py                  # Model and tokenizer initialization
├── dataset.py                     # Dataset loading and preprocessing
├── mk_post_training_dataset.py    # Stage 2 sample selection logic
└── submit.py                      # Inference script for predictions
```

## Usage
> **For other stage1/stage2 hyperparameter settings, please refer to the specific experimental setup in the paper.**
### Stage 1: Global Distillation

```bash
python -m src \
  --model_name_or_path Qwen/Qwen3-4B \
  --data_fn train_with_teacher_scores.csv \
  --knowledge_distillation True \
  --kd_temperature 1.0 \
  --kd_alpha 0.34 0.33 0.33 \
  --fold 0
```

### Stage 2: Sample Selection + Refinement

**Step 1**: Select high-value samples
```bash
python -m src.mk_post_training_dataset
```

**Step 2**: Adaptive refinement training
```bash
python -m src \
  --model_name_or_path Qwen/Qwen3-4B \
  --data_fn train_with_datatype.csv \
  --aug_data_training True \
  --ablation_exp_name exp5 \
  --from_peft stage1_checkpoint
```

### Inference

```bash
python -m src.submit \
  --model_name_or_path Qwen/Qwen3-4B \
  --lora_path stage2_checkpoint \
  --data_fn test.csv \
  --save_fn submission.csv \
  --batch_size 16
```

## Ablation Experiments

### MAP-Charting Dataset

| Method Variant | MAP@10 | MAP@3 | Accuracy |
|----------------|--------|-------|----------|
| **Full Method** | **0.9587** | **0.9585** | **0.9198** |
| w/o Adaptive Loss | 0.9542 | 0.9540 | 0.9123 |
| w/o Sample Selection | 0.9521 | 0.9519 | 0.9085 |
| w/o Stage-1 Distillation | 0.9548 | 0.9546 | 0.9132 |
| w/o Stage-2 Distillation | 0.9495 | 0.9493 | 0.9024 |

### Algebra Misconception Dataset

| Method Variant | MAP@10 | MAP@3 | Accuracy |
|----------------|--------|-------|----------|
| **Full Method** | **0.8915** | **0.8750** | **0.8438** |
| w/o Adaptive Loss | 0.8802 | 0.8657 | 0.8321 |
| w/o Sample Selection | 0.8741 | 0.8603 | 0.8269 |
| w/o Stage-1 Distillation | 0.8823 | 0.8679 | 0.8342 |
| w/o Stage-2 Distillation | 0.8001 | 0.7893 | 0.7577 |


## Key Hyperparameters

- `delta_threshold`: 0.05 (Near-miss confidence margin)
- K-fold: 5

## Dependencies

```
transformers>=4.51.3
torch>=2.6.0
peft>=0.14.0
scipy
pandas
scikit-learn
tqdm
```

## Citation

```bibtex
@inproceedings{liu2026cognitive,
  title={Cognitive-Uncertainty Guided Knowledge Distillation for Accurate Classification of Student Misconceptions},
  author={Liu, Qirui and Chen, Hao and Shi, Weijie and Xu, Jiajie and Zhu, Jia},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
  year={2026}
}
```
