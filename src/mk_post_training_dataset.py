"""
Two-Tier Margin-Based Sample Selection
Identifies Near-miss (NM) and Hard-hard (HH) samples for Stage 2 training
"""
import pandas as pd
import numpy as np
from scipy.special import softmax


class SampleSelector:
    """Two-tier margin-based sample selector for knowledge distillation"""

    def __init__(self, delta_threshold=0.05):
        self.delta_threshold = delta_threshold

    def parse_logits(self, score_str):
        """Parse teacher_score string to logits"""
        return [float(x) for x in score_str[1:-1].split(',')]

    def get_rank(self, logits, true_label):
        """Get rank of true label in predictions (1-based)"""
        true_logit = logits[true_label]
        rank = sum(1 for logit in logits if logit > true_logit) + 1
        return rank

    def get_prob_diff(self, probs):
        """Get difference between top-1 and top-2 probabilities"""
        sorted_probs = sorted(probs, reverse=True)
        return sorted_probs[0] - sorted_probs[1]

    def calculate_distance(self, probs, true_label):
        """Calculate d(x_i, y_i) = |p_s(y_i|x_i) - max_j p_s(j|x_i)|"""
        true_prob = probs[true_label]
        max_prob = max(probs)
        return abs(true_prob - max_prob)

    def classify_samples(self, df):
        """
        Classify samples into categories:
        - NM (Near-miss): Correct but uncertain OR rank in {2,3}
        - HH (Hard-hard): rank > 3
        - Easy: Others
        """
        # Adjust label to 0-based indexing if needed
        if df['label'].min() >= 1:
            df['label'] = df['label'] - 1

        # Parse logits and calculate probabilities
        df['logits'] = df['teacher_score'].apply(self.parse_logits)
        df['probs'] = df['logits'].apply(softmax)

        # Calculate metrics
        df['rank'] = df.apply(lambda row: self.get_rank(row['logits'], row['label']), axis=1)
        df['pred_label'] = df['logits'].apply(np.argmax)
        df['prob_diff'] = df['probs'].apply(self.get_prob_diff)
        df['distance'] = df.apply(lambda row: self.calculate_distance(row['probs'], row['label']), axis=1)

        # Identify NM: (correct & low confidence) OR rank in {2,3}
        df['is_NM'] = ((df['pred_label'] == df['label']) & (df['prob_diff'] <= self.delta_threshold)) | \
                      (df['rank'].isin([2, 3]))

        # Identify HH: rank > 3 and not NM
        df['is_HH'] = (df['rank'] > 3) & (~df['is_NM'])

        return df

    def split_by_difficulty(self, df):
        """
        Split NM and HH samples into close/far based on median distance
        close: distance <= median (easier)
        far: distance > median (harder)
        """
        # Split NM samples
        nm_mask = df['is_NM']
        if nm_mask.sum() > 0:
            nm_median = df.loc[nm_mask, 'distance'].median()
            df.loc[nm_mask, 'data_type'] = df.loc[nm_mask, 'distance'].apply(
                lambda x: 'NM_close' if x <= nm_median else 'NM_far'
            )

        # Split HH samples
        hh_mask = df['is_HH']
        if hh_mask.sum() > 0:
            hh_median = df.loc[hh_mask, 'distance'].median()
            df.loc[hh_mask, 'data_type'] = df.loc[hh_mask, 'distance'].apply(
                lambda x: 'HH_close' if x <= hh_median else 'HH_far'
            )

        # Mark remaining as easy
        df.loc[~nm_mask & ~hh_mask, 'data_type'] = 'easy'

        return df

    def process(self, df):
        df = self.classify_samples(df)
        df = self.split_by_difficulty(df)

        print(f"Total samples: {len(df)}")
        print(f"NM samples: {df['is_NM'].sum()}")
        print(f"HH samples: {df['is_HH'].sum()}")
        print(f"\nData type distribution:")
        print(df['data_type'].value_counts().sort_index())

        df_final = df.drop(columns=['logits', 'probs', 'rank', 'pred_label',
                                     'prob_diff', 'is_NM', 'is_HH', 'distance'])
        return df_final


def main():
    csv_path = "q2572_37C_H37_37labeloof.csv"
    output_path = "q2572_37C_H37_37labeloof_with_datatype.csv"

    df = pd.read_csv(csv_path)

    selector = SampleSelector(delta_threshold=0.05)
    df_final = selector.process(df)
    df_final.to_csv(output_path, index=False)
    print(f"\nProcessed file saved to: {output_path}")

    print("\n" + "="*80)
    print("Sample examples from each data_type:")
    print("="*80)
    for dtype in ['NM_close', 'NM_far', 'HH_close', 'HH_far', 'easy']:
        if dtype in df_final['data_type'].values:
            print(f"\n{dtype}:")
            sample = df_final[df_final['data_type'] == dtype].head(2)
            for idx, row in sample.iterrows():
                print(f"  Row {idx}: label={row['label']}")


if __name__ == '__main__':
    main()
