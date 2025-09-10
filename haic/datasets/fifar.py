import kagglehub
import numpy as np
import pandas as pd
import torch
import os


class DistributionModel:
    def build(self, value_votes):
        self.value_probs = value_votes.mean(axis=1)

    def get(self, idx):
        return self.value_probs[idx]


class MajorityModel:
    def build(self, value_votes):
        self.values = (value_votes.sum(axis=1) >= (value_votes.shape[1] / 2)).astype(int)

    def get(self, idx):
        return self.values[idx]


class SamplingModel:
    def __init__(self, random_state=None):
        self.rng = np.random.default_rng(seed=random_state)

    def build(self, value_votes):
        self.values = [
            self.rng.choice([0, 1], p=[1 - probs.mean(), probs.mean()])
            for probs in value_votes
        ]

    def get(self, idx):
        return self.values[idx]


class FiFAR(torch.utils.data.Dataset):
    """
    FiFAR (Financial Fraud Alert Review) Dataset loader.

    This class loads and processes the FiFAR dataset from Kaggle:
    https://www.kaggle.com/datasets/leonardovalves/fifar-financial-fraud-alert-review-dataset
    Supports both the 'train' and 'test' splits, including options to include human expert labels and model predictions.

    Args:
        split (str): Which dataset split to load: either 'train' or 'test'.

        include_human_labels (bool): Whether to include human labels.
            - For the 'train' split: each sample has a single label from one human expert.
            - For the 'test' split: each sample has 50 expert annotations. These are aggregated
              using the method defined by `human_model`.

        human_model (str): Aggregation method for multiple human labels in the 'test' split.
            Options are:
            - 'majority': returns the majority vote (binary label).
            - 'distribution': returns the proportion of experts voting for class 1 (soft label).
            - 'sample': randomly samples a label based on the empirical distribution (randomized label).
            Ignored for the 'train' split.

        random_state (int): Seed for reproducible sampling in 'sample' mode. Only used if
            `human_model='sample'`.

        include_model_preds (bool): If True, includes model predictions in the dataset.
    """

    def __init__(self,
                 split='train',
                 include_human_labels: bool = True,
                 human_model: str = 'majority',
                 random_state: int = 42,
                 include_model_preds: bool = False
                 ):

        self.split = split
        self.include_human_labels = include_human_labels
        self.include_model_preds = include_model_preds

        path = kagglehub.dataset_download("leonardovalves/fifar-financial-fraud-alert-review-dataset")

        numeric_features = [
            'income', 'name_email_similarity', 'prev_address_months_count',
            'current_address_months_count', 'customer_age', 'days_since_request',
            'intended_balcon_amount', 'zip_count_4w', 'velocity_6h',
            'velocity_24h', 'velocity_4w', 'bank_branch_count_8w',
            'date_of_birth_distinct_emails_4w', 'credit_risk_score',
            'bank_months_count', 'proposed_credit_limit',
            'session_length_in_minutes', 'device_distinct_emails_8w',
            'device_fraud_count', 'month'
        ]

        binary_features = [
            'email_is_free', 'phone_home_valid', 'phone_mobile_valid',
            'has_other_cards', 'foreign_request', 'keep_alive_session'
        ]

        categorical_features = [
            'payment_type', 'employment_status', 'housing_status',
            'source', 'device_os'
        ]

        self.features = numeric_features + binary_features + categorical_features
        self.target = 'fraud_label' if split == 'test' else 'fraud_bool'

        if split == 'train':
            train_df = pd.read_csv(
                os.path.join(path, 'ICAIF_KAGGLE', 'testbed', 'train', 'small__regular', 'train.csv'))
            self.data = train_df[train_df['assignment'] != 'model#0'].reset_index(drop=True)
            self.human_labels = self.data['decision'].astype(int).tolist()
        elif split == 'test':
            test_df = pd.read_csv(os.path.join(path, 'ICAIF_KAGGLE', 'testbed', 'test', 'test.csv'))

            if include_human_labels:
                test_df_human = pd.read_csv(
                    os.path.join(path, 'ICAIF_KAGGLE', 'testbed', 'test', 'test_expert_pred.csv'))

                human_label_cols = (
                        [f'standard#{i}' for i in range(20)] +
                        [f'model_agreeing#{i}' for i in range(10)] +
                        [f'unfair#{i}' for i in range(10)] +
                        [f'sparse#{i}' for i in range(10)]
                )

                test_df = pd.merge(test_df, test_df_human[['case_id'] + human_label_cols], on='case_id')
                self.data = test_df.reset_index(drop=True)

                value_votes = self.data[human_label_cols].astype(int).values

                if human_model == 'distribution':
                    self.human_model = DistributionModel()
                elif human_model == 'majority':
                    self.human_model = MajorityModel()
                elif human_model == 'sample':
                    self.human_model = SamplingModel(random_state)
                else:
                    raise ValueError(f"Unknown human_model: {human_model}")

                self.human_model.build(value_votes)
                self.human_labels = [self.human_model.get(i) for i in range(len(self.data))]
            else:
                self.data = test_df.reset_index(drop=True)
                self.human_labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        x = row[self.features]
        y = int(row[self.target])

        result = [x]

        if self.include_human_labels:
            h = self.human_labels[idx]
            if isinstance(h, float):  # soft label
                pass
            else:
                h = int(h)  # majority/sample
            result.append(h)

        if self.include_model_preds:
            m = float(row['model_score'])
            result.append(m)

        result.append(y)

        return tuple(result)