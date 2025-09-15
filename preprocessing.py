from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class BankPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Age bins
        self.age_bins = [18, 25, 35, 45, 55, 65, 100]
        self.age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        
        # Job clustering
        self.pca_clusters = {
            'admin.': 2, 'blue-collar': 2, 'entrepreneur': 2, 'housemaid': 2,
            'management': 0, 'retired': 0, 'self-employed': 3, 'services': 2,
            'student': 1, 'technician': 2, 'unemployed': 3, 'unknown': 3
        }
        self.cluster_mapping = {0: 'senior', 1: 'student', 2: 'worker', 3: 'independent'}
        self.job_to_group = {job: self.cluster_mapping[cluster] for job, cluster in self.pca_clusters.items()}
        
        # Columns to encode
        self.categorical_cols = [
            'poutcome', 'contact', 'education', 'marital', 'month',
            'housing', 'loan', 'job_group', 'age_group'
        ]
        self.le_dict = {}

    def _add_features(self, X):
        """Create derived columns consistently for both fit and transform"""
        X = X.copy()
        X['age_group'] = pd.cut(X['age'], bins=self.age_bins, labels=self.age_labels)
        X['job_group'] = X['job'].map(self.job_to_group)
        X['was_contacted_before'] = (X['pdays'] != -1).astype(int)
        return X

    def fit(self, X, y=None):
        X = self._add_features(X)
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str).fillna('Unknown'))
            self.le_dict[col] = le
        return self

    def transform(self, X):
        X = self._add_features(X)
        for col, le in self.le_dict.items():
            X[col] = le.transform(X[col].astype(str).fillna('Unknown'))
        
        drop_cols = ['id', 'job', 'age']
        return X.drop(columns=[c for c in drop_cols if c in X.columns])