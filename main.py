# Import libraries
import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from collections import Counter
import joblib
import gzip

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Clean train data
train['job'] = train['job'].str.replace('admin.', 'admin', regex=False)
train = train.drop(columns=['default','duration'])

# Feature Engineering Pipeline
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
        self.cluster_mapping = {0:'senior', 1:'student', 2:'worker', 3:'independent'}
        self.job_to_group = {job: self.cluster_mapping[cluster] for job, cluster in self.pca_clusters.items()}

        # Columns to encode
        self.categorical_cols = ['poutcome', 'contact', 'education', 'marital', 'month',
                                 'housing','loan','job_group','age_group']
        self.le_dict = {}

    def _create_features(self, X):
        """Create derived features consistently for both fit and transform."""
        X = X.copy()
        X['age_group'] = pd.cut(X['age'], bins=self.age_bins, labels=self.age_labels)
        X['job_group'] = X['job'].map(self.job_to_group)
        X['was_contacted_before'] = X['pdays'].apply(lambda x: 0 if x == -1 else 1)
        return X

    def fit(self, X, y=None):
        X = self._create_features(X)

        # Fit LabelEncoders
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str).fillna('Unknown'))
            self.le_dict[col] = le

        return self

    def transform(self, X):
        X = self._create_features(X)

        # Apply LabelEncoders
        for col, le in self.le_dict.items():
            X[col] = le.transform(X[col].astype(str).fillna('Unknown'))

        # Drop unused columns
        drop_cols = ['id','job','age']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])

        return X

# --- Create pipeline ---
pipe = Pipeline([
    ('preprocessor', BankPreprocessor()),
    ('classifier', RandomForestClassifier())
])

# Train test split with SMOTENC
X = train.drop(columns=['y'])
y = train['y']

categorical_cols = ['job','marital','education','housing',
                    'loan','contact','month','poutcome']
categorical_features_mask = X.columns.isin(categorical_cols).tolist()

resampler = SMOTENC(categorical_features = categorical_features_mask, 
                    random_state=42)
X_resampled, y_resampled = resampler.fit_resample(X, y)
print(sorted(Counter(y_resampled).items()))

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the model
pipe.fit(X_train, y_train)

# Print AUC-ROC score
y_probs = pipe.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_probs):.8f}")

# Print classification report
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

# Export model
joblib.dump(pipe, gzip.open('model/rf_smotenc.dat.gz', "wb"))