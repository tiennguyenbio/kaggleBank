# Import libraries
import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib
import gzip
from preprocessing import BankPreprocessor

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Clean train data
train['job'] = train['job'].str.replace('admin.', 'admin', regex=False)
train = train.drop(columns=['default','duration'])

# Train test split
X = train.drop(columns=['y'])
y = train['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
preprocessor = BankPreprocessor()
model = RandomForestClassifier(random_state=42)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Train the model
pipe.fit(X_train, y_train)

# Print AUC-ROC score
y_probs = pipe.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_probs):.8f}")

# Print classification report
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

# Export model
joblib.dump(pipe, gzip.open('model/pipe.dat.gz', "wb"))