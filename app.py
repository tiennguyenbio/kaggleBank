import datetime

from flask import request
from flask import Flask
import pandas as pd

app = Flask(__name__)

from ms.functions import get_model_response

model_name = 'Bank Deposit Prediction'
model_file = 'rf_smote.dat.gz'
version = 'v1.0.0'

@app.route('/info', methods=['GET'])
def info():
    """Return model information, version"""
    result = {}
    result['name'] = model_name
    result['version'] = version

    return result

@app.route('/predict', methods=['POST'])
def predict():
    feature_dict  = request.get_json()
    if not feature_dict:
        return{
            'error': 'Body is empty.'
        }