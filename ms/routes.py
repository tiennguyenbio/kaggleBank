# ms/routes.py
from flask import request, render_template, jsonify
from ms import app
from ms.services import get_model_response
import gzip
import joblib
import traceback
import pandas as pd

# --- Constants and Model Loading ---
MODEL_NAME = 'Bank Deposit Prediction'
MODEL_VERSION = 'v1.0.0'
MODEL_PATH = 'model/pipe.dat.gz'

print("Loading model...")
try:
    with gzip.open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please run train_model.py first.")
    model = None # Set model to None if it can't be loaded

# --- API Endpoints ---

@app.route('/info', methods=['GET'])
def info():
    """Returns basic information about the model."""
    return {
        'name': MODEL_NAME,
        'version': MODEL_VERSION
    }

@app.route('/health', methods=['GET'])
def health():
    """A simple health check endpoint."""
    if model is None:
        return "Model not loaded", 503 # Service Unavailable
    return "OK", 200

@app.route('/')
def home():
    # first load, no prediction
    return render_template('index.html')

@app.route('/json_help')
def json_help():
    return render_template('json_help.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return {'error': 'Model is not available.'}, 503

    feature_data = None
    submission_type = None  # "manual" or "upload"

    # --- 1. JSON file upload ---
    if 'json_file' in request.files:
        file = request.files['json_file']
        if file.filename != '':
            try:
                import json
                feature_data = json.load(file)
                submission_type = "upload"
            except Exception as e:
                return {'error': f'Error reading JSON file: {e}'}, 400

    # --- 2. JSON API ---
    if feature_data is None:
        feature_data = request.get_json(silent=True)
        if feature_data:
            submission_type = "upload"

    # --- 3. Manual form ---
    if not feature_data:
        if request.form:
            feature_data = [{key: request.form[key] for key in request.form}]
            submission_type = "manual"
        else:
            return {'error': 'No data received.'}, 400

    # --- Normalize input format ---
    if isinstance(feature_data, dict):
        feature_data = [feature_data]
    elif not isinstance(feature_data, list):
        return {'error': 'JSON input must be a dict or list of dicts.'}, 400

    # --- Convert numeric fields ---
    numeric_fields = ["id", "age", "balance", "day", "campaign", "pdays", "previous"]
    for record in feature_data:
        for field in numeric_fields:
            if field in record:
                try:
                    record[field] = float(record[field])
                except ValueError:
                    return {
                        'error': f'Invalid value for {field} in record {record.get("id", "Unknown")}'
                    }, 400

    # --- DataFrame & Prediction ---
    try:
        df = pd.DataFrame(feature_data)
        response = get_model_response(df, model)
        predictions = response.get('predictions', [])

        # Format as HTML list
        formatted_results = [
            f"ID: {int(r.get('id',0))}. Prediction: {p.get('label','Unknown')}"
            for r, p in zip(feature_data, predictions)
        ]
        output_html = "<ul>" + "".join(f"<li>{res}</li>" for res in formatted_results) + "</ul>"

    except Exception as e:
        return {'error': str(e)}, 500

    # --- Render template with section flags ---
    return render_template(
        'index.html',
        prediction_text=output_html,
        submission_type=submission_type  # use this in JS to show/hide sections
    )