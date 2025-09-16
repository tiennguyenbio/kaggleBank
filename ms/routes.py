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

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return {'error': 'Model is not available.'}, 503

    feature_data = None

    # --- 1. Check if JSON file was uploaded ---
    if 'json_file' in request.files:
        file = request.files['json_file']
        if file.filename != '':
            try:
                import json
                feature_data = json.load(file)
            except Exception as e:
                return {'error': f'Error reading JSON file: {e}'}, 400

    # --- 2. Check if JSON was sent in request body ---
    if feature_data is None:
        feature_data = request.get_json(silent=True)

    # --- 3. Check if manual form submission ---
    if not feature_data:
        if request.form:
            feature_data = [{key: request.form[key] for key in request.form}]
        else:
            return {'error': 'No data received.'}, 400

    # --- Normalize input format (always a list of dicts) ---
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

    # --- Create DataFrame ---
    try:
        df = pd.DataFrame(feature_data)
        print("Feature DataFrame:\n", df)
    except Exception as e:
        return {'error': f"Error creating DataFrame: {str(e)}"}, 400

    # --- Get predictions ---
    try:
        response = get_model_response(df, model)
        predictions = response.get('predictions', [])

        # --- Format output ---
        formatted_results = []
        for rec, pred in zip(feature_data, predictions):
            cust_id = int(rec.get('id', 0))
            label = pred.get('label', 'Unknown')
            formatted_results.append(f"ID: {cust_id}. Prediction: {label}")

        output_html = "<ul>" + "".join(f"<li>{res}</li>" for res in formatted_results) + "</ul>"
        output_json = {"predictions": formatted_results}

        # --- Decide response type ---
        if request.form:  # manual form
            return render_template('index.html', prediction_text=output_html)
        elif 'json_file' in request.files:  # file upload
            # Render HTML with formatted results
            return render_template('index.html', prediction_text=output_html)
        else:  # API JSON request
            return jsonify(output_json), 200

    except Exception as e:
        print("Unexpected error in prediction:", e)
        return {'error': str(e)}, 500