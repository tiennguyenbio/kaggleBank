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

    # --- Get data ---
    feature_data = request.get_json(silent=True)
    
    if not feature_data:
        # Fallback to form submission (single record)
        if request.form:
            feature_data = [{key: request.form[key] for key in request.form}]
        else:
            return {'error': 'No data received.'}, 400
    else:
        # If single dict is passed, wrap it in a list
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
                    return {'error': f'Invalid value for {field} in record {record.get("id", "Unknown")}'}, 400

    # --- Create DataFrame ---
    try:
        df = pd.DataFrame(feature_data)
        print("Feature DataFrame:\n", df)
    except Exception as e:
        print("Error creating DataFrame:", e)
        traceback.print_exc()
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

        # Join multiple records with a line break
        output_html = "<br>".join(formatted_results)
        output_json = {"predictions": [f"ID: {int(rec.get('id',0))}. Prediction: {pred.get('label','Unknown')}" 
                                for rec, pred in zip(feature_data, predictions)]}
        if request.is_json:
            return jsonify(output_json), 200
        else:
            return render_template('index.html', prediction_text=output_html)

    except Exception as e:
        print("Unexpected error in prediction:", e)
        traceback.print_exc()
        return {'error': str(e)}, 500