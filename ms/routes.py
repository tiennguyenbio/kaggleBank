# ms/routes.py
from flask import request
from ms import app
from ms.services import get_model_response
import gzip
import joblib

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

@app.route('/predict', methods=['POST'])
def predict():
    """Receives a list of feature data in JSON and returns a list of predictions."""
    if model is None:
        return {'error': 'Model is not available. Please check server logs.'}, 503

    feature_dict = request.get_json()
    if not feature_dict:
        return {'error': 'Request body is empty or not in JSON format.'}, 400

    try:
    # Get the prediction from the services layer
        response = get_model_response(feature_dict, model)
        return response, 200
    except ValueError as e:
    # Catches errors from data conversion or other issues in the service
        return {'error': str(e)}, 400
    except Exception as e:
    # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return {'error': 'An internal server error occurred.'}, 500