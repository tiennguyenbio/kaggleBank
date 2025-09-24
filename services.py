# ms/services.py

import pandas as pd

def predict(X, model):
    """
    Generates a prediction from a model.
    """
    prediction = model.predict(X)#[0]
    return prediction

def get_model_response(json_data, model):
    """
    Processes JSON input, gets a prediction, and formats the response.
    
    Args:
        json_data (dict): The input features from the API request.
        model: The trained scikit-learn pipeline object.
        
    Returns:
        dict: A dictionary containing the prediction and status.
    """
    try:
        # Convert the JSON dictionary to a DataFrame
        X = pd.DataFrame.from_dict(json_data)
    except Exception as e:
        # Handle cases where from_dict fails (e.g., mismatched list lengths)
        raise ValueError(f"Error creating DataFrame from JSON: {e}")

    # Get the prediction from the model
    prediction = predict(X, model)
    
    # Determine the human-readable label
    labels = ['Yes' if p == 1 else 'No' for p in prediction]
    return {
        'status': 200,
        'predictions': [
            {'prediction': int(p), 'label': l} 
            for p, l in zip(prediction, labels)
            ]
            }