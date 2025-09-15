import pandas as pd
import model

def predict(X, model):
    prediction = model.predict(X)[0]
    return prediction

def get_model_response(json_data):
    X = pd.DataFrame.from_dict(json_data)
    prediction = predict(X, model)
    if prediction == 1:
        label = 'Yes'
    else:
        label = 'No'
    return label