import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

MODEL_PATH = 'best_isolation_forest_model.pkl'

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def preprocess_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

def predict_anomalies(model, data):
    predictions = model.predict(data)
    anomalies = predictions == -1
    return anomalies
