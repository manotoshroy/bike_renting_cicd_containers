from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("bikeshare__model_output_v0.0.1.pkl")

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}