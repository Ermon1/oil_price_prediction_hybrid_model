from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date
from typing import List
import pandas as pd
import numpy as np
from app.ml.model import HybridARIMALSTM
from app.ml.preprocessing import OilPreprocessor

app = FastAPI(
    title="Oil Price Prediction API",
    version="0.1.0",
    description="API for hybrid ARIMA-LSTM oil price forecasting"
)

# Initialize model and preprocessor
model = HybridARIMALSTM(look_back=14)
preprocessor = OilPreprocessor()

class OilPriceDataPoint(BaseModel):
    Date: date
    Open: float
    High: float
    Low: float
    Close_Last: float
    Volume: float

class PredictionRequest(BaseModel):
    data: List[OilPriceDataPoint]

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make prediction from JSON data"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([d.dict() for d in request.data])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').rename(columns={"Close_Last": "Close/Last"})
        
        # Process data
        processed_df = preprocessor.process_data(df)
        
        # Train model if not trained
        if model.arima_model is None:
            model.train_hybrid(processed_df)
            model.save_models()
        
        # Prepare features for prediction
        features = processed_df.values
        scaled_features = model.scaler.transform(features)
        sequence = scaled_features[-model.look_back:].reshape(1, model.look_back, -1)
        
        # Make prediction
        prediction = model.predict(sequence)
        
        return {
            "prediction": round(float(prediction), 4),
            "currency": "USD",
            "unit": "barrel",
            "last_price": float(df['Close/Last'].iloc[-1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 