from fastapi import FastAPI, status, File, UploadFile, HTTPException, Body
from datetime import datetime
from typing import Dict, Any, List
import psutil
import logging
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from datetime import date
from fastapi.middleware.cors import CORSMiddleware

from app.api.core.logging import configure_logging
from app.ml.model import HybridARIMALSTM
from app.ml.preprocessing import OilPreprocessor
from app.schemas.health import HealthCheck
from app.api.endpoints.predict import router as predict_router
from app.api.core.config import configure_app
# Configure logging
logger = configure_logging()

app = FastAPI(
    title="Oil Price Prediction API",
    version="0.1.0",
    description="API for hybrid ARIMA-LSTM oil price forecasting"
)
configure_app(app)

app.include_router(predict_router, prefix="/api/v1", tags=["predictions"])

# Add CORS middleware

class OilPriceDataPoint(BaseModel):
    Date: date
    Open: float
    High: float
    Low: float
    Close_Last: float
    Volume: float

    class Config:
        json_schema_extra = {
            "example": {
                "Date": "2024-01-01",
                "Open": 75.2,
                "High": 76.1,
                "Low": 74.9,
                "Close_Last": 75.5,
                "Volume": 100000
            }
        }

class PredictionRequest(BaseModel):
    data: List[OilPriceDataPoint]

model = HybridARIMALSTM(look_back=14)
preprocessor = OilPreprocessor()

@app.on_event("startup")
async def startup():
    """Initialize services during startup"""
    try:
        raw_df = preprocessor.load_raw_data()
        processed_df = preprocessor.process_data(raw_df)
        
        try:
            model.load_models()
            logger.info("Loaded pre-trained models successfully")
        except FileNotFoundError:
            logger.warning("No trained models found. Starting training...")
            model.train_hybrid(processed_df)
            model.save_models()
            logger.info("Completed initial model training")

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Initialization error: {str(e)}") from e

@app.get("/health", response_model=HealthCheck, status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """System health check"""
    return {
        "status": "OK",
        "model_loaded": model.arima_model is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "system_stats": {
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent()
        }
    }

@app.post("/predict/json/")
async def predict_json(request: PredictionRequest = Body(...)) -> Dict[str, Any]:
    """Prediction from JSON data"""
    try:
        df = pd.DataFrame([d.dict() for d in request.data])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').rename(columns={"Close_Last": "Close/Last"})
        df['Volume'] = df['Volume'].clip(lower=0)
        return await process_and_predict(df)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, "Internal server error") from e

async def process_and_predict(df: pd.DataFrame) -> Dict[str, Any]:
    """Core prediction logic"""
    # Validate input
    required_columns = ['Close/Last', 'Volume']
    if missing := list(set(required_columns) - set(df.columns)):
        raise HTTPException(400, f"Missing columns: {', '.join(missing)}")
    
    if len(df) < model.look_back:
        raise HTTPException(400, f"Need at least {model.look_back} data points")

    # Process data
    processed_df = preprocessor.process_data(df)
    
    # Validate feature dimensions
    if processed_df.shape[1] != 7:
        raise HTTPException(400, f"Expected 7 features, got {processed_df.shape[1]}")
    
    # Prepare features
    features = processed_df.values
    scaled_features = model.scaler.transform(features)
    sequence = scaled_features[-model.look_back:].reshape(1, model.look_back, -1)
    
    # Make prediction
    prediction = model.predict(sequence)
    
    return {
        "prediction": round(float(prediction), 4),
        "currency": "USD",
        "unit": "barrel",
        "timestamp": datetime.utcnow().isoformat(),
        "data_points_used": model.look_back
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)