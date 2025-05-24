from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import date
from typing import List
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Oil Price Prediction API",
    version="0.1.0",
    description="API for oil price forecasting"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and scaler
look_back = 14
model = None
scaler = MinMaxScaler()

class OilPriceDataPoint(BaseModel):
    Date: date
    Open: float = Field(gt=0)
    High: float = Field(gt=0)
    Low: float = Field(gt=0)
    Close_Last: float = Field(gt=0)
    Volume: float = Field(gt=0)

class PredictionRequest(BaseModel):
    data: List[OilPriceDataPoint]

def create_model(input_shape):
    """Create a more robust LSTM model"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def prepare_data(df: pd.DataFrame) -> tuple:
    """Prepare and validate data for prediction"""
    try:
        # Ensure data is sorted by date
        df = df.sort_index()
        
        # Calculate technical indicators
        df['SMA_5'] = df['Close/Last'].rolling(window=5, min_periods=1).mean()
        df['SMA_20'] = df['Close/Last'].rolling(window=20, min_periods=1).mean()
        
        # Calculate RSI
        delta = df['Close/Last'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close/Last'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close/Last'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        # Handle missing values
        df = df.ffill().bfill()
        
        # Select features
        features = ['Close/Last', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD']
        processed_df = df[features].copy()
        
        # Scale the data
        scaled_data = scaler.fit_transform(processed_df)
        
        # Create sequences
        X = []
        y = []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i])
            y.append(scaled_data[i, 0])  # Predict Close/Last price
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to match LSTM input shape
        X = X.reshape(X.shape[0], look_back, len(features))
        
        return X, y, scaled_data, features
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make prediction from JSON data"""
    try:
        # Validate input data
        if len(request.data) < look_back:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {look_back} data points"
            )
        
        # Convert to DataFrame
        df = pd.DataFrame([d.model_dump() for d in request.data])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').rename(columns={"Close_Last": "Close/Last"})
        
        # Prepare data
        X, y, scaled_data, features = prepare_data(df)
        
        # Create and train model if not exists
        global model
        if model is None:
            logger.info("Creating new model...")
            model = create_model((look_back, len(features)))
            model.fit(
                X, 
                y,
                epochs=50,
                batch_size=32,
                verbose=0
            )
            logger.info("Model training completed")
        
        # Prepare the last sequence for prediction
        last_sequence = scaled_data[-look_back:].reshape(1, look_back, len(features))
        
        # Make prediction
        scaled_prediction = model.predict(last_sequence, verbose=0)[0][0]
        
        # Inverse transform the prediction
        dummy_array = np.zeros((1, len(features)))
        dummy_array[0, 0] = scaled_prediction
        prediction = scaler.inverse_transform(dummy_array)[0, 0]
        
        # Validate prediction
        if not np.isfinite(prediction) or prediction <= 0:
            raise ValueError("Invalid prediction value")
        
        # Calculate confidence score (simple version)
        last_price = float(df['Close/Last'].iloc[-1])
        price_change = abs(prediction - last_price) / last_price
        confidence = max(0, 1 - price_change)
        
        return {
            "prediction": round(float(prediction), 4),
            "currency": "USD",
            "unit": "barrel",
            "last_price": round(last_price, 4),
            "confidence": round(confidence, 4),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port) 