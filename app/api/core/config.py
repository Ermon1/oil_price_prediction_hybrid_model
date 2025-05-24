from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

def configure_app(app: FastAPI) -> None:
    """Configure FastAPI application settings"""
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure FastAPI settings
    app.title = "Oil Price Prediction API"
    app.version = "0.1.0"
    app.description = "API for hybrid ARIMA-LSTM oil price forecasting"
    app.docs_url = "/docs"
    app.redoc_url = "/redoc"
    app.openapi_url = "/openapi.json"
    app.debug = True
