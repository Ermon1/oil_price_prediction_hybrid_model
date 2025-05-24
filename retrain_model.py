import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from ml.preprocessing import OilPreprocessor
from ml.model import HybridARIMALSTM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def retrain_model():
    try:
        # Initialize preprocessor and model with smaller look_back
        preprocessor = OilPreprocessor()
        model = HybridARIMALSTM(look_back=14)  # Using smaller look_back value
        
        # Load and process data
        logger.info("Loading raw data...")
        raw_df = preprocessor.load_raw_data()
        logger.info(f"Loaded {len(raw_df)} data points")
        
        # Process data
        logger.info("Processing data...")
        processed_df = preprocessor.process_data(raw_df)
        logger.info(f"Processed data shape: {processed_df.shape}")
        
        # Train model
        logger.info("Training model with look_back=14...")
        model.train_hybrid(processed_df)
        
        # Save model
        logger.info("Saving model...")
        model.save_models()
        
        logger.info("Model retraining completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model retraining: {str(e)}")
        raise

if __name__ == "__main__":
    retrain_model() 