import os
import sys
from pathlib import Path
import logging

# Add the app directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from ml.preprocessing import OilPreprocessor
from ml.model import HybridARIMALSTM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_save_model():
    try:
        # Initialize preprocessor and model with optimized look_back
        preprocessor = OilPreprocessor()
        model = HybridARIMALSTM(look_back=14)  # Using optimized look_back value
        
        # Load and process data
        logger.info("Loading and processing data...")
        raw_df = preprocessor.load_raw_data()
        processed_df = preprocessor.process_data(raw_df)
        
        # Train model
        logger.info("Training model with look_back=14...")
        model.train_hybrid(processed_df)
        
        # Save model
        logger.info("Saving model...")
        model.save_models()
        
        logger.info("Model training and saving completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model() 