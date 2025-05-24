import logging
from pathlib import Path
from .model import HybridARIMALSTM
from .data_loader import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    try:
        model = HybridARIMALSTM()
        df = load_data()
        logger.info("Data loaded successfully")
        model.train_hybrid(df)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()