import pandas as pd
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, raw_path: str = "data/raw", processed_path: str = "data/processed"):
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        try:
            df = pd.read_csv(self.raw_path / "Crude_oil.csv", parse_dates=['Date'])
            df = df.sort_values('Date').set_index('Date')
            
            # Basic validation
            if df.index.duplicated().any():
                df = df.loc[~df.index.duplicated(keep='first')]
                logger.warning("Removed duplicate timestamps")
            
            return df
        except FileNotFoundError:
            logger.error("Raw data file not found")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def save_processed_data(self, df: pd.DataFrame, version: Optional[str] = None) -> None:
        """Save processed data with optional versioning"""
        try:
            if version is None:
                version = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
            
            # Save as CSV
            csv_path = self.processed_path / f"processed_oil_{version}.csv"
            df.to_csv(csv_path)
            
            # Save as Parquet
            parquet_path = self.processed_path / f"processed_oil_{version}.parquet"
            df.to_parquet(parquet_path)
            
            logger.info(f"Saved processed data to {csv_path} and {parquet_path}")
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
            raise