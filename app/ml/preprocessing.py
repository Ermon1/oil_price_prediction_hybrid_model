import pandas as pd
import numpy as np
import logging
import ta
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

class OilPreprocessor:
    def __init__(self, raw_path: str = "data/raw", processed_path: str = "data/processed"):
        """Initialize preprocessor with paths and scaler"""
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.scaler = RobustScaler()
        self.expected_features = 12  # Number of features after processing
        self.min_samples = 14  # Minimum required samples
        
        # Create directories if they don't exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV file"""
        try:
            # Look for the most recent CSV file in the raw data directory
            csv_files = list(self.raw_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in raw data directory")
            
            # Get the most recent file
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading data from {latest_file}")
            
            # Read the CSV file
            df = pd.read_csv(latest_file)
            
            # Ensure Date column is datetime
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Rename Close_Last to Close/Last if needed
            if 'Close_Last' in df.columns:
                df = df.rename(columns={'Close_Last': 'Close/Last'})
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to load raw data: {str(e)}")
            raise

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Enhanced data validation"""
        try:
            # Check feature count (including Close/Last if present)
            expected_cols = self.expected_features + (1 if 'Close/Last' in df.columns else 0)
            if df.shape[1] != expected_cols:
                raise ValueError(
                    f"Feature mismatch: Expected {expected_cols}, "
                    f"Got {df.shape[1]}. Features: {df.columns.tolist()}"
                )
            
            # Check sample count
            if len(df) < self.min_samples:
                raise ValueError(f"Need at least {self.min_samples} samples")
            
            # Check for missing values
            if df.isnull().any().any():
                raise ValueError("Data contains missing values after processing")
            
            # Check for infinite values
            if np.isinf(df.values).any():
                raise ValueError("Data contains infinite values")
                
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        # Handle division by zero
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # Fill any remaining NaN values
        rsi = rsi.fillna(50)  # Default to neutral RSI value
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD technical indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        return macd

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the data for model input"""
        try:
            # Ensure date index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Handle missing values
            df = df.fillna(method='ffill')  # Forward fill
            df = df.fillna(method='bfill')  # Backward fill for any remaining NaNs
            
            # Calculate technical indicators
            df['SMA_5'] = df['Close/Last'].rolling(window=5, min_periods=1).mean()
            df['SMA_20'] = df['Close/Last'].rolling(window=20, min_periods=1).mean()
            df['RSI'] = self._calculate_rsi(df['Close/Last'])
            df['MACD'] = self._calculate_macd(df['Close/Last'])
            
            # Fill any remaining NaN values with forward fill
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')  # Backward fill for any remaining NaNs
            
            if df.empty:
                raise ValueError("No valid data after processing")
            
            # Select features
            features = ['Close/Last', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD']
            processed_df = df[features].copy()
            
            # Validate processed data
            if processed_df.isnull().any().any():
                raise ValueError("Data contains missing values after processing")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical indicators"""
        try:
            # Price-based features
            df['7d_ma'] = df['Close/Last'].rolling(7, min_periods=1).mean()
            df['30d_ma'] = df['Close/Last'].rolling(30, min_periods=1).mean()
            df['90d_ma'] = df['Close/Last'].rolling(90, min_periods=1).mean()
            
            # RSI
            df['rsi_14'] = ta.momentum.RSIIndicator(df['Close/Last'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close/Last'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close/Last'])
            df['bollinger_upper'] = bollinger.bollinger_hband()
            df['bollinger_lower'] = bollinger.bollinger_lband()

            # ATR
            df['atr_14'] = ta.volatility.AverageTrueRange(
                high=df['Close/Last'],
                low=df['Close/Last'],
                close=df['Close/Last'],
                window=14
            ).average_true_range()
            
            # Volume features
            df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(30, min_periods=1).mean()
            
            # Price volatility
            df['price_volatility'] = df['Close/Last'].rolling(14, min_periods=1).std()
        
            # Ensure all required features are present
            required_features = [
                'Close/Last', 'Volume', '7d_ma', '30d_ma', '90d_ma',
                'rsi_14', 'macd', 'macd_signal', 'bollinger_upper',
                'bollinger_lower', 'atr_14', 'volume_ma_ratio', 'price_volatility'
            ]
            
            # Verify all features exist
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Select features in the correct order
            df = df[required_features]
        
        # Final cleaning
            df = df.ffill().bfill()
            
            return df
        except Exception as e:
            logger.error(f"Feature creation failed: {str(e)}")
            raise

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using RobustScaler"""
        try:
            # Don't scale the target variable
            target = df['Close/Last'] if 'Close/Last' in df.columns else None
            features = df.drop(columns=['Close/Last']) if 'Close/Last' in df.columns else df
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)
            
            # Add back target if it exists
            if target is not None:
                scaled_df['Close/Last'] = target
            
            return scaled_df
        except Exception as e:
            logger.error(f"Feature scaling failed: {str(e)}")
            raise 