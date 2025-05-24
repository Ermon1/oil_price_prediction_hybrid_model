import pmdarima as pm
import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Attention, Concatenate
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Any
import warnings
import tensorflow as tf
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class HybridARIMALSTM:
    def __init__(self, look_back: int = 14, n_features: int = 13):
        self.look_back = look_back
        self.n_features = n_features
        self.arima_model = None
        self.lstm_model = None
        self.scaler = RobustScaler()
        self.history = None
        self.data_size = None
        self.metrics = {}
        
        # Absolute model paths
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory for TensorBoard
        self.logs_dir = self.models_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _check_data_size(self, data_length: int):
        """Warn about potential overfitting based on data size"""
        try:
            self.data_size = data_length
            min_recommended = self.look_back * 100  # 100 sequences
            if data_length < min_recommended:
                logger.warning(f"Data size ({data_length}) may be too small for look_back {self.look_back}. Recommended minimum: {min_recommended}")
        except Exception as e:
            logger.error(f"Data size check failed: {str(e)}")
            raise

    def train_arima(self, train_data: np.ndarray) -> np.ndarray:
        """Train ARIMA model with enhanced error handling"""
        try:
            logger.info("Training ARIMA model...")
            self.arima_model = pm.auto_arima(
                train_data,
                seasonal=True,  # Enable seasonal decomposition
                m=5,  # Weekly seasonality
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=True,
                max_p=5,
                max_q=5,
                max_d=2,
                maxiter=100,
                n_jobs=-1  # Use all available cores
            )
            
            # Get residuals
            residuals = train_data - self.arima_model.predict_in_sample()
            
            # Log ARIMA model summary
            logger.info(f"ARIMA Model Summary:\n{self.arima_model.summary()}")
            
            return residuals
        except Exception as e:
            logger.error(f"ARIMA training failed: {str(e)}")
            raise

    def prepare_lstm_data(self, residuals: np.ndarray, features: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Prepare LSTM data with validation"""
        try:
            min_length = min(len(residuals), features.shape[0])
            self._check_data_size(min_length)
            
            residuals = residuals[:min_length]
            features = features[:min_length]
            
            # Validate feature dimensions
            if features.shape[1] != self.n_features:
                raise ValueError(f"Feature dimension mismatch. Expected {self.n_features}, got {features.shape[1]}")
            
            scaled_features = self.scaler.fit_transform(features)
            
            X, y = [], []
            for i in range(self.look_back, len(scaled_features)):
                X.append(scaled_features[i-self.look_back:i, :])
                y.append(residuals[i])
            
            # Time-based split (last 20% as validation)
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            return (np.array(X_train), np.array(y_train)), (np.array(X_val), np.array(y_val))
        except Exception as e:
            logger.error(f"LSTM data preparation failed: {str(e)}")
            raise

    def build_lstm(self):
        """Build enhanced LSTM model with attention mechanism"""
        try:
            logger.info("Building LSTM model...")
            
            # Input layer
            inputs = Input(shape=(self.look_back, self.n_features))
            
            # First LSTM layer with return sequences
            x = LSTM(64, return_sequences=True, 
                    kernel_regularizer=L1L2(l1=1e-5, l2=1e-4))(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Second LSTM layer with return sequences
            x = LSTM(32, return_sequences=True,
                    kernel_regularizer=L1L2(l1=1e-5, l2=1e-4))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Attention layer
            attention = Attention()([x, x])
            x = Concatenate()([x, attention])
            
            # Third LSTM layer
            x = LSTM(16, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            # Dense layers
            x = Dense(8, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.1)(x)
            
            # Output layer
            outputs = Dense(1)(x)
        
            # Create model
            self.lstm_model = Model(inputs=inputs, outputs=outputs)
            
            # Compile with custom learning rate
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            self.lstm_model.compile(
                optimizer=optimizer,
                loss='huber',  # More robust to outliers
                metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()]
            )
            
            logger.info("LSTM model built successfully")
        except Exception as e:
            logger.error(f"LSTM model building failed: {str(e)}")
            raise

    def train_hybrid(self, df: pd.DataFrame) -> None:
        """Train both ARIMA and LSTM models"""
        try:
            # Get data length
            data_length = len(df)
            self.data_size = data_length
            
            # Train ARIMA
            self.arima_model = ARIMA(df['Close/Last'], order=(5,1,0))
            self.arima_model = self.arima_model.fit()
            
            # Get ARIMA predictions
            arima_pred = self.arima_model.predict(start=0, end=len(df)-1)
            
            # Prepare LSTM data
            features = df.values
            scaled_features = self.scaler.fit_transform(features)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_features) - self.look_back):
                X.append(scaled_features[i:(i + self.look_back)])
                y.append(scaled_features[i + self.look_back, 0])  # Predict Close/Last
            
            X = np.array(X)
            y = np.array(y)
            
            # Build LSTM model
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.look_back, features.shape[1])),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            
            self.lstm_model.compile(optimizer='adam', loss='mse')
            self.lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            
            # Evaluate model
            self._evaluate_model(X, y, df['Close/Last'].values)
            
            # Save training history
            self.save_training_history()
            
            logger.info("Hybrid model training completed")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def _evaluate_model(self, X_val: np.ndarray, y_val: np.ndarray, original_data: np.ndarray):
        """Evaluate model performance with multiple metrics"""
        try:
            # Get predictions
            lstm_pred = self.lstm_model.predict(X_val)
            arima_pred = self.arima_model.predict(n_periods=len(y_val))
            hybrid_pred = arima_pred + lstm_pred.ravel()
            
            # Calculate metrics
            self.metrics = {
                'lstm_mae': mean_absolute_error(y_val, lstm_pred),
                'lstm_mse': mean_squared_error(y_val, lstm_pred),
                'lstm_rmse': np.sqrt(mean_squared_error(y_val, lstm_pred)),
                'lstm_r2': r2_score(y_val, lstm_pred),
                
                'arima_mae': mean_absolute_error(y_val, arima_pred),
                'arima_mse': mean_squared_error(y_val, arima_pred),
                'arima_rmse': np.sqrt(mean_squared_error(y_val, arima_pred)),
                'arima_r2': r2_score(y_val, arima_pred),
                
                'hybrid_mae': mean_absolute_error(y_val, hybrid_pred),
                'hybrid_mse': mean_squared_error(y_val, hybrid_pred),
                'hybrid_rmse': np.sqrt(mean_squared_error(y_val, hybrid_pred)),
                'hybrid_r2': r2_score(y_val, hybrid_pred)
            }
            
            # Log metrics
            logger.info("Model Evaluation Metrics:")
            for metric, value in self.metrics.items():
                logger.info(f"{metric}: {value:.4f}")
                
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise

    def save_training_history(self):
        """Save training history and plots with enhanced visualization"""
        try:
            # Save history as JSON
            history_path = self.models_dir / "training_history.json"
            pd.DataFrame(self.history.history).to_json(history_path)
            
            # Save metrics
            metrics_path = self.models_dir / "model_metrics.json"
            pd.Series(self.metrics).to_json(metrics_path)
            
            # Create enhanced plots
            plt.figure(figsize=(15, 10))
            
            # Plot loss
            plt.subplot(2, 2, 1)
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot MAE
            plt.subplot(2, 2, 2)
            plt.plot(self.history.history['mae'], label='Training MAE')
            plt.plot(self.history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            # Plot RMSE
            plt.subplot(2, 2, 3)
            plt.plot(self.history.history['root_mean_squared_error'], 
                    label='Training RMSE')
            plt.plot(self.history.history['val_root_mean_squared_error'], 
                    label='Validation RMSE')
            plt.title('Model RMSE')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.legend()
            
            # Plot learning rate
            plt.subplot(2, 2, 4)
            plt.plot(self.history.history['lr'] if 'lr' in self.history.history else [], 
                    label='Learning Rate')
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.models_dir / "training_history.png")
            plt.close()
            
            logger.info("Training history and metrics saved successfully")
        except Exception as e:
            logger.error(f"Failed to save training history: {str(e)}")
            raise

    def save_models(self):
        """Save models with error handling"""
        try:
            logger.info(f"Saving models to {self.models_dir}")
            joblib.dump(self.arima_model, self.models_dir/"arima_model.joblib")
            self.lstm_model.save(self.models_dir/"lstm_model.h5")
            joblib.dump(self.scaler, self.models_dir/"scaler.joblib")
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            raise

    def load_models(self):
        """Load models with error handling"""
        try:
            logger.info(f"Loading models from {self.models_dir}")
            self.arima_model = joblib.load(self.models_dir/"arima_model.joblib")
            self.lstm_model = load_model(self.models_dir/"lstm_model.h5")
            self.scaler = joblib.load(self.models_dir/"scaler.joblib")
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def predict(self, sequence: np.ndarray) -> float:
        """Make prediction using both models"""
        try:
            if self.lstm_model is None:
                raise ValueError("Model not trained")
            
            # Get LSTM prediction
            lstm_pred = self.lstm_model.predict(sequence, verbose=0)[0][0]
            
            # Inverse transform the prediction
            dummy_array = np.zeros((1, self.scaler.n_features_in_))
            dummy_array[0, 0] = lstm_pred
            prediction = self.scaler.inverse_transform(dummy_array)[0, 0]
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise