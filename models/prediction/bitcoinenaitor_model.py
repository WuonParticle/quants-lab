import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

import joblib
import pandas as pd
import pandas_ta as ta # noqa: F401

from hummingbot.data_feed.candles_feed.data_types import HumphBDataFrame
from services.prediction_service.prediction_model_base import PredictionModelBase

logger = logging.getLogger(__name__)

class BitcoinenaitorModel(PredictionModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Specific initialization for BitcoinenaitorModel if any, 
        # otherwise, it's handled by load_model_artifacts
        self.scaler = None
        self.model = None
        self.target_rolling_window = self.model_specific_config.get("target_rolling_window", 100)
        self.target_std_window = self.model_specific_config.get("target_std_window", 200)

    def load_model_artifacts(self):
        """Load the scaler and model for the Bitcoinenaitor model."""
        scaler_path = self.model_specific_config.get("scaler_path")
        model_path = self.model_specific_config.get("model_path")
        if not scaler_path or not model_path:
            logger.error(f"[{self.model_name}] scaler_path or model_path not defined in model_specific_config.")
            raise ValueError("Scaler path or model path not configured")
        
        try:
            self.scaler = joblib.load(scaler_path)
            self.model = joblib.load(model_path)
            logger.info(f"[{self.model_name}] Scaler loaded from {scaler_path}")
            logger.info(f"[{self.model_name}] Model loaded from {model_path}")
        except FileNotFoundError as e:
            logger.error(f"[{self.model_name}] Error loading model artifacts: {e}")
            raise
        except Exception as e:
            logger.exception(f"[{self.model_name}] An unexpected error occurred loading model artifacts: {e}")
            raise

    async def preprocess_data(self, candles_dfs: Dict[str, HumphBDataFrame]) -> Optional[Dict[str, Any]]:
        """
        Preprocesses data for the Bitcoinenaitor model.
        Assumes the first candles_config in the list is the primary one for feature engineering.
        """
        if not self.candles_configs:
            logger.warning(f"[{self.model_name}] No candles_configs defined. Cannot preprocess data.")
            return None

        # Use the first candles_config as the primary source for this model
        primary_config = self.candles_configs[0]
        feed_name = f"{primary_config.connector}_{primary_config.trading_pair}_{primary_config.interval}"
        
        candles_df = candles_dfs.get(feed_name)
        if candles_df is None or candles_df.empty:
            logger.debug(f"[{self.model_name}] No data or empty data for {feed_name}.")
            return None

        # Make a copy to avoid modifying the original DataFrame from the feed
        df = candles_df.copy()

        try:
            # 1. Feature Engineering (as in the original PredictionService)
            # Target calculation for context, not directly a feature for this model's input typically
            df["target_pct_val"] = df["close"].rolling(self.target_std_window).std() / df["close"]
            target_pct_for_publishing = df["target_pct_val"].rolling(self.target_rolling_window).mean().iloc[-1]

            df.ta.bbands(length=20, std=2, append=True)  # Standard BB
            df.ta.bbands(length=50, std=2, append=True)  # Longer term BB
            df.ta.macd(fast=12, slow=26, signal=9, append=True)  # Standard MACD
            df.ta.macd(fast=8, slow=21, signal=5, append=True)  # Faster MACD
            df.ta.rsi(length=14, append=True)  # Standard RSI
            df.ta.rsi(length=21, append=True)  # Longer RSI
            df.ta.sma(length=20, append=True)  # Short MA
            df.ta.sma(length=50, append=True)  # Medium MA
            df.ta.ema(length=20, append=True)  # Short EMA
            df.ta.ema(length=50, append=True)  # Medium EMA
            df.ta.atr(length=14, append=True)  # ATR
            df.ta.stoch(k=14, d=3, append=True)  # Stochastic
            df.ta.adx(length=14, append=True)  # ADX
            
            # Retain the latest timestamp from the raw data for context if needed, before dropping columns
            # latest_timestamp_ms = df['timestamp'].iloc[-1] # timestamp is usually index in HumphBDataFrame

            columns_to_drop_initial = ['timestamp', 'taker_buy_base_volume', 'volume']
            # Filter out columns that might not exist to prevent errors
            df = df.drop(columns=[col for col in columns_to_drop_initial if col in df.columns])
            
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    df[f'{col}_ret'] = df[col].pct_change()
            df = df.drop(columns=[col for col in price_columns if col in df.columns])

            if 'taker_buy_quote_volume' in df.columns and 'quote_asset_volume' in df.columns:
                df['buy_volume_ratio'] = df['taker_buy_quote_volume'] / df['quote_asset_volume']
                df = df.drop(columns=['taker_buy_quote_volume'])
            
            df = df.dropna()
            if df.empty:
                logger.debug(f"[{self.model_name}] DataFrame empty after feature engineering and NaN drop for {feed_name}.")
                return None

            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if not numeric_columns: 
                logger.warning(f"[{self.model_name}] No numeric columns found for scaling on {feed_name}.")
                return None
            
            # Ensure scaler is loaded
            if self.scaler is None:
                logger.error(f"[{self.model_name}] Scaler is not loaded. Cannot preprocess.")
                return None
                
            df_scaled = self.scaler.transform(df[numeric_columns])
            df_scaled = pd.DataFrame(df_scaled, columns=numeric_columns, index=df.index)
            
            # Return the last row for prediction and any context needed for publishing
            return {
                "features_df": df_scaled, # The model might expect the last row or a sequence
                "target_info": {"target_pct": target_pct_for_publishing} 
            }

        except Exception as e:
            logger.exception(f"[{self.model_name}] Error during preprocessing for {feed_name}: {e}")
            return None

    async def predict(self, processed_features: Dict[str, Any]) -> Optional[List[float]]:
        """Make a prediction using the Bitcoinenaitor model."""
        if self.model is None:
            logger.error(f"[{self.model_name}] Model is not loaded. Cannot predict.")
            return None
            
        features_df = processed_features.get("features_df")
        if features_df is None or features_df.empty:
            logger.debug(f"[{self.model_name}] No features DataFrame to predict on.")
            return None

        try:
            # Predict using the last row of the scaled features
            prediction_proba = self.model.predict_proba(features_df.iloc[[-1]])[0]
            return prediction_proba.tolist() # Convert numpy array to list for JSON serialization
        except Exception as e:
            logger.exception(f"[{self.model_name}] Error during model prediction: {e}")
            return None

# Example of how you might run this for testing (outside the PredictionServiceRunner)
async def _test_bitcoinenaitor_model():
    from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
    import os
    
    # This test requires actual model and scaler files
    # And a running MQTT broker
    # Create dummy files for testing if you don't have them
    # Ensure root_path is correctly set for your project structure to find models/
    # This assumes this script is run from a location where ../../ is the project root
    # For testing, you might place dummy files in a known relative path.

    # A more robust way to get project root for testing:
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent # services/prediction_service -> services -> quants-lab
    
    # Ensure dummy model/scaler paths exist for testing this script directly
    dummy_model_dir = project_root / "temp_test_models"
    dummy_model_dir.mkdir(exist_ok=True)
    dummy_scaler_path = dummy_model_dir / "dummy_scaler.pkl"
    dummy_model_path = dummy_model_dir / "dummy_model.joblib"

    # Create dummy scaler/model if they don't exist (very basic dummies)
    if not dummy_scaler_path.exists():
        from sklearn.preprocessing import StandardScaler
        dummy_scaler = StandardScaler()
        # Fit with some dummy data that matches expected number of features after TA-Lib
        # This is tricky as the number of features is dynamic. For a true test, use a real scaler.
        # For now, this will likely fail if not aligned with actual features.
        # Consider saving a real scaler/model from a notebook for testing.
        # joblib.dump(dummy_scaler, dummy_scaler_path)
        print(f"Please create a dummy scaler at {dummy_scaler_path} for testing")
    if not dummy_model_path.exists():
        from sklearn.linear_model import LogisticRegression
        # dummy_model = LogisticRegression()
        # joblib.dump(dummy_model, dummy_model_path)
        print(f"Please create a dummy model at {dummy_model_path} for testing")

    if not dummy_scaler_path.exists() or not dummy_model_path.exists():
        print("Dummy model/scaler files are required for this test. Exiting.")
        return

    model_config = {
        "name": "bitcoinenaitor_test",
        "model_class": "models.prediction.bitcoinenaitor_model.BitcoinenaitorModel",
        "watched_files": [str(dummy_scaler_path), str(dummy_model_path)],
        "candles_configs": [
            CandlesConfig(connector="binance", trading_pair="BTC-USDT", interval="1m", max_records=300)
        ],
        "mqtt_broker": "localhost",
        "mqtt_port": 1883,
        "mqtt_topic_prefix": "hbot/test_predictions",
        "model_specific_config": {
            "scaler_path": str(dummy_scaler_path),
            "model_path": str(dummy_model_path),
            "prediction_interval_sec": 2.0,
            "target_rolling_window": 50, # smaller for faster test readiness
            "target_std_window": 100    # smaller for faster test readiness
        }
    }

    model_instance = BitcoinenaitorModel(
        model_name=model_config["name"],
        candles_configs=model_config["candles_configs"],
        mqtt_broker=model_config["mqtt_broker"],
        mqtt_port=model_config["mqtt_port"],
        mqtt_topic_prefix=model_config["mqtt_topic_prefix"],
        model_specific_config=model_config["model_specific_config"]
    )

    try:
        await model_instance.start()
        # Let it run for a bit
        await asyncio.sleep(30) 
    except Exception as e:
        logger.exception(f"Error during test run: {e}")
    finally:
        await model_instance.stop()
        logger.info("Test model stopped.")

if __name__ == "__main__":
    # Setup basic logging for the test
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # To run this test directly: python -m models.prediction.bitcoinenaitor_model
    # Ensure you have dummy_scaler.pkl and dummy_model.joblib or real ones at the specified paths.
    # And an MQTT broker running.
    asyncio.run(_test_bitcoinenaitor_model()) 