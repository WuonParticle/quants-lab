import asyncio
import logging
import os
from typing import Any, List, Dict, Optional, Tuple

import joblib
import pandas as pd
import pandas_ta as ta # For feature generation

from core.prediction_model_base import BasePredictionModel

logger = logging.getLogger(__name__)

class ExamplePredictionModel(BasePredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Model-specific attributes, loaded in load_model_artifacts
        self.scaler = None
        self.model = None
        self.scaler_path = self.model_specific_config.get("scaler_path", "models/scaler.pkl")
        self.model_path = self.model_specific_config.get("model_path", "models/model.joblib")
        # Create directories if they don't exist, for initial setup
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    async def load_model_artifacts(self):
        """Load the scaler and model from files."""
        try:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Model {self.model_name}: Scaler loaded from {self.scaler_path}")
            else:
                logger.warning(f"Model {self.model_name}: Scaler file not found at {self.scaler_path}. Prediction might fail or be inaccurate.")
                self.scaler = None # Ensure it's None if file not found

            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Model {self.model_name}: Model loaded from {self.model_path}")
            else:
                logger.warning(f"Model {self.model_name}: Model file not found at {self.model_path}. Prediction might fail.")
                self.model = None # Ensure it's None

        except Exception as e:
            logger.error(f"Model {self.model_name}: Error loading model artifacts: {e}", exc_info=True)
            # Optionally, re-raise or handle as critical failure
            self.scaler = None
            self.model = None

    async def generate_features(self, candles_dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Generate features from the candles data. 
        This example uses the first candles_df for simplicity.
        Adapts feature engineering from the provided PredictionService example.
        """
        if not candles_dfs or candles_dfs[0].empty:
            logger.debug(f"Model {self.model_name}: No candle data or empty DataFrame for feature generation.")
            return None

        candles_df = candles_dfs[0].copy() # Use the first candle feed

        try:
            # Bollinger Bands
            candles_df.ta.bbands(length=20, std=2, append=True, col_names=('BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'))
            candles_df.ta.bbands(length=50, std=2, append=True, col_names=('BBL_50_2.0', 'BBM_50_2.0', 'BBU_50_2.0', 'BBB_50_2.0', 'BBP_50_2.0'))

            # MACD
            candles_df.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9'))
            candles_df.ta.macd(fast=8, slow=21, signal=5, append=True, col_names=('MACD_8_21_5', 'MACDH_8_21_5', 'MACDS_8_21_5'))

            # RSI
            candles_df.ta.rsi(length=14, append=True, col_names=('RSI_14',))
            candles_df.ta.rsi(length=21, append=True, col_names=('RSI_21',))

            # Moving Averages
            candles_df.ta.sma(length=20, append=True, col_names=('SMA_20',))
            candles_df.ta.sma(length=50, append=True, col_names=('SMA_50',))
            candles_df.ta.ema(length=20, append=True, col_names=('EMA_20',))
            candles_df.ta.ema(length=50, append=True, col_names=('EMA_50',))

            # Volatility and Momentum
            candles_df.ta.atr(length=14, append=True, col_names=('ATRr_14',))
            candles_df.ta.stoch(k=14, d=3, append=True, col_names=('STOCHk_14_3_3', 'STOCHd_14_3_3')) 
            candles_df.ta.adx(length=14, append=True, col_names=('ADX_14', 'DMP_14', 'DMN_14'))

            # Price returns
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in candles_df.columns:
                    candles_df[f'{col}_ret'] = candles_df[col].pct_change()
            
            # Volume features (handle missing columns gracefully)
            if 'taker_buy_quote_volume' in candles_df.columns and 'quote_asset_volume' in candles_df.columns and candles_df['quote_asset_volume'].ne(0).any():
                candles_df['buy_volume_ratio'] = candles_df['taker_buy_quote_volume'] / candles_df['quote_asset_volume']
                candles_df['buy_volume_ratio'].fillna(0.5, inplace=True) # Fill NaNs that might result from 0/0
            else:
                candles_df['buy_volume_ratio'] = 0.5 # Default if columns are missing

            # Select only feature columns (that were successfully created)
            # This list should ideally be dynamic or defined based on what the scaler expects
            # For now, let's define a potential list and filter by what exists in candles_df
            potential_feature_cols = [
                'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
                'BBL_50_2.0', 'BBM_50_2.0', 'BBU_50_2.0', 'BBB_50_2.0', 'BBP_50_2.0',
                'MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9',
                'MACD_8_21_5', 'MACDH_8_21_5', 'MACDS_8_21_5',
                'RSI_14', 'RSI_21', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
                'ATRr_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ADX_14', 'DMP_14', 'DMN_14',
                'open_ret', 'high_ret', 'low_ret', 'close_ret', 'buy_volume_ratio'
            ]
            
            feature_cols = [col for col in potential_feature_cols if col in candles_df.columns]
            features_df = candles_df[feature_cols].copy()
            features_df.dropna(inplace=True)

            if features_df.empty:
                logger.debug(f"Model {self.model_name}: Feature DataFrame is empty after processing and NaN drop.")
                return None

            return features_df

        except Exception as e:
            logger.error(f"Model {self.model_name}: Error generating features: {e}", exc_info=True)
            return None

    async def predict(self, features_df: pd.DataFrame) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        Make a prediction using the loaded model and scaler.
        Returns a tuple: (prediction_probabilities, target_info_dict)
        """
        if self.model is None or self.scaler is None:
            logger.warning(f"Model {self.model_name}: Model or scaler not loaded. Cannot predict.")
            return None
        
        if features_df.empty:
            logger.debug(f"Model {self.model_name}: Empty features_df received for prediction.")
            return None

        try:
            # Ensure columns are in the same order as during scaler fitting
            # The scaler should have `feature_names_in_` if it was fit on a DataFrame
            if hasattr(self.scaler, 'feature_names_in_'):
                ordered_features = features_df[self.scaler.feature_names_in_]
            else:
                # Fallback if scaler doesn't have feature_names_in_ (e.g. older scikit-learn or not fit on df)
                logger.warning(f"Model {self.model_name}: Scaler does not have feature_names_in_. Using existing column order.")
                ordered_features = features_df
            
            scaled_features = self.scaler.transform(ordered_features)
            # Get the last row for prediction
            prediction_proba = self.model.predict_proba(scaled_features)[-1]
            
            # Example target_info, adapt as needed
            # This could come from features_df or other calculations
            target_pct = features_df['BBB_20_2.0'].iloc[-1] * 0.1 if 'BBB_20_2.0' in features_df else 0.01 # Example
            target_info = {"target_percentage": float(target_pct)} 

            return prediction_proba.tolist(), target_info
        except KeyError as e:
            logger.error(f"Model {self.model_name}: Missing feature for scaling/prediction: {e}. Ensure all required features are generated and scaler is trained on them.", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Model {self.model_name}: Error during prediction: {e}", exc_info=True)
            return None

# Example usage (for direct testing, not part of the service runner flow)
if __name__ == '__main__':
    # This is for standalone testing of the model class, not how it runs in the service
    async def test_model():
        # Dummy config for testing
        model_config = {
            "model_name": "test_example",
            "model_class_path": "services.prediction_models.example_prediction_model.ExamplePredictionModel",
            "candles_config_list": [
                {
                    "connector": "binance_perpetual",
                    "trading_pair": "BTC-USDT",
                    "interval": "1s",
                    "max_records": 100
                }
            ],
            "mqtt_topic_prefix": "hbot/test_predictions",
            "model_specific_config": {
                "scaler_path": "models/dummy_scaler.pkl",
                "model_path": "models/dummy_model.joblib"
            },
            "watched_files": [] # No watching for this simple test
        }
        
        # Create dummy model and scaler files for testing load_model_artifacts
        os.makedirs("models", exist_ok=True)
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        # Create a dummy scaler
        dummy_scaler = StandardScaler()
        dummy_data = pd.DataFrame(np.random.rand(10, 5), columns=[f'feat_{i}' for i in range(5)])
        dummy_scaler.fit(dummy_data)
        joblib.dump(dummy_scaler, model_config["model_specific_config"]["scaler_path"])

        # Create a dummy model
        dummy_model = LogisticRegression()
        dummy_model.fit(dummy_scaler.transform(dummy_data), np.random.randint(0, 2, 10))
        joblib.dump(dummy_model, model_config["model_specific_config"]["model_path"])

        model = ExamplePredictionModel(**model_config)
        await model.load_model_artifacts()

        # Simulate some candle data
        # In a real scenario, this would come from the running candle feeds
        mock_candles_df = pd.DataFrame({
            'timestamp': pd.to_datetime(pd.Timestamp.now(tz='UTC').timestamp - np.arange(100, 0, -1), unit='s'),
            'open': np.random.rand(100) * 100 + 40000,
            'high': np.random.rand(100) * 100 + 40050,
            'low': np.random.rand(100) * 100 + 39950,
            'close': np.random.rand(100) * 100 + 40000,
            'volume': np.random.rand(100) * 10 + 1,
            'taker_buy_quote_volume': np.random.rand(100) * 500000 + 20000000,
            'quote_asset_volume': np.random.rand(100) * 1000000 + 40000000
        })
        model.candles_list[0].candles_df = mock_candles_df # Manually set for testing
        model.candles_list[0]._ready = True

        print("Running one prediction loop iteration for testing...")
        await model._prediction_loop_iteration() # Call the internal loop method for one cycle
        
        print("Test model run complete. Check MQTT (if broker is running and subscribed) or logs.")
        await model.cleanup()
        # Clean up dummy files
        os.remove(model_config["model_specific_config"]["scaler_path"])
        os.remove(model_config["model_specific_config"]["model_path"])

    asyncio.run(test_model()) 