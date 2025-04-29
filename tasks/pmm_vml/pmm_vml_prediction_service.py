import asyncio
import logging
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import joblib
import paho.mqtt.client as mqtt

from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from tasks.pmm_vml.pmm_vml_utils import generate_features, preprocess_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMMVMLPredictionService:
    """
    Service that runs continuously, generates features from live candle data,
    uses trained models to predict 6 PMM parameters, and publishes predictions to MQTT.
    """
    
    def __init__(self, 
                 model_path: Path, 
                 scaler_path: Path,
                 imputer_path: Path,
                 candles_config: List[CandlesConfig],
                 mqtt_broker: str = "localhost", 
                 mqtt_port: int = 1883,
                 mqtt_topic_prefix: str = "hbot/predictions", 
                 mqtt_qos: int = 1, 
                 mqtt_retain: bool = True):
        """
        Initialize the prediction service
        
        Args:
            model_path: Path to the directory containing the trained models
            scaler_path: Path to the directory containing the scalers
            imputer_path: Path to the directory containing the imputers
            candles_config: List of CandlesConfig objects, one per trading pair
            mqtt_broker: MQTT broker address
            mqtt_port: MQTT broker port
            mqtt_topic_prefix: Prefix for MQTT topics
            mqtt_qos: MQTT quality of service level
            mqtt_retain: Whether to retain MQTT messages
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.imputer_path = Path(imputer_path)
        self.candles_config = candles_config
        
        # Setup MQTT client
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic_prefix = mqtt_topic_prefix
        self.mqtt_qos = mqtt_qos
        self.mqtt_retain = mqtt_retain
        self.mqtt_client = mqtt.Client()
        
        # Initialize dictionaries to store models, scalers, imputers, and candle feeds
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.candles = {}
        
        # Load models, scalers, and imputers for each trading pair
        self._load_models_and_preprocessing()
        
        # Initialize candle feeds
        self._initialize_candle_feeds()
        
        # Setup MQTT connection
        self.setup_mqtt()
    
    def _load_models_and_preprocessing(self):
        """Load trained models, scalers, and imputers for each trading pair"""
        for candle_config in self.candles_config:
            trading_pair = candle_config.trading_pair
            normalized_pair = trading_pair.replace('-', '_')
            
            # Load model
            model_file = self.model_path / f"{normalized_pair}_model.joblib"
            if model_file.exists():
                try:
                    self.models[trading_pair] = joblib.load(model_file)
                    logger.info(f"Loaded model for {trading_pair}")
                except Exception as e:
                    logger.error(f"Error loading model for {trading_pair}: {e}")
                    continue
            else:
                logger.error(f"Model file not found for {trading_pair}: {model_file}")
                continue
            
            # Load scaler
            scaler_file = self.scaler_path / f"{normalized_pair}_scaler.joblib"
            if scaler_file.exists():
                try:
                    self.scalers[trading_pair] = joblib.load(scaler_file)
                    logger.info(f"Loaded scaler for {trading_pair}")
                except Exception as e:
                    logger.error(f"Error loading scaler for {trading_pair}: {e}")
            else:
                logger.warning(f"Scaler file not found for {trading_pair}: {scaler_file}")
            
            # Load imputer
            imputer_file = self.imputer_path / f"{normalized_pair}_imputer.joblib"
            if imputer_file.exists():
                try:
                    self.imputers[trading_pair] = joblib.load(imputer_file)
                    logger.info(f"Loaded imputer for {trading_pair}")
                except Exception as e:
                    logger.error(f"Error loading imputer for {trading_pair}: {e}")
            else:
                logger.warning(f"Imputer file not found for {trading_pair}: {imputer_file}")
    
    def _initialize_candle_feeds(self):
        """Initialize candle feeds for each trading pair"""
        for candle_config in self.candles_config:
            trading_pair = candle_config.trading_pair
            try:
                self.candles[trading_pair] = CandlesFactory.get_candle(candles_config=candle_config)
                self.candles[trading_pair].start()
                logger.info(f"Started candle feed for {trading_pair}")
            except Exception as e:
                logger.error(f"Error starting candle feed for {trading_pair}: {e}")
    
    def setup_mqtt(self):
        """Configure and connect to the MQTT broker"""
        # Setup connection callbacks
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        # Connect to the broker
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port)
            self.mqtt_client.loop_start()  # Start the loop in a separate thread
            logger.info(f"Connected to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback for when the client receives a CONNACK response from the server"""
        if rc == 0:
            logger.info("Successfully connected to MQTT broker")
            # Publish a status message that we're online
            self.mqtt_client.publish(
                f"{self.mqtt_topic_prefix}/status", 
                json.dumps({"status": "online", "timestamp": int(time.time() * 1000)}),
                qos=self.mqtt_qos,
                retain=True
            )
        else:
            logger.error(f"Failed to connect to MQTT broker with code {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        """Callback for when the client disconnects from the server"""
        if rc != 0:
            logger.error(f"Unexpected disconnection from MQTT broker: {rc}")
            # Try to reconnect
            logger.info("Attempting to reconnect...")
            self.mqtt_client.loop_stop()
            self.setup_mqtt()
    
    def publish_prediction(self, predictions, trading_pair):
        """
        Publish prediction to MQTT broker
        
        Args:
            predictions: Array of 6 PMM parameters
            trading_pair: Trading pair the predictions are for
        """
        # Parameter names matching the training data
        param_names = [
            'buy_0_bps', 'buy_1_step', 'sell_0_bps', 
            'sell_1_step', 'take_profit_pct', 'stop_loss_pct'
        ]
        
        # Create a dictionary mapping parameter names to predicted values
        parameters = {}
        for i, param in enumerate(param_names):
            # Convert numpy types to Python native types
            parameters[param] = float(predictions[i])
        
        # Create the payload
        payload = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "trading_pair": trading_pair,
            "parameters": parameters
        }
        
        # Create topic based on trading pair
        normalized_pair = trading_pair.replace("-", "_").lower()
        topic = f"{self.mqtt_topic_prefix}/{normalized_pair}/PMM_PARAMS"
        
        # Convert to JSON and publish
        try:
            message = json.dumps(payload)
            result = self.mqtt_client.publish(
                topic, 
                message, 
                qos=self.mqtt_qos, 
                retain=self.mqtt_retain
            )
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Published prediction to {topic}: {message}")
            else:
                logger.error(f"Failed to publish prediction, error code: {result.rc}")
        except Exception as e:
            logger.error(f"Failed to publish prediction: {e}")
    
    async def prediction_loop(self):
        """Main prediction loop that runs continuously"""
        logger.info("Starting prediction loop")
        
        while True:
            for trading_pair, candle_feed in self.candles.items():
                if trading_pair not in self.models:
                    continue
                    
                if candle_feed.ready:
                    try:
                        # Get latest candle data
                        candles_df = candle_feed.candles_df.copy()
                        
                        # Generate features using shared utility function
                        features_df = generate_features(candles_df)
                        
                        # Skip if we don't have enough data for feature calculation
                        if features_df.empty:
                            logger.warning(f"No features generated for {trading_pair}, skipping prediction")
                            continue
                        
                        # Get the imputer and scaler for this trading pair
                        imputer = self.imputers.get(trading_pair)
                        scaler = self.scalers.get(trading_pair)
                        
                        # Preprocess features (handle missing values and scale)
                        processed_features = preprocess_features(features_df, imputer, scaler)
                        
                        # Skip if we don't have enough data after preprocessing
                        if processed_features.empty:
                            logger.warning(f"No valid data after preprocessing for {trading_pair}, skipping prediction")
                            continue
                        
                        # Get the latest row for prediction
                        latest_features = processed_features.iloc[-1:].values
                        
                        # Make prediction
                        predictions = self.models[trading_pair].predict(latest_features)[0]
                        
                        # Publish prediction
                        self.publish_prediction(predictions, trading_pair)
                        
                    except Exception as e:
                        logger.error(f"Error in prediction loop for {trading_pair}: {e}", exc_info=True)
            
            # Short sleep to yield control
            await asyncio.sleep(0.1)
    
    def cleanup(self):
        """Clean up resources when done"""
        logger.info("Cleaning up resources")
        
        # Publish offline status
        try:
            self.mqtt_client.publish(
                f"{self.mqtt_topic_prefix}/status", 
                json.dumps({"status": "offline", "timestamp": int(time.time() * 1000)}),
                qos=self.mqtt_qos,
                retain=True
            )
        except Exception as e:
            logger.error(f"Error publishing offline status: {e}")
        
        # Stop MQTT loop and disconnect
        try:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            logger.info("Disconnected from MQTT broker")
        except Exception as e:
            logger.error(f"Error disconnecting from MQTT: {e}")
        
        # Stop candle feeds
        for trading_pair, candle_feed in self.candles.items():
            try:
                candle_feed.stop()
                logger.info(f"Stopped candle feed for {trading_pair}")
            except Exception as e:
                logger.error(f"Error stopping candle feed for {trading_pair}: {e}")


async def main():
    # Force is needed to override other logging configurations
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    
    # Get paths
    root_path = Path(os.path.abspath(os.path.join(os.getcwd())))
    model_path = root_path / "models" / "pmm_vml"
    scaler_path = model_path
    imputer_path = model_path
    
    # Configure trading pairs
    trading_pairs = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    
    # Create CandlesConfig objects
    candles_configs = []
    for trading_pair in trading_pairs:
        candles_configs.append(
            CandlesConfig(
                connector="binance_perpetual",
                trading_pair=trading_pair,
                interval="1m",
                max_records=1000  # Sufficient for feature calculation
            )
        )
    
    # Create prediction service
    prediction_service = PMMVMLPredictionService(
        model_path=model_path,
        scaler_path=scaler_path,
        imputer_path=imputer_path,
        candles_config=candles_configs,
        mqtt_broker="localhost",
        mqtt_port=1883,
        mqtt_topic_prefix="hbot/predictions",
        mqtt_qos=1,
        mqtt_retain=True
    )
    
    try:
        await prediction_service.prediction_loop()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    finally:
        prediction_service.cleanup()


if __name__ == "__main__":
    debug = os.environ.get("ASYNC_DEBUG", False)
    asyncio.run(main(), debug=debug) 