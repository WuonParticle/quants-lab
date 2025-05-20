import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional

import joblib
import paho.mqtt.client as mqtt
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig, HumphBDataFrame

# Configure logging
logger = logging.getLogger(__name__)


class PredictionModelBase(ABC):
    def __init__(self,
                 model_name: str,
                 candles_configs: List[CandlesConfig],
                 mqtt_broker: str,
                 mqtt_port: int,
                 mqtt_topic_prefix: str,
                 mqtt_qos: int = 1,
                 mqtt_retain: bool = True,
                 model_specific_config: Optional[Dict[str, Any]] = None,
                 watched_files: Optional[List[str]] = None):
        self.model_name = model_name
        self.candles_configs = candles_configs
        self.model_specific_config = model_specific_config or {}
        self.watched_files = watched_files or [] # For future use by the runner

        self._candles_feeds: Dict[str, HumphBDataFrame] = {}
        self._mqtt_client = mqtt.Client()
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic_prefix = mqtt_topic_prefix
        self.mqtt_qos = mqtt_qos
        self.mqtt_retain = mqtt_retain
        
        self._running = False
        self._prediction_task: Optional[asyncio.Task] = None

        self._setup_mqtt()
        self._initialize_candles_feeds()
        self.load_model_artifacts() # To be implemented by subclasses if needed

    def _setup_mqtt(self):
        """Configure and connect to the MQTT broker"""
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_disconnect = self._on_disconnect
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            logger.info(f"[{self.model_name}] Connected to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
        except Exception as e:
            logger.error(f"[{self.model_name}] Failed to connect to MQTT broker: {e}")
            raise

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"[{self.model_name}] Successfully connected to MQTT broker.")
            self.publish_status("online")
        else:
            logger.error(f"[{self.model_name}] Failed to connect to MQTT broker with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning(f"[{self.model_name}] Unexpected disconnection from MQTT broker: {rc}. Attempting to reconnect...")
            # Reconnect logic can be added here if needed, though paho-mqtt handles some reconnection.
            # For robust reconnection, a custom loop might be better.
            try:
                self.mqtt_client.reconnect()
            except Exception as e:
                logger.error(f"[{self.model_name}] MQTT reconnection failed: {e}")


    def _initialize_candles_feeds(self):
        for config in self.candles_configs:
            feed_name = f"{config.connector}_{config.trading_pair}_{config.interval}"
            self._candles_feeds[feed_name] = CandlesFactory.get_candle(candles_config=config)
            self._candles_feeds[feed_name].start()
            logger.info(f"[{self.model_name}] Initialized and started candles feed: {feed_name}")

    def get_candles_df(self, connector: str, trading_pair: str, interval: str) -> Optional[HumphBDataFrame]:
        feed_name = f"{connector}_{trading_pair}_{interval}"
        feed = self._candles_feeds.get(feed_name)
        if feed and feed.ready:
            return feed.candles_df.copy()
        elif feed:
            logger.debug(f"[{self.model_name}] Candles feed {feed_name} not ready yet.")
        else:
            logger.warning(f"[{self.model_name}] Candles feed {feed_name} not found.")
        return None

    @abstractmethod
    def load_model_artifacts(self):
        """
        Load model-specific artifacts like trained models, scalers, etc.
        This method should be implemented by subclasses.
        Example:
        self.scaler = joblib.load(self.model_specific_config["scaler_path"])
        self.model = joblib.load(self.model_specific_config["model_path"])
        """
        pass

    @abstractmethod
    async def preprocess_data(self, candles_dfs: Dict[str, HumphBDataFrame]) -> Any:
        """
        Preprocess the raw candle data to generate features for the model.
        Args:
            candles_dfs: A dictionary of DataFrames, where keys are feed_names
                         and values are the corresponding candle DataFrames.
        Returns:
            Processed features ready for the model.
        """
        pass

    @abstractmethod
    async def predict(self, processed_features: Any) -> Any:
        """
        Make a prediction using the loaded model and preprocessed features.
        Args:
            processed_features: Features from preprocess_data.
        Returns:
            The model's prediction.
        """
        pass

    def publish_prediction(self, prediction_data: Any, trading_pair: str, target_info: Optional[Dict[str, Any]] = None):
        """Publish prediction to MQTT broker."""
        signal = {
            "id": int(time.time() * 1000),
            "model_name": self.model_name,
            "trading_pair": trading_pair, # Assuming one primary trading pair for prediction output
            "prediction": prediction_data,
            "timestamp": datetime.now().isoformat(),
        }
        if target_info:
            signal.update(target_info)

        normalized_pair = trading_pair.replace("-", "_").lower()
        topic = f"{self.mqtt_topic_prefix}/{self.model_name}/{normalized_pair}/ML_SIGNALS"
        
        try:
            message = json.dumps(signal, default=str) # Use default=str for non-serializable types like numpy arrays
            result = self.mqtt_client.publish(topic, message, qos=self.mqtt_qos, retain=self.mqtt_retain)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"[{self.model_name}] Published prediction to {topic}: {message}")
            else:
                logger.error(f"[{self.model_name}] Failed to publish prediction to {topic}, error code: {result.rc}")
        except Exception as e:
            logger.error(f"[{self.model_name}] Failed to publish prediction: {e}")
            
    def publish_status(self, status: str):
        """Publish the model's status (e.g., online, offline, reloading)."""
        status_topic = f"{self.mqtt_topic_prefix}/{self.model_name}/status"
        payload = {"status": status, "timestamp": int(time.time() * 1000)}
        try:
            self.mqtt_client.publish(status_topic, json.dumps(payload), qos=self.mqtt_qos, retain=True)
            logger.info(f"[{self.model_name}] Published status to {status_topic}: {status}")
        except Exception as e:
            logger.error(f"[{self.model_name}] Failed to publish status: {e}")

    async def _prediction_loop(self):
        logger.info(f"[{self.model_name}] Starting prediction loop.")
        while self._running:
            try:
                # Check if all candle feeds are ready
                all_ready = True
                current_candles_dfs = {}
                for feed_name, feed_instance in self._candles_feeds.items():
                    if feed_instance.ready:
                        current_candles_dfs[feed_name] = feed_instance.candles_df.copy()
                    else:
                        all_ready = False
                        logger.debug(f"[{self.model_name}] Feed {feed_name} not ready yet.")
                        break 
                
                if not all_ready or not current_candles_dfs:
                    await asyncio.sleep(1) # Wait for feeds to become ready
                    continue

                processed_features = await self.preprocess_data(current_candles_dfs)
                if processed_features is not None:
                    prediction_output = await self.predict(processed_features)
                    if prediction_output is not None:
                        # Determine the primary trading pair for publishing.
                        # This might need to be more sophisticated if models handle multiple pairs.
                        # For now, using the pair from the first candles_config.
                        primary_trading_pair = self.candles_configs[0].trading_pair
                        
                        # Allow subclasses to define target_info if necessary
                        target_info = processed_features.get("target_info") if isinstance(processed_features, dict) else None
                        
                        self.publish_prediction(prediction_output, primary_trading_pair, target_info)
                
                # Adjust sleep time as needed. Could be based on candle interval.
                # For "1s" candles, a short sleep is okay. For longer intervals, align with new candle.
                await asyncio.sleep(self.model_specific_config.get("prediction_interval_sec", 0.5))

            except asyncio.CancelledError:
                logger.info(f"[{self.model_name}] Prediction loop cancelled.")
                break
            except Exception as e:
                logger.exception(f"[{self.model_name}] Error in prediction loop: {e}")
                await asyncio.sleep(5) # Wait a bit before retrying on error
        logger.info(f"[{self.model_name}] Prediction loop stopped.")

    async def start(self):
        if self._running:
            logger.warning(f"[{self.model_name}] Model is already running.")
            return
        self._running = True
        self.publish_status("online")
        self._prediction_task = asyncio.create_task(self._prediction_loop())
        logger.info(f"[{self.model_name}] Model started.")

    async def stop(self):
        if not self._running:
            logger.warning(f"[{self.model_name}] Model is not running.")
            return
        self._running = False
        if self._prediction_task:
            self._prediction_task.cancel()
            try:
                await self._prediction_task
            except asyncio.CancelledError:
                logger.info(f"[{self.model_name}] Prediction task successfully cancelled.")
        
        # Stop candle feeds
        for feed_name, feed in self._candles_feeds.items():
            try:
                feed.stop()
                logger.info(f"[{self.model_name}] Stopped candles feed: {feed_name}")
            except Exception as e:
                logger.error(f"[{self.model_name}] Error stopping candles feed {feed_name}: {e}")

        self.publish_status("offline")
        if self.mqtt_client.is_connected():
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        logger.info(f"[{self.model_name}] Model stopped and MQTT client disconnected.")

    async def reload(self):
        logger.info(f"[{self.model_name}] Attempting to reload model...")
        await self.stop()
        # Give a moment for resources to release
        await asyncio.sleep(1)
        
        # Re-initialize MQTT and candles (in case connection parameters changed, though unlikely here)
        # Or just reload artifacts if that's the primary goal
        try:
            self._setup_mqtt() # Reconnect
            self._initialize_candles_feeds() # Re-init feeds
            self.load_model_artifacts() # Reload model files
            logger.info(f"[{self.model_name}] Model artifacts reloaded.")
            await self.start() # Restart the prediction loop
            logger.info(f"[{self.model_name}] Model reloaded and restarted successfully.")
            self.publish_status("reloaded")
        except Exception as e:
            logger.error(f"[{self.model_name}] Error during model reload: {e}")
            self.publish_status("reload_failed")
            # Attempt to revert to a safe state or stop
            await self.stop() # Ensure it's stopped if reload failed badly

    def is_running(self) -> bool:
        return self._running
