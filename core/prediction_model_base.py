import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import paho.mqtt.client as mqtt
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReloadHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        if not event.is_directory:
            logger.info(f"Detected change in {event.src_path}, signaling reload.")
            self.callback()


class BasePredictionModel(ABC):
    def __init__(self,
                 model_name: str,
                 model_class_path: str,
                 candles_config_list: List[Dict[str, Any]],
                 mqtt_topic_prefix: str,
                 watched_files: Optional[List[str]] = None,
                 model_specific_config: Optional[Dict[str, Any]] = None,
                 mqtt_broker: str = "localhost",
                 mqtt_port: int = 1883,
                 mqtt_qos: int = 1,
                 mqtt_retain: bool = True,
                 prediction_interval: float = 1.0):
        self.model_name = model_name
        self.model_class_path = model_class_path # For reloading reference
        self._candles_config_list_defs = candles_config_list # Store original dicts
        self.mqtt_topic_prefix = mqtt_topic_prefix
        self.model_specific_config = model_specific_config or {}
        self.watched_files = watched_files or []
        self.prediction_interval = prediction_interval

        self.candles_list: List[CandlesFactory.cls] = []
        self._setup_candles()

        # MQTT setup
        self.mqtt_client = mqtt.Client(client_id=f"pred-model-{self.model_name}-{int(time.time())}")
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_qos = mqtt_qos
        self.mqtt_retain = mqtt_retain
        self._setup_mqtt()

        self._running = False
        self._reload_event = asyncio.Event()

        # File watcher for hot reloading
        self._observer = None
        if self.watched_files:
            self._setup_file_watcher()
        
        logger.info(f"Initialized Prediction Model: {self.model_name}")

    def _setup_candles(self):
        self.candles_list = []
        for conf_dict in self._candles_config_list_defs:
            # Ensure max_records is present, default if not
            if "max_records" not in conf_dict:
                conf_dict["max_records"] = 1000 # Default value, can be configured
            candles_config = CandlesConfig(**conf_dict)
            candles = CandlesFactory.get_candle(candles_config=candles_config)
            self.candles_list.append(candles)
        logger.info(f"Model {self.model_name}: Candles setup for {len(self.candles_list)} feeds.")

    def _setup_mqtt(self):
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_disconnect = self._on_disconnect
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            logger.info(f"Model {self.model_name}: Connected to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
        except Exception as e:
            logger.error(f"Model {self.model_name}: Failed to connect to MQTT broker: {e}")
            raise

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Model {self.model_name}: Successfully connected to MQTT broker.")
            status_topic = f"{self.mqtt_topic_prefix}/{self.model_name}/status"
            status_payload = {"status": "online", "timestamp": int(time.time() * 1000)}
            self.mqtt_client.publish(status_topic, json.dumps(status_payload), qos=self.mqtt_qos, retain=self.mqtt_retain)
        else:
            logger.error(f"Model {self.model_name}: Failed to connect to MQTT broker with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning(f"Model {self.model_name}: Unexpected disconnection from MQTT broker (rc: {rc}).")
        else:
            logger.info(f"Model {self.model_name}: Disconnected from MQTT broker.")

    def _setup_file_watcher(self):
        event_handler = ReloadHandler(self._signal_reload)
        self._observer = Observer()
        # Ensure watched directories exist or Watchdog will error
        watched_paths = set()
        for f_path in self.watched_files:
            abs_path = os.path.abspath(f_path)
            if os.path.exists(abs_path):
                # Watch the directory containing the file
                watched_paths.add(os.path.dirname(abs_path))
            else:
                logger.warning(f"Model {self.model_name}: Watched file {f_path} does not exist. Skipping.")
        
        for path_to_watch in watched_paths:
            self._observer.schedule(event_handler, path_to_watch, recursive=False) # Recursive False if watching dir
            logger.info(f"Model {self.model_name}: Watching for changes in directory {path_to_watch} for files like {self.watched_files}")

        if watched_paths:
            self._observer.start()
            logger.info(f"Model {self.model_name}: File watcher started for {self.watched_files}.")

    def _signal_reload(self):
        logger.info(f"Model {self.model_name}: Reload signaled.")
        self._reload_event.set()

    @abstractmethod
    async def load_model_artifacts(self):
        """Load any model files, scalers, etc. specific to this prediction model."""
        pass

    @abstractmethod
    async def generate_features(self, candles_dfs: List[Any]) -> Any:
        """
        Generate features from the provided candles dataframes.
        Args:
            candles_dfs: A list of pandas DataFrames, one for each configured candle feed.
        Returns:
            Features ready for the predict method.
        """
        pass

    @abstractmethod
    async def predict(self, features: Any) -> Any:
        """
        Make a prediction based on the generated features.
        Args:
            features: The features generated by generate_features.
        Returns:
            The prediction result.
        """
        pass

    def publish_prediction(self, prediction_data: Any, trading_pair: str, target_info: Optional[Dict[str, Any]] = None):
        signal_id = int(time.time() * 1000)
        payload = {
            "id": signal_id,
            "model_name": self.model_name,
            "trading_pair": trading_pair, # Assuming one primary trading pair for the prediction output for now
            "prediction": prediction_data,
            "timestamp_ms": signal_id,
            "timestamp_iso": datetime.now().isoformat(),
        }
        if target_info:
            payload.update(target_info)

        normalized_pair = trading_pair.replace("-", "_").lower()
        topic = f"{self.mqtt_topic_prefix}/{self.model_name}/{normalized_pair}/prediction"

        try:
            message = json.dumps(payload)
            result = self.mqtt_client.publish(topic, message, qos=self.mqtt_qos, retain=self.mqtt_retain)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Model {self.model_name}: Published prediction to {topic}: {message}")
            else:
                logger.error(f"Model {self.model_name}: Failed to publish prediction to {topic}, error code: {result.rc}")
        except Exception as e:
            logger.error(f"Model {self.model_name}: Failed to publish prediction: {e}")

    async def start_feeds(self):
        for candles in self.candles_list:
            candles.start()
        logger.info(f"Model {self.model_name}: All candle feeds started.")

    async def stop_feeds(self):
        for candles in self.candles_list:
            await candles.stop()
        logger.info(f"Model {self.model_name}: All candle feeds stopped.")

    async def _prediction_loop_iteration(self):
        all_candles_ready = all(c.ready for c in self.candles_list)
        if not all_candles_ready:
            await asyncio.sleep(0.1) # Wait for candles to be ready
            return

        candles_dfs = [c.candles_df.copy() for c in self.candles_list]
        
        # Assuming the first candle config's trading pair is the primary one for publishing
        # This might need refinement if a model uses multiple pairs for a single output signal.
        primary_trading_pair = self.candles_list[0].trading_pair 
        
        try:
            features = await self.generate_features(candles_dfs)
            if features is not None:
                prediction_result = await self.predict(features)
                if prediction_result is not None:
                    # Try to get target_info if predict returns a tuple (prediction, target_info)
                    actual_prediction = prediction_result
                    target_info_dict = None
                    if isinstance(prediction_result, tuple) and len(prediction_result) == 2 and isinstance(prediction_result[1], dict):
                        actual_prediction = prediction_result[0]
                        target_info_dict = prediction_result[1]
                    self.publish_prediction(actual_prediction, primary_trading_pair, target_info_dict)
            else:
                logger.debug(f"Model {self.model_name}: generate_features returned None, skipping prediction.")

        except Exception as e:
            logger.error(f"Model {self.model_name}: Error in prediction loop iteration: {e}", exc_info=True)

    async def run(self):
        self._running = True
        self._reload_event.clear()
        await self.start_feeds()
        await self.load_model_artifacts() # Load initially
        logger.info(f"Prediction model {self.model_name} started.")

        try:
            while self._running:
                if self._reload_event.is_set():
                    logger.info(f"Model {self.model_name}: Reload event triggered. Reloading model artifacts...")
                    await self.load_model_artifacts()
                    self._reload_event.clear()
                    logger.info(f"Model {self.model_name}: Model artifacts reloaded.")
                
                await self._prediction_loop_iteration()
                
                try:
                    # Allow for quick check of reload event
                    await asyncio.wait_for(self._reload_event.wait(), timeout=self.prediction_interval)
                except asyncio.TimeoutError:
                    pass # Normal execution, continue loop
        finally:
            await self.cleanup()

    async def stop(self):
        logger.info(f"Stopping prediction model {self.model_name}...")
        self._running = False
        self._reload_event.set() # Wake up the loop if it's waiting
        # The loop will exit and call cleanup

    async def cleanup(self):
        logger.info(f"Cleaning up prediction model {self.model_name}...")
        await self.stop_feeds()
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info(f"Model {self.model_name}: File watcher stopped.")
        
        status_topic = f"{self.mqtt_topic_prefix}/{self.model_name}/status"
        status_payload = {"status": "offline", "timestamp": int(time.time() * 1000)}
        if self.mqtt_client.is_connected():
            self.mqtt_client.publish(status_topic, json.dumps(status_payload), qos=self.mqtt_qos, retain=self.mqtt_retain)
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        logger.info(f"Model {self.model_name}: MQTT client disconnected. Cleanup complete.") 