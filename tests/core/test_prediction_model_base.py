import asyncio
import os
import time
from typing import Any, List
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call
import pandas as pd
import joblib
import paho.mqtt.client as mqtt
from pathlib import Path
import json # Added for parsing MQTT payload in test

# Ensure core and services are importable
import sys
PROJECT_ROOT_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT_TEST not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_TEST)

from core.prediction_model_base import BasePredictionModel, ReloadHandler
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig

# A concrete implementation for testing BasePredictionModel
class MockConcretePredictionModel(BasePredictionModel):
    def __init__(self, *args, **kwargs):
        self.load_artifacts_called = 0
        self.generate_features_called = 0
        self.predict_called = 0
        # Use model_specific_config to get paths for clarity
        self.custom_scaler_path = kwargs.get('model_specific_config', {}).get('scaler_path', 'test_scaler.pkl')
        self.custom_model_path = kwargs.get('model_specific_config', {}).get('model_path', 'test_model.joblib')
        super().__init__(*args, **kwargs)

    async def load_model_artifacts(self):
        self.load_artifacts_called += 1
        Path(self.custom_scaler_path).touch(exist_ok=True)
        Path(self.custom_model_path).touch(exist_ok=True)
        self.scaler = MagicMock() 
        self.model = MagicMock()  
        await asyncio.sleep(0.01) # Simulate async load

    async def generate_features(self, candles_dfs: List[pd.DataFrame]) -> Any:
        self.generate_features_called += 1
        if not candles_dfs or candles_dfs[0].empty:
            return None
        return pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})

    async def predict(self, features: Any) -> Any:
        self.predict_called += 1
        if features is None:
            return None
        return [0.6, 0.3, 0.1], {"target_example": 0.05}

class TestBasePredictionModel(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.model_name = "test_model"
        self.mqtt_topic_prefix = "test/predictions"
        self.candles_config_list_dict = [
            {
                "connector": "binance_perpetual",
                "trading_pair": "BTC-USDT",
                "interval": "1s",
                "max_records": 100
            }
        ]
        self.model_specific_config_dict = {
            "scaler_path": "test_scaler.pkl",
            "model_path": "test_model.joblib"
        }
        # Ensure watched files are always defined for tests that expect observer behavior
        self.watched_files_list = [self.model_specific_config_dict["scaler_path"], self.model_specific_config_dict["model_path"]]

        for f_path in self.watched_files_list:
            Path(f_path).parent.mkdir(parents=True, exist_ok=True)
            Path(f_path).touch()

        self.mock_mqtt_client_instance = MagicMock(spec=mqtt.Client) # Using spec for better mocking
        self.mock_mqtt_client_instance.is_connected.return_value = True

        self.mock_candles_instance = MagicMock()
        self.mock_candles_instance.ready = True
        self.mock_candles_instance.candles_df = pd.DataFrame({
            'timestamp': [time.time() - 2, time.time() - 1],
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [10, 11]
        })
        self.mock_candles_instance.trading_pair = "BTC-USDT"
        self.mock_candles_instance.start = MagicMock()
        self.mock_candles_instance.stop = AsyncMock()

        self.mock_observer_instance = MagicMock(spec=unittest.mock.Mock) # Using spec
        self.mock_observer_instance.start = MagicMock()
        self.mock_observer_instance.stop = MagicMock()
        self.mock_observer_instance.join = MagicMock()
        self.mock_observer_instance.schedule = MagicMock()
        self.mock_observer_instance.is_alive = MagicMock(return_value=True) # For cleanup tests

        self.patch_mqtt_client = patch("paho.mqtt.client.Client", return_value=self.mock_mqtt_client_instance)
        self.patch_candles_factory = patch("hummingbot.data_feed.candles_feed.candles_factory.CandlesFactory.get_candle", return_value=self.mock_candles_instance)
        self.patch_observer = patch("core.prediction_model_base.Observer", return_value=self.mock_observer_instance)
        
        self.mock_mqtt_client_class = self.patch_mqtt_client.start()
        self.mock_candles_factory_get_candle = self.patch_candles_factory.start()
        self.mock_observer_class = self.patch_observer.start()
        
        self.model = MockConcretePredictionModel(
            model_name=self.model_name,
            model_class_path="path.to.MockConcretePredictionModel",
            candles_config_list=self.candles_config_list_dict.copy(),
            mqtt_topic_prefix=self.mqtt_topic_prefix,
            watched_files=self.watched_files_list.copy(),
            model_specific_config=self.model_specific_config_dict.copy(),
            prediction_interval=0.05
        )

    def tearDown(self):
        self.patch_mqtt_client.stop()
        self.patch_candles_factory.stop()
        self.patch_observer.stop()
        for f_path in self.watched_files_list:
            if os.path.exists(f_path): os.remove(f_path)
        dir_path = os.path.dirname(self.watched_files_list[0])
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            try:
                os.rmdir(dir_path)
            except OSError: # Ignore if other files were created e.g. by other tests
                pass

    async def test_initialization_with_watched_files(self):
        self.assertEqual(self.model.model_name, self.model_name)
        self.mock_mqtt_client_class.assert_called_once()
        self.mock_mqtt_client_instance.connect.assert_called_once()
        self.mock_candles_factory_get_candle.assert_called_once()
        self.assertEqual(len(self.model.candles_list), 1)
        
        # Verifying Observer interactions
        self.mock_observer_class.assert_called_once() # Observer() should be called
        self.assertEqual(self.model._observer, self.mock_observer_instance) # _observer attribute should be our mock instance
        self.mock_observer_instance.schedule.assert_called() # schedule should be called on the instance
        self.mock_observer_instance.start.assert_called_once() # start should be called on the instance

    async def test_initialization_without_watched_files(self):
        # Stop the global observer patch to re-initialize model without it being called
        self.patch_observer.stop()
        # Create a new patcher for Observer that WON'T have a return_value, so it's just a class mock
        observer_class_mock_no_watch = patch("watchdog.observers.Observer").start()

        model_no_watch = MockConcretePredictionModel(
            model_name="no_watch_model",
            model_class_path="path.to.Model",
            candles_config_list=self.candles_config_list_dict.copy(),
            mqtt_topic_prefix="test/nowatch",
            watched_files=[], # Explicitly empty
            model_specific_config=self.model_specific_config_dict.copy(),
        )
        self.assertIsNone(model_no_watch._observer)
        observer_class_mock_no_watch.assert_not_called() # Observer() should NOT be called
        self.patch_observer.start() # Restart global patcher for other tests
        observer_class_mock_no_watch.stop()

    async def test_prediction_loop_iteration(self):
        await self.model.load_model_artifacts()
        await self.model._prediction_loop_iteration()
        self.assertEqual(self.model.generate_features_called, 1)
        self.assertEqual(self.model.predict_called, 1)
        self.mock_mqtt_client_instance.publish.assert_called()
        last_call_args, _ = self.mock_mqtt_client_instance.publish.call_args_list[-1]
        self.assertIn(f"{self.mqtt_topic_prefix}/{self.model_name}/btc_usdt/prediction", last_call_args[0])

    async def test_run_and_stop(self):
        run_task = asyncio.create_task(self.model.run())
        await asyncio.sleep(self.model.prediction_interval * 3)
        self.assertTrue(self.model._running)
        self.assertGreaterEqual(self.model.load_artifacts_called, 1)
        self.assertGreaterEqual(self.model.generate_features_called, 1)
        self.assertGreaterEqual(self.model.predict_called, 1)
        await self.model.stop()
        await run_task
        self.assertFalse(self.model._running)

        offline_call_found = False
        expected_status_topic = f"{self.mqtt_topic_prefix}/{self.model_name}/status"
        for pub_call in self.mock_mqtt_client_instance.publish.call_args_list:
            topic, payload_str = pub_call[0][:2]
            if topic == expected_status_topic:
                try:
                    payload = json.loads(payload_str)
                    if payload.get("status") == "offline":
                        offline_call_found = True; break
                except json.JSONDecodeError: pass
        self.assertTrue(offline_call_found, "Offline status not published on cleanup")

    async def test_file_watching_and_reload(self):
        await self.model.start_feeds()
        await self.model.load_model_artifacts()
        initial_load_calls = self.model.load_artifacts_called
        self.assertEqual(initial_load_calls, 1)

        self.model._reload_event.clear()
        self.model._signal_reload()
        self.assertTrue(self.model._reload_event.is_set())

        # Simulate one iteration of the run loop's core logic for reload
        if self.model._reload_event.is_set():
            await self.model.load_model_artifacts()
            self.model._reload_event.clear()
        
        self.assertFalse(self.model._reload_event.is_set(), "_reload_event should be cleared")
        self.assertEqual(self.model.load_artifacts_called, initial_load_calls + 1, "load_model_artifacts called again")
        await self.model.stop_feeds()

    async def test_cleanup_with_observer(self):
        # Model is already initialized with watched files, so _observer should be mock_observer_instance
        self.assertEqual(self.model._observer, self.mock_observer_instance)
        await self.model.cleanup()
        self.mock_candles_instance.stop.assert_called_once()
        self.mock_observer_instance.stop.assert_called_once()
        self.mock_observer_instance.join.assert_called_once()
        self.mock_mqtt_client_instance.disconnect.assert_called_once()

    async def test_cleanup_without_observer(self):
        self.patch_observer.stop() # Stop global patch
        observer_class_mock_no_watch = patch("watchdog.observers.Observer").start()
        model_no_watch = MockConcretePredictionModel(
            model_name="no_watch_model_cleanup",
            model_class_path="path.to.Model",
            candles_config_list=self.candles_config_list_dict.copy(),
            mqtt_topic_prefix="test/nowatch_cleanup",
            watched_files=[], # No watched files
            model_specific_config=self.model_specific_config_dict.copy(),
        )
        self.assertIsNone(model_no_watch._observer)
        await model_no_watch.cleanup()
        observer_class_mock_no_watch.assert_not_called() # Observer class never instantiated
        # Check other cleanup actions still happen
        self.mock_candles_instance.stop.assert_called_once() # This will fail if not reset for this model
        self.mock_mqtt_client_instance.disconnect.assert_called_once() # This will fail if not reset
        
        observer_class_mock_no_watch.stop()
        self.patch_observer.start() # Restart global patch
        # Reset mocks for other tests if they were called by model_no_watch
        self.mock_candles_instance.reset_mock()
        self.mock_mqtt_client_instance.reset_mock()

    def test_reload_handler(self):
        callback_mock = MagicMock()
        handler = ReloadHandler(callback_mock)
        event_mock = MagicMock()
        event_mock.is_directory = False
        event_mock.src_path = "some/file.py"
        handler.on_modified(event_mock)
        callback_mock.assert_called_once()

if __name__ == "__main__":
    unittest.main() 