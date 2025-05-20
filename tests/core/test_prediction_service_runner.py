import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call
import yaml
import signal # Import the signal module
import paho.mqtt.client as mqtt # Added for spec

# Ensure core and services are importable
import sys
PROJECT_ROOT_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT_TEST not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_TEST)

import core.prediction_service_runner # Added for direct patching
from core.prediction_service_runner import PredictionServiceRunner
from core.prediction_model_base import BasePredictionModel # Needed for issubclass check

# A mock model class for the runner to instantiate
class ValidMockModelForRunner(BasePredictionModel):
    def __init__(self, *args, **kwargs):
        # Call super() but ensure critical mocks are in place if BasePredictionModel does a lot in init
        # For this test, BasePredictionModel's init will be heavily mocked by the test runner's setup
        self.init_args = args
        self.init_kwargs = kwargs
        self.run_called_flag = False
        self.stop_called_flag = False
        self.cleanup_called_flag = False
        # Minimal super call, or mock out its dependencies if it tries to connect/load
        # For these tests, PredictionServiceRunner.setUp will patch MQTT, Candles, Observer
        super().__init__(*args, **kwargs)

    async def load_model_artifacts(self): pass
    async def generate_features(self, dfs): return None
    async def predict(self, feats): return None
    async def run(self): self.run_called_flag = True; await asyncio.sleep(0.5)
    async def stop(self): self.stop_called_flag = True; await asyncio.sleep(0.01)
    async def cleanup(self): self.cleanup_called_flag = True; await asyncio.sleep(0.01)

class NonBaseModel:
    pass

class TestPredictionServiceRunner(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.config_path = "test_prediction_services_runner.yml"
        self.mock_config_data = {
            "common_mqtt_config": {"mqtt_broker": "test_broker"},
            "default_prediction_interval": 2.0,
            "services": {
                "service1": {
                    "enabled": True,
                    "model_class": "tests.core.test_prediction_service_runner.ValidMockModelForRunner",
                    "candles_config": [{"connector": "binance", "trading_pair": "BTC-USDT", "interval": "1m"}],
                    "mqtt_topic_prefix": "hbot/s1",
                    "watched_files": ["file1.py"],
                    "config": {"key": "value"}
                },
                "service2_disabled": {
                    "enabled": False,
                    "model_class": "tests.core.test_prediction_service_runner.ValidMockModelForRunner",
                    "candles_config": [{"connector": "kucoin", "trading_pair": "ETH-USDT", "interval": "5m"}],
                    "mqtt_topic_prefix": "hbot/s2",
                },
                "service3_invalid_class": {
                    "enabled": True,
                    "model_class": "tests.core.test_prediction_service_runner.NonBaseModel",
                    "candles_config": [{"connector": "gateio", "trading_pair": "LTC-USDT", "interval": "1h"}],
                    "mqtt_topic_prefix": "hbot/s3",
                }
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.mock_config_data, f)

        # --- Patches for dependencies of BasePredictionModel --- 
        # These are needed because when PredictionServiceRunner instantiates a model class
        # (like ValidMockModelForRunner), the BasePredictionModel.__init__ will be called.
        self.mock_mqtt_client_instance = MagicMock(spec=mqtt.Client)
        self.mock_mqtt_client_instance.is_connected.return_value = True
        # Patch where 'Client' is looked up in core.prediction_model_base (after 'import paho.mqtt.client as mqtt')
        self.patch_mqtt_client = patch("core.prediction_model_base.mqtt.Client", return_value=self.mock_mqtt_client_instance)
        self.mock_mqtt_client_class_patcher = self.patch_mqtt_client.start()

        self.mock_candles_instance = MagicMock()
        self.mock_candles_instance.ready = True
        self.mock_candles_instance.candles_df = MagicMock() # Basic mock for df
        self.mock_candles_instance.start = MagicMock()
        self.mock_candles_instance.stop = AsyncMock()
        # Patch where 'CandlesFactory.get_candle' is looked up in core.prediction_model_base
        self.patch_candles_factory = patch("core.prediction_model_base.CandlesFactory.get_candle", return_value=self.mock_candles_instance)
        self.mock_candles_factory_get_candle_patcher = self.patch_candles_factory.start()

        self.mock_observer_instance = MagicMock()
        # Patch where 'Observer' is looked up in core.prediction_model_base
        self.patch_observer = patch("core.prediction_model_base.Observer", return_value=self.mock_observer_instance)
        self.mock_observer_class_patcher = self.patch_observer.start()
        # --- End Patches for BasePredictionModel dependencies --- 

        self.module_patcher = patch('importlib.import_module')
        self.mock_import_module = self.module_patcher.start()

        def side_effect_import_module(module_path):
            mock_module = MagicMock()
            if module_path == "tests.core.test_prediction_service_runner":
                mock_module.ValidMockModelForRunner = ValidMockModelForRunner
                mock_module.NonBaseModel = NonBaseModel
            else:
                # Fallback for any other unexpected imports
                mock_module.SomeOtherClass = MagicMock()
            return mock_module
        
        self.mock_import_module.side_effect = side_effect_import_module
        self.runner = PredictionServiceRunner(config_path=self.config_path)

    def tearDown(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        self.module_patcher.stop()
        self.patch_mqtt_client.stop()
        self.patch_candles_factory.stop()
        self.patch_observer.stop()

    async def test_load_config(self):
        self.assertEqual(self.runner.global_config["common_mqtt_config"]["mqtt_broker"], "test_broker")
        self.assertIn("service1", self.runner.services_config)

    async def test_initialize_services(self):
        await self.runner.initialize_services()
        self.assertEqual(len(self.runner.active_models), 1) 
        active_model = self.runner.active_models[0]
        self.assertIsInstance(active_model, ValidMockModelForRunner)
        self.assertEqual(active_model.model_name, "service1")
        self.assertEqual(active_model.mqtt_broker, "test_broker") 
        self.assertEqual(active_model.mqtt_topic_prefix, "hbot/s1")
        self.assertEqual(active_model.model_specific_config["key"], "value")

    async def test_run_all_services_and_shutdown_via_event(self):
        await self.runner.initialize_services()
        self.assertEqual(len(self.runner.active_models), 1)
        real_model_instance = self.runner.active_models[0]

        # Spy on the methods of the actual instance using AsyncMock
        # because the original methods are async in ValidMockModelForRunner
        real_model_instance.run = AsyncMock(wraps=real_model_instance.run) 
        real_model_instance.stop = AsyncMock(wraps=real_model_instance.stop)
        real_model_instance.cleanup = AsyncMock(wraps=real_model_instance.cleanup)

        run_task = asyncio.create_task(self.runner.run_all_services())
        await asyncio.sleep(0.1) # Let services start
        
        self.runner._stop_event.set() # Trigger stop
        await run_task 

        real_model_instance.run.assert_called_once()
        real_model_instance.stop.assert_called_once()
        real_model_instance.cleanup.assert_called_once()
        self.assertEqual(len(self.runner.active_models), 0)

    async def test_signal_handler_sets_stop_event(self):
        self.assertFalse(self.runner._stop_event.is_set())
        self.runner._signal_handler(signal.SIGINT) # Use imported signal
        self.assertTrue(self.runner._stop_event.is_set())

    async def test_run_main_flow_with_signal_shutdown(self):
        with patch.object(core.prediction_service_runner.asyncio, 'get_event_loop') as mock_get_loop:
            mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
            mock_get_loop.return_value = mock_loop

            run_main_task = asyncio.create_task(self.runner.run())
            await asyncio.sleep(0.2) # Allow init and services to start
            
            mock_get_loop.assert_called() # Check if the patched get_event_loop was called

            self.assertEqual(len(self.runner.active_models), 1)
            active_model_instance = self.runner.active_models[0]
            self.assertTrue(active_model_instance.run_called_flag)

            # Simulate receiving a signal by calling the handler
            self.runner._signal_handler(signal.SIGTERM)
            await run_main_task # Wait for the main run task to complete shutdown

            self.assertTrue(active_model_instance.stop_called_flag)
            self.assertTrue(active_model_instance.cleanup_called_flag)
            self.assertEqual(len(self.runner.active_models), 0)
            mock_loop.add_signal_handler.assert_any_call(signal.SIGINT, self.runner._signal_handler, signal.SIGINT)
            mock_loop.add_signal_handler.assert_any_call(signal.SIGTERM, self.runner._signal_handler, signal.SIGTERM)

#    async def test_run_adds_signal_handlers(self): # New focused test
#        with patch('core.prediction_service_runner.asyncio.get_event_loop') as mock_get_loop, \
#             patch.object(self.runner, 'initialize_services', new_callable=AsyncMock) as mock_init:
#            
#            mock_loop = MagicMock()
#            mock_get_loop.return_value = mock_loop
#            
#            class StopRun(Exception): pass
#            mock_init.side_effect = StopRun
#
#            with self.assertRaises(StopRun):
#                await self.runner.run() # Directly await
#            
#            # print(f"DEBUG (new test): mock_loop.method_calls: {mock_loop.method_calls}")
#            mock_loop.add_signal_handler.assert_any_call(signal.SIGINT, self.runner._signal_handler, signal.SIGINT)
#            mock_loop.add_signal_handler.assert_any_call(signal.SIGTERM, self.runner._signal_handler, signal.SIGTERM)

if __name__ == "__main__":
    unittest.main() 