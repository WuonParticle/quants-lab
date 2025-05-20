import asyncio
import importlib
import logging
import os
import signal
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from core.prediction_model_base import BasePredictionModel
from core.task_config_helpers import TaskConfigHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionServiceRunner:
    def __init__(self, config_path: str = "config/prediction_services.yml"):
        load_dotenv()
        self.global_config = TaskConfigHelper.load_config_file(config_path)
        self.services_config = self.global_config.get("services", {})
        self.active_models: List[BasePredictionModel] = []
        self._stop_event = asyncio.Event()

    def _import_model_class(self, class_path: str) -> type:
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            if not issubclass(model_class, BasePredictionModel):
                raise TypeError(f"Class {class_path} must be a subclass of BasePredictionModel")
            return model_class
        except (ImportError, AttributeError, TypeError) as e:
            logger.error(f"Error importing prediction model class {class_path}: {e}")
            raise

    async def initialize_services(self):
        common_mqtt_config = self.global_config.get("common_mqtt_config", {})
        default_prediction_interval = self.global_config.get("default_prediction_interval", 1.0)

        for service_name, service_conf in self.services_config.items():
            if not service_conf.get("enabled", True):
                logger.info(f"Skipping disabled prediction service: {service_name}")
                continue

            try:
                model_class_path = service_conf.get("model_class")
                if not model_class_path:
                    logger.error(f"'model_class' not defined for service {service_name}. Skipping.")
                    continue
                
                PredictionModelClass = self._import_model_class(model_class_path)

                candles_config_list = service_conf.get("candles_config", [])
                if not candles_config_list:
                    logger.error(f"'candles_config' not defined or empty for service {service_name}. Skipping.")
                    continue

                mqtt_topic_prefix = service_conf.get("mqtt_topic_prefix")
                if not mqtt_topic_prefix:
                    logger.error(f"'mqtt_topic_prefix' not defined for service {service_name}. Skipping.")
                    continue

                model_specific_config = service_conf.get("config", {})
                watched_files = service_conf.get("watched_files", [])
                prediction_interval = service_conf.get("prediction_interval", default_prediction_interval)

                # Merge common MQTT settings with service-specific, service-specific takes precedence
                current_mqtt_config = {**common_mqtt_config, **service_conf.get("mqtt_config", {})}
                
                model_instance = PredictionModelClass(
                    model_name=service_name,
                    model_class_path=model_class_path,
                    candles_config_list=candles_config_list,
                    mqtt_topic_prefix=mqtt_topic_prefix,
                    watched_files=watched_files,
                    model_specific_config=model_specific_config,
                    mqtt_broker=current_mqtt_config.get("mqtt_broker", "localhost"),
                    mqtt_port=current_mqtt_config.get("mqtt_port", 1883),
                    mqtt_qos=current_mqtt_config.get("mqtt_qos", 1),
                    mqtt_retain=current_mqtt_config.get("mqtt_retain", True),
                    prediction_interval=prediction_interval
                )
                self.active_models.append(model_instance)
                logger.info(f"Initialized prediction service: {service_name} with model {model_class_path}")

            except Exception as e:
                logger.error(f"Error initializing prediction service {service_name}: {e}", exc_info=True)
                continue
        
        if not self.active_models:
            logger.warning("No prediction services were initialized successfully.")

    async def run_all_services(self):
        if not self.active_models:
            logger.info("No active prediction models to run.")
            return

        logger.info(f"Starting {len(self.active_models)} prediction service(s)...")
        
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_handler, sig)

        tasks = [model.run() for model in self.active_models]
        # We run tasks and also wait for the stop_event
        # This allows external signals or calls to stop() to interrupt the run.
        runner_task = asyncio.gather(*tasks, return_exceptions=True)
        stop_event_task = asyncio.create_task(self._stop_event.wait()) # Wrap in a task
        await asyncio.wait([runner_task, stop_event_task], return_when=asyncio.FIRST_COMPLETED)
        
        if self._stop_event.is_set():
            logger.info("Stop event received, initiating graceful shutdown of services...")
        elif runner_task.done() and runner_task.exception():
            logger.error(f"One or more services failed: {runner_task.exception()}")
            # Potentially trigger shutdown or other error handling here

        await self.shutdown_all_services()

    def _signal_handler(self, sig):
        logger.info(f"Received signal {sig.name}, initiating shutdown...")
        self._stop_event.set()

    async def shutdown_all_services(self):
        logger.info("Shutting down all active prediction services...")
        # Stop all model loops first
        for model in self.active_models:
            asyncio.create_task(model.stop()) # schedule stop, don't await here
        
        # Then wait for all cleanup tasks to complete
        cleanup_tasks = [model.cleanup() for model in self.active_models if hasattr(model, 'cleanup')]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
        self.active_models = []
        logger.info("All prediction services have been shut down.")

    async def run(self):
        await self.initialize_services()
        await self.run_all_services() 