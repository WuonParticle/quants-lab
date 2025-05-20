import asyncio
import importlib
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Type

import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from services.prediction_service.prediction_model_base import PredictionModelBase

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelFileChangeHandler(FileSystemEventHandler):
    def __init__(self, runner, model_name: str):
        super().__init__()
        self.runner = runner
        self.model_name = model_name
        self._last_event_time = 0
        self._debounce_period = 2 # seconds to wait for more events

    def on_modified(self, event: FileModifiedEvent):
        # This can fire multiple times for a single save, debounce it.
        current_time = time.time()
        if current_time - self._last_event_time < self._debounce_period:
            return
        self._last_event_time = current_time
        
        if not event.is_directory:
            logger.info(f"Detected modification in {event.src_path} for model {self.model_name}. Triggering reload.")
            # Schedule the reload in the runner's event loop
            asyncio.run_coroutine_threadsafe(self.runner.reload_model(self.model_name), self.runner.loop)

class PredictionServiceRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config()
        self.models: Dict[str, PredictionModelBase] = {}
        self.file_observers: Dict[str, Observer] = {}
        self.loop = asyncio.get_event_loop()

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if not config_data or "prediction_models" not in config_data:
                    logger.error("Config file must contain a 'prediction_models' list.")
                    sys.exit(1)
                logger.info(f"Configuration loaded successfully from {self.config_path}")
                return config_data
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            sys.exit(1)

    def _import_model_class(self, class_path: str) -> Type[PredictionModelBase]:
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            if not issubclass(model_class, PredictionModelBase):
                raise TypeError(f"Class {class_path} must be a subclass of PredictionModelBase")
            return model_class
        except (ImportError, AttributeError, ValueError, TypeError) as e:
            logger.error(f"Error importing model class {class_path}: {e}")
            raise

    async def initialize_models(self):
        model_configs = self.config.get("prediction_models", [])
        if not model_configs:
            logger.warning("No prediction models defined in the configuration.")
            return

        for mc in model_configs:
            model_name = mc.get("name")
            if not model_name:
                logger.error("Model configuration missing 'name'. Skipping.")
                continue
            if model_name in self.models:
                logger.warning(f"Model with name '{model_name}' already initialized. Skipping duplicate.")
                continue
            
            logger.info(f"Initializing model: {model_name}")
            try:
                model_class_path = mc["model_class"]
                ModelClass = self._import_model_class(model_class_path)
                
                # Prepare CandlesConfig instances
                parsed_candles_configs = []
                for cc_dict in mc.get("candles_configs", []):
                    parsed_candles_configs.append(CandlesConfig(**cc_dict))
                
                model_instance = ModelClass(
                    model_name=model_name,
                    candles_configs=parsed_candles_configs,
                    mqtt_broker=mc.get("mqtt_broker", "localhost"),
                    mqtt_port=mc.get("mqtt_port", 1883),
                    mqtt_topic_prefix=mc["mqtt_topic_prefix"],
                    mqtt_qos=mc.get("mqtt_qos", 1),
                    mqtt_retain=mc.get("mqtt_retain", True),
                    model_specific_config=mc.get("model_specific_config", {}),
                    watched_files=mc.get("watched_files", [])
                )
                self.models[model_name] = model_instance
                logger.info(f"Model '{model_name}' initialized with class {model_class_path}.")

                # Setup file watcher for this model if watched_files are specified
                watched_files = mc.get("watched_files", [])
                if watched_files:
                    self._setup_file_watcher(model_name, watched_files)

            except KeyError as e:
                logger.error(f"Missing required key '{e}' in configuration for model '{model_name}'. Skipping.")
            except Exception as e:
                logger.exception(f"Error initializing model '{model_name}': {e}. Skipping.")

    def _setup_file_watcher(self, model_name: str, watched_files: List[str]):
        if not watched_files:
            return

        observer = Observer()
        event_handler = ModelFileChangeHandler(self, model_name)
        
        # Ensure watched paths are absolute and directories containing them are watched
        watched_paths_absolute = set()
        for file_path_str in watched_files:
            file_path = Path(file_path_str).resolve()
            if not file_path.parent.exists():
                logger.warning(f"Directory for watched file {file_path} does not exist. Cannot watch.")
                continue
            # Watch the directory, filtering will happen in the handler
            # Or, watch individual files if supported and preferred, but directory watching is more robust for new/atomic saves
            watched_paths_absolute.add(str(file_path.parent)) 
            logger.info(f"[{model_name}] Watching for changes in directory of {file_path} for reload.")

        # To avoid watching the same directory multiple times if multiple files are in it
        unique_dirs_to_watch = set()
        for f_path_str in watched_files:
            p = Path(f_path_str).resolve()
            if p.exists(): # Watch only if file exists, parent dir otherwise for creation
                 unique_dirs_to_watch.add(str(p.parent))
            elif p.parent.exists():
                 unique_dirs_to_watch.add(str(p.parent))
            else:
                logger.warning(f"Cannot watch {f_path_str}: neither file nor its parent directory exists.") 

        for directory_to_watch in unique_dirs_to_watch:
            try:
                observer.schedule(event_handler, directory_to_watch, recursive=False) # Non-recursive, check files in this dir
                logger.info(f"[{model_name}] Watching directory {directory_to_watch} for file changes.")
            except Exception as e:
                 logger.error(f"[{model_name}] Failed to schedule watcher for {directory_to_watch}: {e}")

        if observer.emitters: # Check if any emitters were actually scheduled
            observer.start()
            self.file_observers[model_name] = observer
            logger.info(f"File watcher started for model '{model_name}' on paths: {unique_dirs_to_watch}")
        else:
            logger.warning(f"[{model_name}] No valid paths found to watch. File watcher not started.")

    async def start_models(self):
        if not self.models:
            logger.info("No models to start.")
            return
        start_tasks = [model.start() for model_name, model in self.models.items() if not model.is_running()]
        if start_tasks:
            await asyncio.gather(*start_tasks)
            logger.info(f"Started {len(start_tasks)} models.")
        else:
            logger.info("All configured models are already running or no models to start.")

    async def stop_models(self):
        logger.info("Stopping all models...")
        stop_tasks = [model.stop() for model_name, model in self.models.items() if model.is_running()]
        if stop_tasks:
            await asyncio.gather(*stop_tasks)
        logger.info("All models stopped.")
        for observer in self.file_observers.values():
            observer.stop()
            observer.join()
        logger.info("All file observers stopped.")

    async def reload_model(self, model_name: str):
        model = self.models.get(model_name)
        if model:
            logger.info(f"Reloading model: {model_name}")
            await model.reload()
        else:
            logger.warning(f"Attempted to reload non-existent model: {model_name}")

    async def run(self):
        # Handle signals for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))

        await self.initialize_models()
        await self.start_models()
        
        # Keep the runner alive
        try:
            while True:
                # If models can stop themselves (e.g. due to error limits), 
                # this loop could check and try to restart them based on policy.
                # For now, it just keeps the main runner alive.
                await asyncio.sleep(3600) # Wake up periodically
        except asyncio.CancelledError:
            logger.info("PredictionServiceRunner main loop cancelled.")
        finally:
            # This might be redundant if shutdown is called via signal
            if any(model.is_running() for model in self.models.values()):
                 await self.stop_models()

    async def shutdown(self, sig: Optional[signal.Signals] = None):
        if sig:
            logger.info(f"Received shutdown signal: {sig.name}. Shutting down...")
        else:
            logger.info("Shutting down PredictionServiceRunner...")
        
        # Stop models first
        await self.stop_models()

        # Cancel all other tasks in the loop
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        logger.info(f"Cancelling {len(tasks)} outstanding tasks.")
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.loop.stop()
        logger.info("PredictionServiceRunner shut down gracefully.")


if __name__ == "__main__":
    # This is a placeholder for where you'd create the config file path
    # and instantiate and run the service.
    # Example: config/prediction_services.yml
    default_config_path = Path(__file__).parent / "config" / "prediction_services.yml"
    
    # Ensure the config directory and a default config file exist for the example
    config_dir = default_config_path.parent
    if not config_dir.exists():
        config_dir.mkdir(parents=True)
    
    if not default_config_path.exists():
        logger.warning(f"Default config {default_config_path} not found. Creating a sample.")
        sample_config = {
            "prediction_models": [
                {
                    "name": "sample_model_1",
                    "model_class": "services.prediction_service.prediction_model_base.PredictionModelBase", # Replace with actual model
                    "watched_files": ["path/to/your/model.pkl", "path/to/your/scaler.joblib"],
                    "candles_configs": [
                        {"connector": "binance", "trading_pair": "BTC-USDT", "interval": "1m", "max_records": 200}
                    ],
                    "mqtt_broker": "localhost",
                    "mqtt_port": 1883,
                    "mqtt_topic_prefix": "hbot/dev_predictions",
                    "model_specific_config": {
                        "some_param": "value",
                        "prediction_interval_sec": 1.0
                    }
                }
            ]
        }
        with open(default_config_path, 'w') as f:
            yaml.dump(sample_config, f, indent=2)
        logger.info(f"Sample config created at {default_config_path}. Please review and update it.")
        # For a real run, you might exit here or use a proper default model.
        # For this example, we'll just point out that the base class won't run well on its own.
        print(f"Please create a concrete subclass of PredictionModelBase and update {default_config_path}.")
        sys.exit(0)

    runner = PredictionServiceRunner(config_path=str(default_config_path))
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
        # The signal handler should take care of shutdown, but as a fallback:
        if not runner.loop.is_closed():
            asyncio.run_coroutine_threadsafe(runner.shutdown(), runner.loop)
            # Give it a moment to process before loop might be stopped externally
            time.sleep(2)
    finally:
        if not runner.loop.is_closed():
            runner.loop.close()
        logger.info("Runner loop closed.") 