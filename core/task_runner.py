import importlib
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, List

import yaml
from dotenv import load_dotenv
import hummingbot

from core.task_base import TaskOrchestrator, BaseTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskRunner:
    def __init__(self, config_path: str = "config/tasks.yml"):
        load_dotenv()
        # TODO: provide option to disable polling api utils class sys.modules['core.services.backend_api_client'] = None
        self.config_path = config_path
        self.orchestrator = TaskOrchestrator()
        self.global_config = self.load_config()
        self.tasks_config = self.global_config.get("tasks", {})
        self.run_sequentially = self.global_config.get("run_sequentially", False)
        self.global_frequency_hours = self.global_config.get("frequency_hours")

    def load_config(self) -> Dict[str, Any]:
        """Load task configuration from YAML file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Try adding "config/" prefix if the original path doesn't exist
        config_prefixed_path = os.path.join("config", self.config_path)
        if os.path.exists(config_prefixed_path):
            with open(config_prefixed_path, 'r') as f:
                return yaml.safe_load(f)
           
        # TODO: only check if extension is missing
        with_yml_path = config_prefixed_path + ".yml"
        if os.path.exists(with_yml_path):
            with open(with_yml_path, 'r') as f:
                return yaml.safe_load(f)
        
        raise FileNotFoundError(f"Config file not found at {self.config_path} or {config_prefixed_path} or {with_yml_path}")
    

    def import_task_class(self, task_class_path: str) -> type:
        """Dynamically import task class from string path"""
        try:
            module_path, class_name = task_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Error importing task class {task_class_path}: {e}")
            raise

    def initialize_tasks(self) -> List[BaseTask]:
        """Initialize all enabled tasks from configuration"""
        tasks = []
        common_config = BaseTask.get_common_config()

        self.initialize_hummingbot_client_config()
        self.enable_vpn_compatibility()

        global_task_class_path = self.global_config.get("task_class")
        global_config_values = self.global_config.get("config", {})
        for task_name, task_config in self.tasks_config.items():
            if not task_config.get("enabled", True):
                logger.info(f"Skipping disabled task: {task_name}")
                continue

            try:
                # Determine task_class: task-specific or global
                task_class_path = task_config.get("task_class", global_task_class_path)
                if not task_class_path:
                    logger.error(f"Task class not defined for task {task_name} and no global task_class is set.")
                    continue
                task_class = self.import_task_class(task_class_path)
                
                # Merge common_config, global_config_values, and task_specific_config
                # Order of precedence: task_specific > global_config_values > common_config
                config = {**common_config, **global_config_values, **task_config.get("config", {})}
                
                # TODO: implement support for task_groups so that  
                # 1. multiple groups of sequential tasks can be run in parallel
                # 2. multiple groups of parallel tasks can be run in sequence
                frequency_hours = task_config.get("frequency_hours")
                if self.run_sequentially and frequency_hours is not None:
                    logger.warning(f"Task {task_name} has frequency_hours defined but run_sequentially is true. Task-level frequency will be ignored.")
                
                current_frequency = None
                if self.run_sequentially:
                    if self.global_frequency_hours is not None:
                        current_frequency = timedelta(hours=self.global_frequency_hours)
                elif frequency_hours is not None: # run in parallel
                    current_frequency = timedelta(hours=frequency_hours)

                # Create task instance
                task = task_class(
                    name=task_name,
                    frequency=current_frequency,
                    config=config
                )
                tasks.append(task)
                logger.info(f"Initialized task: {task_name}")

            except Exception as e:
                import traceback
                logger.error(f"Error initializing task {task_name}: {e}")
                logger.error(f"Stacktrace: {traceback.format_exc()}")
                continue

        return tasks

    def enable_vpn_compatibility(self):
        # TODO: only enable if use_vpn is on (needs to be passed to run_tasks first)
        # Disable TLS 1.3 to avoid vpn issues
        from hummingbot.core.web_assistant.connections.connections_factory import ConnectionsFactory
        ConnectionsFactory().set_disable_tls_1_3(disable=True)

    def initialize_hummingbot_client_config(self):
        config_password = os.getenv("HUMMGINGBOT_CONFIG_PASSWORD")
        source_path = os.getenv("HUMMGINGBOT_SOURCE_PATH")
        if source_path is not None:
            path = Path(source_path)
            if not path.exists():
                logger.warning(f"Source path {source_path} does not exist, using default root path")
            else:
                try:
                    # This is the easiest way because as soon as other modules are imported, they set a large number of properties
                    hummingbot.root_path = lambda: path
                    if config_password is not None:
                        from hummingbot.client.config.config_crypt import ETHKeyFileSecretManger
                        from hummingbot.client.config.security import Security
                        secrets_manager = ETHKeyFileSecretManger(config_password)
                        Security.login(secrets_manager)
                        # TODO check if the await Security.wait_til_decryption_done() is needed

                    from hummingbot.client.config.config_helpers import load_client_config_map_from_file
                    _ = load_client_config_map_from_file()
                except Exception as e:
                    logger.warning(f"Error loading client config map from file: {e} continuing without custom config")

    async def run(self):
        """Run all configured tasks"""
        try:
            tasks = self.initialize_tasks()
            if not tasks:
                logger.info("No tasks to run.")
                return

            if self.run_sequentially:
                if self.global_frequency_hours is None:
                    logger.warning("Running tasks sequentially but no global frequency_hours is set. Tasks will run once in order.")
                await self.orchestrator.run_sequentially(tasks, timedelta(hours=self.global_frequency_hours) if self.global_frequency_hours is not None else None)
            else:
                for task in tasks:
                    self.orchestrator.add_task(task)
                logger.info(f"Starting orchestrator with {len(tasks)} tasks in parallel.")
                await self.orchestrator.run()

        except Exception as e:
            logger.error(f"Error running tasks: {e}")
            raise