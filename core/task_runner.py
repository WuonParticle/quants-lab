import importlib
import logging
import os
from datetime import timedelta
from typing import Dict, Any, List

import yaml
from dotenv import load_dotenv

from core.task_base import TaskOrchestrator, BaseTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskRunner:
    def __init__(self, config_path: str = "config/tasks.yml"):
        global logger
        logger = logging.getLogger(__name__)
        load_dotenv()
        # TODO: provide option to disable polling api utils class sys.modules['core.services.backend_api_client'] = None
        self.config_path = config_path
        self.orchestrator = TaskOrchestrator()
        self.tasks_config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load task configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def import_task_class(self, task_class_path: str) -> type:
        """Dynamically import task class from string path"""
        try:
            module_path, class_name = task_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Error importing task class {task_class_path}: {e}")
            raise

    def get_common_config(self) -> Dict[str, Any]:
        """Get common configuration from environment variables"""
        return {
            "timescale_config": {
                "host": os.getenv("TIMESCALE_HOST", "localhost"),
                "port": int(os.getenv("TIMESCALE_PORT", "5432")),
                "user": os.getenv("TIMESCALE_USER", "admin"),
                "password": os.getenv("TIMESCALE_PASSWORD", "admin"),
                "database": os.getenv("TIMESCALE_DB", "timescaledb")
            },
            "postgres_config": {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "user": os.getenv("POSTGRES_USER", "admin"),
                "password": os.getenv("POSTGRES_PASSWORD", "admin"),
                "database": os.getenv("POSTGRES_DB", "optimization_database")
            },
            "mongo_config": {
                "uri": os.getenv("MONGO_URI"),
                "db": os.getenv("MONGO_DB")
            }
        }

    def initialize_tasks(self) -> List[BaseTask]:
        """Initialize all enabled tasks from configuration"""
        tasks = []
        common_config = self.get_common_config()

        for task_name, task_config in self.tasks_config["tasks"].items():
            if not task_config.get("enabled", True):
                logger.info(f"Skipping disabled task: {task_name}")
                continue

            try:
                # Import task class
                task_class = self.import_task_class(task_config["task_class"])
                
                # Merge common config with task-specific config
                config = {**common_config, **task_config.get("config", {})}
                
                # Get number of parallel processes (default to 1 if not specified)
                num_parallel_processes = task_config.get("num_parallel_processes", 1)
                logger.info(f"Task {task_name} will run with {num_parallel_processes} parallel processes")
                
                # Create task instance
                task = task_class(
                    name=task_name,
                    frequency=timedelta(hours=task_config["frequency_hours"]) if task_config.get("frequency_hours") is not None else None,
                    config=config
                )
                task.num_parallel_processes = num_parallel_processes
                tasks.append(task)
                logger.info(f"Initialized task: {task_name}")

            except Exception as e:
                logger.error(f"Error initializing task {task_name}: {e}")
                continue

        return tasks

    async def run(self):
        """Run all configured tasks"""
        try:
            tasks = self.initialize_tasks()
            for task in tasks:
                self.orchestrator.add_task(task)
            
            logger.info(f"Starting orchestrator with {len(tasks)} tasks")
            await self.orchestrator.run()

        except Exception as e:
            logger.error(f"Error running tasks: {e}")
            raise