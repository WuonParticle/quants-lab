import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import logging
import traceback
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTask(ABC):
    def __init__(self, name: str, frequency: Optional[timedelta], config: Dict[str, Any]):
        self.name = name
        self.frequency = frequency
        self.config = config
        self.last_run = None

    @abstractmethod
    async def execute(self):
        pass

    async def run_once(self):
        try:
            await self.execute()
        except Exception as e:
            logger.error(f"Error executing task {self.name}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Sometimes logger is not working, so print to the console as well
            print(f"Error executing task {self.name}: {e}")
            print(f"Full traceback: {traceback.format_exc()}")

    async def run_with_frequency(self):
        while True:
            now = datetime.now()
            if self.last_run is None or (now - self.last_run) >= self.frequency:
                try:
                    self.last_run = now
                    await self.execute()
                except Exception as e:
                    logger.error(f"Error executing task {self.name}: {e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
            if self.frequency is None:
                return # if frequency is None, exit to support debugging
            await asyncio.sleep(1)  # Check every second

    @staticmethod
    def get_common_config() -> Dict[str, Any]:
        """Get common configuration defaults from environment variables"""
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

    @staticmethod
    def load_single_task_config() -> Dict[str, Any]:
        """
        Parse command line arguments to load configuration from YAML file and merge with common config.
        Useful for debugging a single task from the command line, and for giving LLM agents a way to run a task. 
        
        Returns:
            Dict containing the merged configuration
        """
        import yaml
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
        args = parser.parse_args()
        
        with open(args.config, 'r') as f:
            config_file = yaml.safe_load(f)
        
        task_config = next(iter(config_file.get("tasks").values())).get("config")
        common_config = BaseTask.get_common_config()
        
        # Merge common config with task config, with task config taking precedence
        return {**common_config, **task_config}


class TaskOrchestrator:
    def __init__(self):
        self.tasks = []

    def add_task(self, task: BaseTask):
        self.tasks.append(task)

    async def run(self):
        task_coroutines = [task.run_with_frequency() for task in self.tasks]
        await asyncio.gather(*task_coroutines)
