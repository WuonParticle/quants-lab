import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import random
from typing import Any, Dict, Optional
import logging
import traceback
import os
import fcntl
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTask(ABC):
    def __init__(self, name: str, frequency: Optional[timedelta], config: Dict[str, Any]):
        self.name = name
        self.frequency = frequency
        self.config = config
        self.last_run = None
        
        # Initialize root_path
        self.root_path = Path(os.getenv("root_path") or config.get("root_path", Path.cwd()))
        
        # Create standard directories if they don't exist
        (self.root_path / "data").mkdir(parents=True, exist_ok=True)

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
                "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
                "database": os.getenv("MONGO_DB", "quants_lab")
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


class ParallelWorkerTask(BaseTask):
    """
    A task that operates in a parallel worker environment with leader election capabilities.
    Worker-1 is automatically designated as the leader.
    
    This class combines the functionality of parallel worker distribution and leadership.
    """
    def __init__(self, name: str,
                 config: Dict[str, Any],
                 frequency: Optional[timedelta] = None,
                 leader_head_start: Optional[float] = 0.5,
                 worker_start_stagger: Optional[float] = 0.05):
        super().__init__(name, frequency, config)
        # Initialize worker ID and total workers
        self._worker_id = None
        self._total_workers = None
        self.worker_start_stagger = worker_start_stagger
        self._initialize_worker_info()
        
        # Leader properties
        self.leader_head_start = leader_head_start
    
    def _initialize_worker_info(self):
        """Initialize worker ID and total workers from environment variables"""
        # Try to get worker ID from WORKER_ID environment variable first
        worker_id_str = os.environ.get("WORKER_ID", "0")
        if worker_id_str is not None and worker_id_str.isdigit():
            self._worker_id = int(worker_id_str)
        else:
            # As a fallback, try to get it from config (for debugging)
            self._worker_id = self.config.get("worker_id", 0)
            
        # Get total number of workers from environment variable
        self._total_workers = int(os.environ.get("TOTAL_WORKERS", "1"))
        
        # Log initialization info
        logger.info(f"Task {self.name} initialized as worker {self._worker_id+1}/{self._total_workers} (Leader: {self.is_leader})")
    
    @property
    def worker_id(self) -> int:
        """Get the worker ID (0-based index)"""
        return self._worker_id
    
    @property
    def total_workers(self) -> int:
        """Get the total number of workers"""
        return self._total_workers
    
    @property
    def is_leader(self) -> bool:
        """
        Determine if this instance is the leader.
        Worker-1, which has worker_id=0 internally, is always the leader.
        """
        return self._worker_id == 0
    
    def should_process_item(self, item_idx: int) -> bool:
        """
        Determine if this worker should process the given item based on its index
        
        Args:
            item_idx: The index of the item to check
            
        Returns:
            True if this worker should process the item, False otherwise
        """
        return item_idx % self.total_workers == self.worker_id
    
    async def execute(self):
        """
        Execute the task with leader election awareness.
        
        If this worker is not the leader, it will wait for a short time to give
        the leader a head start for any initialization tasks.
        """
        try:
            if not self.is_leader:
                # Give the leader a head start to do any initialization
                await asyncio.sleep(self.leader_head_start + self.worker_id * self.worker_start_stagger)
            # Execute the actual task implementation
            await self.task_execute()
        except Exception as e:
            logger.error(f"Error executing task {self.name}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    @abstractmethod
    async def task_execute(self):
        """
        The actual task implementation. Subclasses should implement this method
        instead of overriding execute().
        
        Use self.is_leader to check if this instance is the leader.
        """
        pass


class TaskOrchestrator:
    def __init__(self):
        self.tasks = []

    def add_task(self, task: BaseTask):
        self.tasks.append(task)

    async def run(self):
        task_coroutines = [task.run_with_frequency() for task in self.tasks]
        await asyncio.gather(*task_coroutines)
