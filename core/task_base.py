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


class LeaderElectedTask(BaseTask):
    """
    A task that implements leader election to ensure certain operations
    only happen once across multiple instances of the same task.
    
    Leader election is based on file locking. The task that successfully
    acquires the lock becomes the leader.
    """
    def __init__(self, name: str, frequency: Optional[timedelta], config: Dict[str, Any], leader_head_start: Optional[float] = 0.5):
        super().__init__(name, frequency, config)
        self.is_leader = False
        self.leader_lock_file = None
        self.lock_fd = None
        
        # Specify the lock file name or allow overriding via config
        self.lock_file_name = config.get("leader_lock_file", f"{name.replace(' ', '_').lower()}_leader.lock")
        self.leader_head_start = leader_head_start
    def _try_acquire_leadership(self) -> bool:
        """
        Try to acquire leadership by obtaining an exclusive lock on a leadership file.
        Returns True if leadership was acquired, False otherwise.
        """
        # Create leadership lock file path
        leadership_file = self.root_path / "data/locks" / self.lock_file_name
        
        try:
            # Create the lock file if it doesn't exist
            leadership_file.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(str(leadership_file), os.O_CREAT | os.O_WRONLY)
            
            # Try to acquire an exclusive lock
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Write process ID to file
            os.write(lock_fd, str(os.getpid()).encode())
            os.fsync(lock_fd)
            
            # Store the file descriptor to keep the lock
            self.lock_fd = lock_fd
            self.leader_lock_file = leadership_file
            self.is_leader = True
            logger.info(f"Leadership acquired by process {os.getpid()} for task {self.name}")
            return True
            
        except BlockingIOError:
            # Another process has the lock
            if 'lock_fd' in locals() and lock_fd is not None:
                os.close(lock_fd)
            logger.info(f"Another process is already the leader for task {self.name}")
            return False
        except Exception as e:
            logger.error(f"Error during leadership acquisition for task {self.name}: {str(e)}")
            if 'lock_fd' in locals() and lock_fd is not None:
                os.close(lock_fd)
            return False

    def _release_leadership(self):
        """Release the leadership lock if this process is the leader"""
        if self.is_leader and self.lock_fd is not None:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
                logger.info(f"Leadership released for task {self.name}")
            except Exception as e:
                logger.error(f"Error releasing leadership for task {self.name}: {str(e)}")
            finally:
                self.is_leader = False
                self.lock_fd = None
    
    async def execute(self):
        """
        Execute with leader election. This method:
        1. Tries to acquire leadership
        2. Calls the task_execute method (which subclasses must implement)
        3. Releases leadership when done
        """
        # Try to become the leader
        self._try_acquire_leadership()
        
        try:
            if not self.is_leader:
                # Give the leader a head start to do any initialization
                await asyncio.sleep(self.leader_head_start + random.uniform(0, self.leader_head_start))
            # Execute the actual task implementation
            await self.task_execute()
        finally:
            # Always release leadership at the end
            self._release_leadership()
    
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
