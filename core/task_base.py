import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import sys
from typing import Any, Dict, List, Optional
import logging
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os
import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Global variable to store the ProcessPoolExecutor
_executor = None

def signal_handler(sig, frame):
    """Handle keyboard interrupt by terminating all processes"""
    logger.info("Keyboard interrupt received. Shutting down all processes...")
    if _executor is not None:
        logger.info("Terminating process pool executor")
        _executor.shutdown(wait=False)
    sys.exit(0)

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


class BaseTask(ABC):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any], num_parallel_processes: int = 1):
        self.name = name
        self.frequency = frequency
        self.config = config
        self.last_run = None
        self.num_parallel_processes = num_parallel_processes
        self.process_id = 0  # Used to identify which process this task is running in

    @abstractmethod
    async def execute(self):
        pass

    async def run_with_frequency(self):
        logger.info(f"Starting frequency-based execution for task '{self.name}' (process {self.process_id})")
        while True:
            now = datetime.now()
            if self.last_run is None or (now - self.last_run) >= self.frequency:
                try:
                    self.last_run = now
                    logger.info(f"Executing task '{self.name}' (process {self.process_id}) at {now}")
                    await self.execute()
                    logger.info(f"Completed execution of task '{self.name}' (process {self.process_id})")
                except Exception as e:
                    logger.error(f"Error executing task '{self.name}' (process {self.process_id}): {e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
            if self.frequency is None:
                return # If frequency is None, run only once (useful for debugging)
            await asyncio.sleep(1)  # Check every second


def run_task_in_process(task_index, task):
    """Function to run a task in a separate process"""
    logger.info(f"Starting process {task_index + 1} for task '{task.name}'")
    # Set process ID for the task
    task.process_id = task_index + 1
    
    # Create a new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run the task
    try:
        logger.info(f"Process {task_index + 1}: Running task '{task.name}' with loop {id(loop)}")
        loop.run_until_complete(task.run_with_frequency())
    except Exception as e:
        logger.error(f"Error in process {task_index + 1} for task '{task.name}': {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"Closing loop for process {task_index + 1}, task '{task.name}'")
        loop.close()


class TaskOrchestrator:
    def __init__(self):
        self.tasks = []
        self.executor = None

    def add_task(self, task: BaseTask):
        logger.info(f"Adding task '{task.name}' to orchestrator")
        self.tasks.append(task)

    async def run(self):
        logger.info("Starting TaskOrchestrator")
        # Group tasks by their parallel process count
        task_groups = {}
        for task in self.tasks:
            if task.num_parallel_processes not in task_groups:
                task_groups[task.num_parallel_processes] = []
            task_groups[task.num_parallel_processes].append(task)
            logger.info(f"Grouped task '{task.name}' with {task.num_parallel_processes} parallel processes")
        
        # Process each group according to its parallel process count
        for num_processes, tasks in task_groups.items():
            if num_processes == 1:
                # Run single process tasks directly
                logger.info(f"Running {len(tasks)} tasks in single process mode")
                task_coroutines = [task.run_with_frequency() for task in tasks]
                await asyncio.gather(*task_coroutines)
            else:
                # Run tasks in parallel processes
                logger.info(f"Running {len(tasks)} tasks in {num_processes} parallel processes")
                await self._run_tasks_in_parallel(tasks, num_processes)
    
    async def _run_tasks_in_parallel(self, tasks: List[BaseTask], num_processes: int):
        """
        Run tasks in parallel processes using ProcessPoolExecutor.
        
        Args:
            tasks: List of tasks to run in parallel
            num_processes: Number of parallel processes to use
        """
        logger.info(f"Starting parallel execution with {num_processes} processes for {len(tasks)} tasks")
        # Use ProcessPoolExecutor to run tasks in parallel
        global _executor
        _executor = ProcessPoolExecutor(max_workers=num_processes)
        self.executor = _executor
        
        try:
            # Submit all tasks to the executor
            futures = []
            for i, task in enumerate(tasks):
                logger.info(f"Submitting task '{task.name}' to process pool")
                futures.append(_executor.submit(run_task_in_process, i, task))
            
            # Keep the executor running without waiting for task completion
            while True:
                await asyncio.sleep(60)  # Check status every minute
                # Check if any tasks have failed
                for i, future in enumerate(futures):
                    if future.done():
                        try:
                            future.result()  # This will raise any exceptions that occurred
                            logger.warning(f"Task {i+1} completed unexpectedly")
                        except Exception as e:
                            if not isinstance(e, TimeoutError):  # Ignore timeout errors
                                logger.error(f"Error in task process {i+1}: {str(e)}")
                                logger.error(traceback.format_exc())
        except asyncio.CancelledError:
            logger.info("Task orchestrator was cancelled, shutting down processes")
            self.shutdown()
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the executor and terminate all processes"""
        if self.executor is not None:
            logger.info("Shutting down process pool executor")
            self.executor.shutdown(wait=False)
            self.executor = None