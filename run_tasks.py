import asyncio
import argparse
import logging.config
import yaml
import os
import re
from pathlib import Path
from core.task_runner import TaskRunner

def setup_logging():
    log_config_path = 'config/logging_format.yml'
    if os.path.exists(log_config_path):
        with open(log_config_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

def parse_args():
    parser = argparse.ArgumentParser(description='Run tasks from configuration')
    parser.add_argument('--config', 
                       default='config/tasks.yml',
                       help='Path to tasks configuration file')
    return parser.parse_args()

async def main():
    setup_logging()
    args = parse_args()
    runner = TaskRunner(config_path=args.config)
    await runner.run()

if __name__ == "__main__":
    asyncio.run(main()) 