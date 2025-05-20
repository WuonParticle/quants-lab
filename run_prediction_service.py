import asyncio
import logging.config
import argparse
from pathlib import Path
import os # Make sure os is imported

from services.prediction_service.prediction_service_runner import PredictionServiceRunner

# It's good practice to set up logging as early as possible.
# Basic config for the runner script itself.
# The runner and model base classes also configure their loggers.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run Prediction Service')
    # Construct the default path relative to this script's location
    # Assumes run_prediction_service.py is at the project root
    # and the config is at services/prediction_service/config/prediction_services.yml
    default_config = Path(__file__).parent / "services" / "prediction_service" / "config" / "prediction_services.yml"
    parser.add_argument('--config',
                        default=str(default_config.resolve()),
                        help='Path to prediction services configuration file')
    return parser.parse_args()

async def main():
    args = parse_args()
    logger.info(f"Starting Prediction Service Runner with config: {args.config}")
    runner = PredictionServiceRunner(config_path=args.config)
    await runner.run()

if __name__ == "__main__":
    # Set ASYNC_DEBUG for more verbose asyncio logging if needed
    # debug = os.environ.get("ASYNC_DEBUG", False)
    # asyncio.run(main(), debug=debug)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("run_prediction_service.py: KeyboardInterrupt received, exiting.")
    except Exception as e:
        logger.exception(f"run_prediction_service.py: An unhandled exception occurred: {e}")
    finally:
        logger.info("run_prediction_service.py: Exited.") 