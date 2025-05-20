import asyncio
import argparse
import logging
import os

# Ensure the root of the project is in PYTHONPATH for imports like `core.` to work
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True # Override any existing logger configurations
    )
    # Specific log level for noisy libraries if needed
    logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.INFO) # Can be WARNING if too noisy

def parse_args():
    parser = argparse.ArgumentParser(description='Run prediction services from configuration')
    parser.add_argument('--config',
                       default='config/prediction_services.yml', # Default config file
                       help='Path to prediction services configuration file (e.g., config/prediction_services.yml)')
    return parser.parse_args()

async def main():
    setup_logging()
    args = parse_args()
    
    # Import runner class after path adjustments and logging setup
    from core.prediction_service_runner import PredictionServiceRunner

    runner = PredictionServiceRunner(config_path=args.config)
    try:
        await runner.run()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, shutting down services...")
        # The runner's signal handler should already trigger shutdown,
        # but we can call it explicitly if needed, though it might lead to double calls.
        # await runner.shutdown_all_services() 
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}", exc_info=True)
        # Ensure cleanup on unexpected errors
        if hasattr(runner, 'shutdown_all_services') and runner.active_models:
            logging.info("Attempting emergency shutdown of services due to unhandled exception...")
            await runner.shutdown_all_services()
    finally:
        logging.info("Prediction service main finished.")

if __name__ == "__main__":
    # For ASYNC_DEBUG, see https://docs.python.org/3/library/asyncio-dev.html#debug-mode
    debug_asyncio = os.environ.get("ASYNC_DEBUG", "0") == "1"
    asyncio.run(main(), debug=debug_asyncio) 