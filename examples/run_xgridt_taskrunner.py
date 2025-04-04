import os
import sys
import yaml
import logging
import argparse
import datetime
from run_xgridt_config_generator_simple import main as run_generator

"""
Wrapper script for the XGridT Config Generator to be used with the task runner.
This allows the generator to be run as a scheduled task.

Usage with task runner:
  make run-task config=xgridt_config_generator_tasks.yml
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(params=None):
    """
    Entry point for the task runner.
    """
    if params is None:
        # If run directly, parse from command line
        parser = argparse.ArgumentParser(description='Run XGridT Config Generator')
        parser.add_argument('--output-dir', dest='output_dir', default='config/xgridt',
                          help='Directory to save configuration files')
        parser.add_argument('--trading-pairs', dest='trading_pairs', nargs='+',
                          default=["ADA-USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"],
                          help='List of trading pairs to optimize')
        parser.add_argument('--fee-pct', dest='fee_pct', type=float, default=0.1,
                          help='Fee percentage')
        parser.add_argument('--n-trials', dest='n_trials', type=int, default=50,
                          help='Number of optimization trials per trading pair')
        args = parser.parse_args()
        
        # Convert args to params dict
        params = {
            'output_dir': args.output_dir,
            'trading_pairs': args.trading_pairs,
            'fee_pct': args.fee_pct,
            'n_trials': args.n_trials
        }
    
    # Log the parameters
    logger.info(f"Running XGridT Config Generator with parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    # Create output directory if it doesn't exist
    os.makedirs(params.get('output_dir', 'config/xgridt'), exist_ok=True)
    
    # Modify global variables in the generator script
    import examples.run_xgridt_config_generator_simple as generator
    generator.OUTPUT_DIR = params.get('output_dir', 'config/xgridt')
    generator.TRADING_PAIRS = params.get('trading_pairs', ["ADA-USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"])
    generator.FEE_PCT = params.get('fee_pct', 0.1)
    generator.N_TRIALS = params.get('n_trials', 50)
    
    # Run the generator
    logger.info("Starting XGridT Config Generator...")
    generator.main()
    logger.info("XGridT Config Generator completed successfully.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 