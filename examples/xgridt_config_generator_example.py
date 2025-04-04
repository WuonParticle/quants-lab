import asyncio
import os
from datetime import timedelta

from tasks.backtesting.xgridt_config_generator_task import XGridTConfigGeneratorTask

"""
Example script for running the XGridT configuration generator task.

This script generates optimized XGridT configurations for multiple trading pairs,
focusing on high trading volume while ensuring profitability after fees.

To run this script:
1. Make sure you have activated the quants-lab conda environment
2. Run: python examples/xgridt_config_generator_example.py
"""

async def main():
    # Define configuration parameters
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        "connector_name": "okx",
        "lookback_days": 30,  # Use 30 days of historical data
        "end_time_buffer_hours": 6,  # Exclude the most recent 6 hours
        "n_trials": 100,  # Number of optimization trials per trading pair
        "output_dir": "configs/xgridt",  # Directory to save configuration files
        
        # Trading pairs to optimize
        "selected_pairs": [
            "ADA-USDT",  # Cardano
            "BTC-USDT",  # Bitcoin
            "ETH-USDT",  # Ethereum
            "SOL-USDT",  # Solana
            "XRP-USDT",  # Ripple
        ],
        
        # Optimization parameters
        "fee_pct": 0.1,  # Fee percentage (0.1%)
        "target_min_volume": 500,  # Target minimum trading volume
        "profit_weight": 1.0,  # Weight for profit in objective function
        "volume_weight": 2.0,  # Weight for volume in objective function (higher priority)
    }

    # Create and run the task
    task = XGridTConfigGeneratorTask("XGridTConfigGenerator", timedelta(hours=24), config)
    await task.execute()
    
    print("\nTask completed successfully!")
    print("Check the output directory for optimized configuration files.")


if __name__ == "__main__":
    asyncio.run(main()) 