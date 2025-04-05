import asyncio
import datetime
import logging
import os
import time
from datetime import timedelta
from typing import Any, Dict, List
from decimal import Decimal

import pandas as pd
from dotenv import load_dotenv
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from hummingbot.strategy_v2.utils.distributions import Distributions

from controllers.directional_trading.xgridt import XGridTControllerConfig
from core.backtesting.optimizer import StrategyOptimizer, BacktestingConfig, BaseStrategyConfigGenerator
from core.task_base import BaseTask
from core.task_config_helpers import TaskConfigHelper
from core.services.timescale_client import TimescaleClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class XGridTConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for XGridT optimization.
    """
    async def generate_config(self, trial) -> BacktestingConfig:
        # Controller configuration
        connector_name = self.config.get("connector_name", "hyperliquid_perpetual")
        trading_pair = self.config.get("trading_pair", "PNUT-USDT")
        interval = self.config.get("interval", "1m")
        trial.set_user_attr("connector_name", connector_name)
        trial.set_user_attr("trading_pair", trading_pair)
        trial.set_user_attr("interval", interval)
        ema_short = trial.suggest_int("ema_short", 9, 59)
        ema_medium = trial.suggest_int("ema_medium", ema_short + 10, 150)
        ema_long = trial.suggest_int("ema_long", ema_medium + 10, 201)
        donchian_channel_length = trial.suggest_int("donchian_channel_length", 50, 200, step=50)
        natr_length = 100
        natr_multiplier = 2.0
        tp_default = trial.suggest_float("tp_default", 0.04, 0.05, step=0.01)
        # Suggest hyperparameters using the trial object
        total_amount_quote = 1000
        max_executors_per_side = 1
        time_limit = 60 * 60 * 24 * 2
        cooldown_time = 60 * 15

        # Create the strategy configuration
        # Creating the instance of the configuration and the controller
        config = XGridTControllerConfig(
            connector_name=connector_name,
            trading_pair=trading_pair,
            interval=interval,
            total_amount_quote=Decimal(total_amount_quote),
            time_limit=time_limit,
            max_executors_per_side=max_executors_per_side,
            cooldown_time=cooldown_time,
            ema_short=ema_short,
            ema_medium=ema_medium,
            ema_long=ema_long,
            donchian_channel_length=donchian_channel_length,
            natr_length=natr_length,
            natr_multiplier=natr_multiplier,
            tp_default=tp_default
        )

        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)
    
    def generate_custom_configs(self) -> List[BacktestingConfig]:
        """
        Generate a list of configurations for each trading pair to optimize.
        
        Returns:
            List of BacktestingConfig objects
        """
        configs = []
        
        # Default parameter values
        ema_short = 20
        ema_medium = 50
        ema_long = 100
        donchian_channel_length = 50
        natr_length = 100
        natr_multiplier = 2.0
        tp_default = 0.04
        total_amount_quote = 1000
        max_executors_per_side = 1
        time_limit = 60 * 60 * 24 * 2
        cooldown_time = 60 * 15
        
        # Get trading pairs from config
        trading_pairs = self.config.get("trading_pairs", [self.config.get("trading_pair")])
        connector_name = self.config.get("connector_name", "hyperliquid_perpetual")
        interval = self.config.get("interval", "1m")
        
        for trading_pair in trading_pairs:
            config = XGridTControllerConfig(
                connector_name=connector_name,
                trading_pair=trading_pair,
                interval=interval,
                total_amount_quote=Decimal(total_amount_quote),
                time_limit=time_limit,
                max_executors_per_side=max_executors_per_side,
                cooldown_time=cooldown_time,
                ema_short=ema_short,
                ema_medium=ema_medium,
                ema_long=ema_long,
                donchian_channel_length=donchian_channel_length,
                natr_length=natr_length,
                natr_multiplier=natr_multiplier,
                tp_default=tp_default
            )
            
            configs.append(BacktestingConfig(config=config, start=self.start, end=self.end))
            
        return configs


class XGridTBacktestingTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.config_helper = TaskConfigHelper(config)
    
    async def execute(self):
        # Create Timescale client
        db_client = self.config_helper.create_timescale_client()
        
        # Create optimizer with the Timescale client
        optimizer = StrategyOptimizer(
            root_path=self.config["root_path"],
            resolution=self.config.get("resolution", "1m"),
            db_client=db_client
        )

        selected_pairs = self.config.get("selected_pairs", [])
        connector_name = self.config.get("connector_name")
        
        try:
            # Connect to the database
            await db_client.connect()
            
            # Calculate dates
            end_date = time.time() - self.config.get("end_time_buffer_hours", 6) * 3600
            start_date = end_date - self.config.get("lookback_days", 20) * 24 * 60 * 60
            
            # Format for study name
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            study_name = f"xgridt_{connector_name}_{today_str}"
            
            logger.info(f"Backtesting for {len(selected_pairs)} pairs from {pd.to_datetime(start_date, unit='s')} to {pd.to_datetime(end_date, unit='s')}")
            
            # Create config generator with all pairs
            config_generator = XGridTConfigGenerator(
                start_date=pd.to_datetime(start_date, unit="s"),
                end_date=pd.to_datetime(end_date, unit="s"),
                config={
                    "connector_name": connector_name,
                    "trading_pairs": selected_pairs,
                    "interval": self.config.get("resolution", "1m")
                }
            )
            
            # Use optimize_custom_configs which supports Timescale
            await optimizer.optimize_custom_configs(
                study_name=study_name,
                config_generator=config_generator
            )
            
            logger.info(f"Backtesting completed. Results stored in study: {study_name}")
            
        except Exception as e:
            logger.exception(f"Error during backtesting: {str(e)}")
        finally:
            await db_client.close()


async def main():
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
        "connector_name": "okx",
        "total_amount": 100,
        "lookback_days": 20,
        "end_time_buffer_hours": 6,
        "resolution": "1m",
        "n_trials": 200,
        "selected_pairs": ['1000BONK-USDT', 'BTC-USDT', 'ETH-USDT', 'SOL-USDT'],
        "timescale_config": {
            "db_host": os.getenv("TIMESCALE_HOST", "localhost"),
            "db_port": int(os.getenv("TIMESCALE_PORT", "5432")),
            "db_user": os.getenv("TIMESCALE_USER", "admin"),
            "db_password": os.getenv("TIMESCALE_PASSWORD", "admin"),
            "db_name": os.getenv("TIMESCALE_DB", "timescaledb")
        }
    }

    task = XGridTBacktestingTask("Backtesting", timedelta(hours=12), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
