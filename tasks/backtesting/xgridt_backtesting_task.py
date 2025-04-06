import asyncio
import datetime
import logging
import os
import time
from datetime import timedelta
from typing import Any, Dict
from decimal import Decimal
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from hummingbot.strategy_v2.utils.distributions import Distributions

from controllers.directional_trading.xgridt import XGridTControllerConfig
from core.backtesting.optimizer import StrategyOptimizer, BacktestingConfig, BaseStrategyConfigGenerator
from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class XGridTConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for XGridT optimization.
    """
    async def generate_config(self, trial) -> BacktestingConfig:
        # Controller configuration
        connector_name = self.config.get("connector_name", "binance_perpetual")
        trading_pair = self.config.get("trading_pair", "PNUT-USDT")
        interval = self.config.get("interval", "1m")
        logger.debug(f"Generating config for {connector_name} {trading_pair} in trial {trial.number}")
        
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

        logger.debug(f"Selected parameters: ema_short={ema_short}, ema_medium={ema_medium}, ema_long={ema_long}, tp_default={tp_default}")

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

        logger.debug(f"Config generated for trial {trial.number}")
        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)


class XGridTBacktestingTask(BaseTask):
    async def execute(self):
        start_time = time.time()
        logger.info(f"Starting XGridT backtesting task at {datetime.datetime.now()}")
        
        # Create necessary directories using Path
        root_path = self.config["root_path"]
        candles_dir = Path(root_path) / "data" / "candles"
        candles_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create the directory for backtesting database
        backtesting_dir = Path(root_path) / "data" / "backtesting"
        backtesting_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Ensured directories exist: {candles_dir} and {backtesting_dir}")
        
        optimizer = StrategyOptimizer(root_path=self.config["root_path"])
        logger.info(f"StrategyOptimizer initialized with root_path: {self.config['root_path']}")
        
        selected_pairs = self.config.get("selected_pairs")
        connector_name = self.config.get("connector_name")
        logger.info(f"Processing {len(selected_pairs)} trading pairs for connector {connector_name}")
        
        for i, trading_pair in enumerate(selected_pairs):
            pair_start_time = time.time()
            logger.info(f"[{i+1}/{len(selected_pairs)}] Processing {trading_pair}")
            
            end_date = time.time() - self.config["end_time_buffer_hours"] * 3600
            start_date = end_date - self.config["lookback_days"] * 24 * 60 * 60
            
            human_start = datetime.datetime.fromtimestamp(start_date).strftime('%Y-%m-%d %H:%M:%S')
            human_end = datetime.datetime.fromtimestamp(end_date).strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Backtesting period: {human_start} to {human_end}")
            
            logger.info(f"Optimizing strategy for {connector_name} {trading_pair} {start_date} {end_date}")
            
            config_generator = XGridTConfigGenerator(
                start_date=pd.to_datetime(start_date, unit="s"),
                end_date=pd.to_datetime(end_date, unit="s"),
                config={"connector_name": connector_name, "trading_pair": trading_pair}
            )
            
            logger.info(f"Fetching candles for {connector_name} {trading_pair}")
            candles_start_time = time.time()
            
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            try:
                optimizer.load_candles_cache_by_connector_pair(connector_name=connector_name, trading_pair=trading_pair)
                candles_duration = time.time() - candles_start_time
                logger.info(f"Candles loaded in {candles_duration:.2f} seconds")
                
                # Check if there are any candles loaded
                # Since we created the directory, we need more data to proceed
                
                optimize_start_time = time.time()
                logger.info(f"Starting optimization with {self.config['n_trials']} trials for {trading_pair}")
                
                await optimizer.optimize(
                    study_name=f"xgridt_{today_str}_{trading_pair}",
                    config_generator=config_generator, 
                    n_trials=self.config["n_trials"]
                )
                
                optimize_duration = time.time() - optimize_start_time
                logger.info(f"Optimization completed in {optimize_duration:.2f} seconds for {trading_pair}")
                
            except Exception as e:
                logger.error(f"Error processing {trading_pair}: {str(e)}")
                logger.error(traceback.format_exc())
            
            pair_duration = time.time() - pair_start_time
            logger.info(f"Completed {trading_pair} in {pair_duration:.2f} seconds")
        
        total_duration = time.time() - start_time
        logger.info(f"XGridT backtesting task completed in {total_duration:.2f} seconds")


async def main():
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
        "connector_name": "binance_perpetual",
        "total_amount": 100,
        "lookback_days": 20,
        "end_time_buffer_hours": 6,
        "resolution": "1m",
        "n_trials": 200,
        # "selected_pairs": ['PNUT-USDT', '1000SHIB-USDT', 'WLD-USDT', '1000BONK-USDT', 'DOGE-USDT', '1000PEPE-USDT',
        #                   'SUI-USDT', '1000SATS-USDT', 'MOODENG-USDT', 'NEIRO-USDT', 'HBAR-USDT', 'ENA-USDT',
        #                   'HMSTR-USDT', 'TROY-USDT', '1000X-USDT', 'SOL-USDT', 'ACT-USDT',
        #                   'XRP-USDT', 'SWELL-USDT', 'AGLD-USDT']
        "selected_pairs": ['1000BONK-USDT']
    }

    task = XGridTBacktestingTask("Backtesting", timedelta(hours=12), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
