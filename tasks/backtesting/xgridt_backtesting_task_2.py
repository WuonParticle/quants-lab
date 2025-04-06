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
import yaml

import pandas as pd
from dotenv import load_dotenv
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from hummingbot.strategy_v2.utils.distributions import Distributions

from core.data_sources import CLOBDataSource
from controllers.directional_trading.xgridt import XGridTControllerConfig
from core.backtesting.optimizer import StrategyOptimizer, BacktestingConfig, BaseStrategyConfigGenerator
from core.task_base import BaseTask
from core.task_config_helpers import TaskConfigHelper

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

        logger.debug(f"Config generated for trial {trial.number}")
        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)


class XGridTBacktestingTask2(BaseTask):
    async def execute(self):
        start_time = time.time()
        logger.info(f"Starting XGridT backtesting task at {datetime.datetime.now()}")
        self.config_helper = TaskConfigHelper(self.config)
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
        
        # Validate trading pairs against connector's trading rules
        clob = CLOBDataSource()
        trading_rules = await clob.get_trading_rules(connector_name)
        valid_pairs = [pair for pair in selected_pairs if pair in trading_rules.get_all_trading_pairs()]
        
        if len(valid_pairs) < len(selected_pairs):
            logger.warning(f"Filtered out {len(selected_pairs) - len(valid_pairs)} invalid trading pairs")
            logger.warning(f"Original pairs: {selected_pairs}")
            logger.warning(f"Valid pairs: {valid_pairs}")
        
        intervals = self.config.get("intervals", ["1m"])
        
        for i, trading_pair in enumerate(valid_pairs):
            pair_start_time = time.time()
            logger.info(f"[{i+1}/{len(valid_pairs)}] Processing {trading_pair}")
            
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
            
            candles_start_time = time.time()
            # TODO: remove date from trial and custom name
            #  TODO: make study have maximum trial count so not required to have 1 trial every time. 
            today_str = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            try:
                # Use the new TimescaleDB loading method instead of local cache
                candles_loaded = await optimizer._backtesting_engine.load_candles_cache_for_connector_pair_from_timescale(
                    connector_name=connector_name, 
                    trading_pair=trading_pair,
                    intervals=intervals,
                    start_time = start_date - 60 * 60, # add 1 hour buffer
                    end_time = end_date + 60 * 60,
                    timescale_client=self.config_helper.create_timescale_client()
                )
                
                candles_duration = time.time() - candles_start_time
                if candles_loaded:
                    logger.info(f"Candles loaded from TimescaleDB in {candles_duration:.2f} seconds")
                else:
                    logger.warning(f"Failed to load candles from TimescaleDB, trying local cache")
                    # optimizer.load_candles_cache_by_connector_pair(connector_name=connector_name, trading_pair=trading_pair)
                    # logger.info(f"Candles loaded from local cache in {time.time() - candles_start_time:.2f} seconds")
                
                # Check if there are any candles loaded
                optimize_start_time = time.time()
                logger.info(f"Starting optimization with {self.config['n_trials']} trials for {trading_pair}")
                
                await optimizer.optimize(
                    study_name=f"xgridt_{today_str}_{trading_pair}",
                    config_generator=config_generator, 
                    n_trials=self.config["n_trials"],
                    num_parallel_trials=self.config.get("max_parallel_trials", 4)
                )
                
                optimize_duration = time.time() - optimize_start_time
                logger.info(f"Optimization completed in {optimize_duration:.2f} seconds for {trading_pair}")
                
                # Save the best configuration to a file
                configs_dir = os.path.abspath(os.path.join(self.config["root_path"], "configs", "xgridt"))
                os.makedirs(configs_dir, exist_ok=True)
                
                config_file = os.path.join(configs_dir, f"xgridt_{today_str}_{trading_pair}_best.yml")
                
                # Direct call to optimizer's method without template
                await optimizer.save_best_config_to_yaml(
                    study_name=f"xgridt_{today_str}_{trading_pair}",
                    output_path=str(config_file),
                    config_generator=config_generator
                )
            
            except Exception as e:
                logger.error(f"Error saving best configuration for {trading_pair}: {e}")
            
            pair_duration = time.time() - pair_start_time
            logger.info(f"Completed {trading_pair} in {pair_duration:.2f} seconds")
        
        total_duration = time.time() - start_time
        logger.info(f"XGridT backtesting task completed in {total_duration:.2f} seconds")


async def main():
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
        "connector_name": "hyperliquid_perpetual",
        "total_amount": 100,
        "lookback_days": 20,
        "end_time_buffer_hours": 6,
        "resolution": "1m",
        "n_trials": 200,
        # "selected_pairs": ['PNUT-USDT', '1000SHIB-USDT', 'WLD-USDT', '1000BONK-USDT', 'DOGE-USDT', '1000PEPE-USDT',
        #                   'SUI-USDT', '1000SATS-USDT', 'MOODENG-USDT', 'NEIRO-USDT', 'HBAR-USDT', 'ENA-USDT',
        #                   'HMSTR-USDT', 'TROY-USDT', '1000X-USDT', 'SOL-USDT', 'ACT-USDT',
        #                   'XRP-USDT', 'SWELL-USDT', 'AGLD-USDT']
        "selected_pairs": ['1000BONK-USDT'],
        "intervals": ["1m", "5m", "15m", "1h"],
        "timescale_config": {
            "host": "localhost",
            "port": 5432,
            "user": "admin",
            "password": "admin",
            "database": "timescaledb"
        }
    }

    task = XGridTBacktestingTask("Backtesting", timedelta(hours=12), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
