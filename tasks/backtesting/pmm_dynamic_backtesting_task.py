import asyncio
import datetime
import logging
import os
import time
from datetime import timedelta
from typing import Any, Dict
from decimal import Decimal
import traceback
import optuna
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from hummingbot.strategy_v2.utils.distributions import Distributions

from controllers.market_making.pmm_dynamic import PMMDynamicControllerConfig
from core.backtesting.optimizer import StrategyOptimizer, BacktestingConfig, BaseStrategyConfigGenerator
from core.task_base import BaseTask
from core.task_config_helpers import TaskConfigHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class PMMDynamicConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for PMM Dynamic optimization.
    """
    async def generate_config(self, trial: optuna.Trial) -> BacktestingConfig:
        # Controller configuration
        connector_name = self.config["connector_name"]
        trading_pair = self.config["trading_pair"]
        interval = self.config.get("interval", "1m")
        logger.debug(f"Generating config for {connector_name} {trading_pair} in trial {trial.number}")
        
        trial.set_user_attr("connector_name", connector_name)
        trial.set_user_attr("trading_pair", trading_pair)
        trial.set_user_attr("interval", interval)
        
        # Suggest hyperparameters using the trial object
        # MACD parameters
        macd_fast = trial.suggest_int("macd_fast", 8, 20)
        macd_slow = trial.suggest_int("macd_slow", macd_fast + 5, 40)
        macd_signal = trial.suggest_int("macd_signal", 5, 15)
        
        # NATR parameters
        natr_length = trial.suggest_int("natr_length", 10, 30)
        
        # Order parameters
        levels = trial.suggest_int("levels", 2, 3)
        start_spread = trial.suggest_float("start_spread", 0.1, 1, step=0.05)
        step_spread = trial.suggest_float("step_spread", 0.1, 1, step=0.05)
        spreads = Distributions.arithmetic(levels, start_spread, step_spread)
        
        # Risk management parameters
        total_amount_quote = self.config.get("total_amount_quote", 100)
        take_profit = trial.suggest_float("take_profit", 0.01, 0.05, step=0.01)
        stop_loss = trial.suggest_float("stop_loss", 0.01, 0.05, step=0.01)
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.005, 0.02, step=0.005)
        trailing_delta_ratio = trial.suggest_float("trailing_delta_ratio", 0.1, 0.5, step=0.1)
        trailing_stop_trailing_delta = trailing_stop_activation_price * trailing_delta_ratio
        
        # Time parameters
        time_limit = trial.suggest_int("time_limit", 60 * 60 * 1, 60 * 60 * 8, step=60 * 60)
        executor_refresh_time = trial.suggest_int("executor_refresh_time", 60 * 1, 60 * 10, step=60)
        cooldown_time = trial.suggest_int("cooldown_time", 60 * 5, 60 * 30, step=60 * 5)

        logger.debug(f"Selected parameters: macd_fast={macd_fast}, macd_slow={macd_slow}, macd_signal={macd_signal}, natr_length={natr_length}")

        # Create the strategy configuration
        config = PMMDynamicControllerConfig(
            connector_name=connector_name,
            trading_pair=trading_pair,
            interval=interval,
            total_amount_quote=Decimal(total_amount_quote),
            buy_spreads=spreads,
            sell_spreads=spreads,
            take_profit=Decimal(take_profit),
            stop_loss=Decimal(stop_loss),
            trailing_stop=TrailingStop(
                activation_price=Decimal(trailing_stop_activation_price), 
                trailing_delta=Decimal(trailing_stop_trailing_delta)
            ),
            time_limit=time_limit,
            cooldown_time=cooldown_time,
            executor_refresh_time=executor_refresh_time,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            natr_length=natr_length
        )

        logger.debug(f"Config generated for trial {trial.number}")
        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)


class PMMDynamicBacktestingTask(BaseTask):
    async def execute(self):
        start_time = time.time()
        self.config_helper = TaskConfigHelper(self.config)
        logger.info(f"Starting PMMDynamicBacktestingTask at {datetime.datetime.now()} with config: {self.config}")
        
        root_path = Path(os.getenv("root_path") or self.config.get("root_path", "../.."))
        (root_path / "data" / "candles").mkdir(parents=True, exist_ok=True)
        (root_path / "data" / "backtesting").mkdir(parents=True, exist_ok=True)
        
        backtesting_interval = self.config.get("backtesting_interval", "1m")
        candle_interval = self.config.get("interval", "1m")
        optimizer = StrategyOptimizer(root_path=root_path.absolute(), resolution=backtesting_interval)
        logger.info(f"StrategyOptimizer initialized with root_path: {root_path.absolute()}")
        
        selected_pairs = self.config.get("selected_pairs")
        connector_name = self.config.get("connector_name")
        
        for i, trading_pair in enumerate(selected_pairs):
            pair_start_time = time.time()
            logger.info(f"[{i+1}/{len(selected_pairs)}] Processing {trading_pair}")
            
            end_date = time.time() - self.config["end_time_buffer_hours"] * 3600
            start_date = end_date - self.config["lookback_days"] * 24 * 60 * 60
            
            human_start = datetime.datetime.fromtimestamp(start_date).strftime('%Y-%m-%d %H:%M:%S')
            human_end = datetime.datetime.fromtimestamp(end_date).strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Optimizing strategy for {connector_name} {trading_pair} {human_start} {human_end}")

            config_generator = PMMDynamicConfigGenerator(
                start_date=pd.to_datetime(start_date, unit="s"),
                end_date=pd.to_datetime(end_date, unit="s"),
                config={**self.config, "trading_pair": trading_pair}
            )
            
            logger.info(f"Fetching candles for {connector_name} {trading_pair}")
            
            try:
                candles_loaded = await optimizer._backtesting_engine.load_candles_cache_for_connector_pair_from_timescale(
                    connector_name=connector_name, 
                    trading_pair=trading_pair,
                    intervals=[backtesting_interval, candle_interval],
                    start_time = start_date - 60 * 60, # add 1 hour buffer
                    end_time = end_date + 60 * 60,
                    timescale_client=self.config_helper.create_timescale_client()
                )
                
                optimize_start_time = time.time()
                logger.info(f"Starting optimization with {self.config['n_trials']} trials for {trading_pair}")
                study_name = f"pmm_dynamic_{trading_pair}_{backtesting_interval}"
                await optimizer.optimize(
                    study_name=study_name,
                    config_generator=config_generator, 
                    n_trials=self.config["n_trials"],
                    max_parallel_trials=self.config.get("max_parallel_trials", 4)
                )
                
                optimize_duration = time.time() - optimize_start_time
                logger.info(f"Optimization completed in {optimize_duration:.2f} seconds for {trading_pair}")
                
                today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                # Save the best configuration to YAML
                best_config_path = root_path / "config" / "generated" / f"pmm_dynamic_{trading_pair.replace('-', '_')}_{today_str}.yml"
                best_config_path.parent.mkdir(parents=True, exist_ok=True)
                
                await optimizer.save_best_config_to_yaml(
                    study_name=study_name, 
                    output_path=str(best_config_path),
                    config_generator=config_generator
                )
                logger.info(f"Best configuration saved to {best_config_path}")
                
            except Exception as e:
                logger.error(f"Error processing {trading_pair}: {str(e)}")
                logger.error(traceback.format_exc())
            
            pair_duration = time.time() - pair_start_time
            logger.info(f"Completed {trading_pair} in {pair_duration:.2f} seconds")
        
        total_duration = time.time() - start_time
        logger.info(f"PMM Dynamic backtesting task completed in {total_duration:.2f} seconds")


async def main():
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_file = yaml.safe_load(f)
        
    config = next(iter(config_file.get("tasks").values())).get("config")

    task = PMMDynamicBacktestingTask("PMM Dynamic Backtesting", timedelta(hours=12), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())