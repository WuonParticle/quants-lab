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

from controllers.market_making.pmm_simple import PMMSimpleController, PMMSimpleConfig
from core.backtesting.optimizer import StrategyOptimizer, BacktestingConfig, BaseStrategyConfigGenerator
from core.task_base import BaseTask
from core.task_config_helpers import TaskConfigHelper

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMMSimpleConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for PMM Simple optimization.
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
        
        # Order parameters
        num_levels = 1 
        # trial.suggest_int("levels", 2, 5)
        # Generate buy and sell spreads
        # Buy spreads
        buy_0 = trial.suggest_float("buy_0", 0.00001, 0.01001, step=0.00005)
        buy_1_step = trial.suggest_float("buy_1_step", 0.00001, 0.01001, step=0.00005)
        sell_0 = trial.suggest_float("sell_0", 0.00001, 0.01001, step=0.00005)
        sell_1_step = trial.suggest_float("sell_1_step", 0.00001, 0.01001, step=0.00005)
        buy_spreads = [buy_0, buy_0 + buy_1_step]
        sell_spreads = [sell_0, sell_0 + sell_1_step]
        
        # Risk management parameters
        total_amount_quote = self.config.get("total_amount_quote", 100)
        take_profit = trial.suggest_float("take_profit", 0.01, 0.05, step=0.005)
        stop_loss = trial.suggest_float("stop_loss", 0.01, 0.05, step=0.005)
        # trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.005, 0.02, step=0.005)
        # trailing_delta_ratio = trial.suggest_float("trailing_delta_ratio", 0.1, 0.5, step=0.1)
        # trailing_stop_trailing_delta = trailing_stop_activation_price * trailing_delta_ratio
        
        # Time parameters
        time_limit = trial.suggest_int("time_limit", 60, 35940, step=60)
        executor_refresh_time = trial.suggest_int("executor_refresh_time", 60, 300, step=30)
        cooldown_time = trial.suggest_int("cooldown_time", 60, 300, step=30)

        # logger.debug(f"Selected parameters: buy_spread={buy_spread}, sell_spread={sell_spread}, levels={num_levels}")

        # Create the strategy configuration
        config = PMMSimpleConfig(
            connector_name=connector_name,
            trading_pair=trading_pair,
            total_amount_quote=Decimal(total_amount_quote),
            buy_spreads=buy_spreads,
            sell_spreads=sell_spreads,
            # we must explicitly set these to get the pydantic validator to get called  
            buy_amounts_pct=None,
            sell_amounts_pct=None,
            take_profit=Decimal(take_profit),
            
            stop_loss=Decimal(stop_loss),
            # trailing_stop=TrailingStop(
            #     activation_price=Decimal(trailing_stop_activation_price), 
            #     trailing_delta=Decimal(trailing_stop_trailing_delta)
            # ),
            time_limit=time_limit,
            cooldown_time=cooldown_time,
            executor_refresh_time=executor_refresh_time,
        )

        logger.debug(f"Config generated for trial {trial.number}")
        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)


class PMMSimpleBacktestingTask(BaseTask):
    async def execute(self):
        start_time = time.time()
        self.config_helper = TaskConfigHelper(self.config)
        
        filtered_config = {k: v for k, v in self.config.items() if k not in ['timescale_config', 'postgres_config', 'mongo_config']}
        logger.info(f"Starting PMMSimpleBacktestingTask at {datetime.datetime.now()} with config: {filtered_config}")
        
        root_path = Path(os.getenv("root_path") or self.config.get("root_path", "../.."))
        (root_path / "data" / "candles").mkdir(parents=True, exist_ok=True)
        (root_path / "data" / "backtesting").mkdir(parents=True, exist_ok=True)
        
        backtesting_interval = self.config.get("backtesting_interval", "1m")
        candle_interval = self.config.get("interval", "1m")
        optimizer = StrategyOptimizer(root_path=root_path.absolute(),
                                     resolution=backtesting_interval,
                                     db_client=self.config_helper.create_timescale_client(),
                                     storage_name=StrategyOptimizer.get_storage_name("postgres", **self.config),
                                     custom_objective= lambda _, x: x["total_volume"] if x["net_pnl_quote"] > 0 else 0.0
                                    )
        logger.info(f"StrategyOptimizer initialized with root_path: {root_path.absolute()}")
        
        selected_pairs = self.config.get("selected_pairs")
        connector_name = self.config.get("connector_name")
        
        for i, trading_pair in enumerate(selected_pairs):
            pair_start_time = time.time()
            logger.info(f"[{i+1}/{len(selected_pairs)}] Processing {trading_pair}")
            
            start_date, end_date, human_start, human_end = self.config_helper.get_backtesting_time_range()
            
            logger.info(f"Optimizing strategy for {connector_name} {trading_pair} {human_start} {human_end}")

            config_generator = PMMSimpleConfigGenerator(
                start_date=pd.to_datetime(start_date, unit="s"),
                end_date=pd.to_datetime(end_date, unit="s"),
                config={**self.config, "trading_pair": trading_pair}
            )
            
            logger.info(f"Fetching candles for {connector_name} {trading_pair}")
            
            try:
                await optimizer._backtesting_engine.load_candles_cache_for_connector_pair_from_timescale(
                    connector_name=connector_name, 
                    trading_pair=trading_pair,
                    intervals=[backtesting_interval, candle_interval],
                    start_time = start_date - 60 * 60, # add 1 hour buffer for TA calculations
                    end_time = end_date + 60 * 60,
                    timescale_client=self.config_helper.create_timescale_client()
                )
                
                optimize_start_time = time.time()
                logger.info(f"Starting optimization with {self.config['n_trials']} trials for {trading_pair}")
                study_name = f"pmm_simple_{trading_pair}_{backtesting_interval}_{self.config['n_trials']}_{optimize_start_time}"
                debug_trial = self.config.get("debug_trial", False)
                if debug_trial:
                    await optimizer.repeat_trial(
                        study_name=study_name,
                        trial_number=debug_trial,
                        config_generator=config_generator
                    )
                else:
                    await optimizer.optimize(
                        study_name=study_name,
                        config_generator=config_generator, 
                        n_trials=self.config["n_trials"]
                    )
                
                optimize_duration = time.time() - optimize_start_time
                logger.info(f"Optimization completed in {optimize_duration:.2f} seconds for {trading_pair}")
                
                today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                # Save the best configuration to YAML
                best_config_path = root_path / "config" / "generated" / f"pmm_simple_{trading_pair.replace('-', '_')}_{today_str}.yml"
                best_config_path.parent.mkdir(parents=True, exist_ok=True)
                
                await optimizer.save_best_config_to_yaml(
                    study_name=study_name, 
                    output_path=str(best_config_path.absolute()),
                    config_generator=config_generator
                )
                
            except Exception as e:
                logger.error(f"Error processing {trading_pair}: {str(e)}")
                logger.error(traceback.format_exc())
            
            pair_duration = time.time() - pair_start_time
            logger.info(f"Completed {trading_pair} in {pair_duration:.2f} seconds")
        
        total_duration = time.time() - start_time
        logger.info(f"PMM Simple backtesting task completed in {total_duration:.2f} seconds")


async def main():
    # Force is needed to override other logging configurations
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    # Run from command line with: python -m tasks.backtesting.pmm_simple_backtesting_task --config config/pmm_simple_backtesting_task.yml
    config = BaseTask.load_single_task_config()
    task = PMMSimpleBacktestingTask("PMM Simple Backtesting", None, config)
    await task.run_once()

if __name__ == "__main__":
    debug = os.environ.get("ASYNC_DEBUG", False)
    asyncio.run(main(), debug=debug) 