import asyncio
import datetime
import logging
import os
import random
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

from controllers.directional_trading.dman_v3 import DManV3Controller, DManV3ControllerConfig
from core.backtesting.optimizer import StrategyOptimizer, BacktestingConfig, BaseStrategyConfigGenerator
from core.task_base import BaseTask
from core.task_config_helpers import TaskConfigHelper

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class DManV3HyperparameterTuningConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for DManV3 optimization.
    """
    async def generate_config(self, trial: optuna.Trial) -> BacktestingConfig:
        # Controller configuration
        connector_name = self.config["connector_name"]
        trading_pair = self.config["trading_pair"]
        candle_interval = self.config.get("candle_interval", "3m")
        logger.debug(f"Generating config for {connector_name} {trading_pair} in trial {trial.number}")
        total_amount_quote = self.config.get("total_amount_quote", 100)
        # leverage = self.config.get("leverage", None)
        
        trial.set_user_attr("connector_name", connector_name)
        trial.set_user_attr("trading_pair", trading_pair)
        trial.set_user_attr("candle_interval", candle_interval)
        
        # Bollinger Bands parameters
        bb_length = trial.suggest_int("bb_length", 1, 200, step=10)
        bb_std = trial.suggest_float("bb_std", 1.0, 3.5, step=0.1)
        bb_long_threshold = trial.suggest_float("bb_long_threshold", 0.0, 0.4, step=0.05)
        bb_short_threshold = trial.suggest_float("bb_short_threshold", 0.6, 1.0, step=0.05)
        
        # Trailing stop parameters
        ts_activation_price_pct = trial.suggest_float("ts_activation_price_pct", 0.5, 5.0, step=0.25)
        ts_trailing_delta_pct = trial.suggest_float("ts_trailing_delta_pct", 0.1, 2.0, step=0.1)
        trailing_stop = TrailingStop(
            activation_price=Decimal(ts_activation_price_pct / 100),
            trailing_delta=Decimal(ts_trailing_delta_pct / 100)
        )
        
        # DCA parameters
        num_dca_levels = trial.suggest_int("num_dca_levels", 2, 5)
        dca_spread_0_pct = trial.suggest_float("dca_spread_0_pct", 0.1, 3.0, step=0.1)
        dca_spread_step_pct = trial.suggest_float("dca_spread_step_pct", 0.5, 5.0, step=0.25)
        dca_spreads = [Decimal(dca_spread_0_pct / 100 + i * dca_spread_step_pct / 100) for i in range(num_dca_levels)]
        
        # Dynamic order parameters
        dynamic_order_spread = trial.suggest_categorical("dynamic_order_spread", [True, False])
        dynamic_target = trial.suggest_categorical("dynamic_target", [True, False])
        
        # Activation bounds
        activation_bounds_enabled = trial.suggest_categorical("activation_bounds_enabled", [True, False])
        activation_bounds = None
        if activation_bounds_enabled:
            activation_bound_0_pct = trial.suggest_float("activation_bound_0_pct", 0.1, 2.0, step=0.1)
            activation_bound_step_pct = trial.suggest_float("activation_bound_step_pct", 0.5, 5.0, step=0.25)
            activation_bounds = [Decimal(activation_bound_0_pct / 100 + i * activation_bound_step_pct / 100) 
                                for i in range(num_dca_levels - 1)]  # one less than levels since last level doesn't need a bound
        
        # Risk management parameters (in %)
        stop_loss = Decimal(trial.suggest_float("stop_loss_pct", 0.5, 10.0, step=0.25) / 100)
        take_profit = Decimal(trial.suggest_float("take_profit_pct", 0.5, 10.0, step=0.25) / 100)
        
        # Time parameters
        time_limit = trial.suggest_int("time_limit_sec", 300, 7200, step=300)
        cooldown_time = trial.suggest_int("cooldown_time_sec", 60, 600, step=60)

        # Create the strategy configuration
        config = DManV3ControllerConfig(
            connector_name=connector_name,
            trading_pair=trading_pair,
            # temporary workaround for pydantic validation
            candles_connector=None,
            candles_trading_pair=None,
            interval=candle_interval,
            bb_length=bb_length,
            bb_std=bb_std,
            bb_long_threshold=bb_long_threshold,
            bb_short_threshold=bb_short_threshold,
            trailing_stop=trailing_stop,
            dca_spreads=dca_spreads,
            dca_amounts_pct=None,  # Let the config validator handle equal distribution
            dynamic_order_spread=dynamic_order_spread,
            dynamic_target=dynamic_target,
            activation_bounds=activation_bounds,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_limit=time_limit,
            cooldown_time=cooldown_time,
            total_amount_quote=Decimal(total_amount_quote),
            # leverage=leverage,
        )

        logger.debug(f"Config generated for trial {trial.number}")
        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)


class DManV3HyperparameterTuningTask(BaseTask):
    async def execute(self):
        task_start_time = time.time()
        self.config_helper = TaskConfigHelper(self.config)
        random.seed(42)
        filtered_config = {k: v for k, v in self.config.items() if k not in ['timescale_config', 'postgres_config', 'mongo_config']}
        logger.info(f"Starting DManV3HyperparameterTuningTask at {datetime.datetime.now()} with config: {filtered_config}")
        
        # Get the path relative to this file's location
        root_path = Path(os.getenv("root_path") or self.config.get("root_path", Path(__file__).parent / "../.."))
        (root_path / "data" / "candles").mkdir(parents=True, exist_ok=True)
        (root_path / "data" / "backtesting").mkdir(parents=True, exist_ok=True)
        
        backtesting_interval = self.config.get("backtesting_interval", "1s")
        candle_interval = self.config.get("candle_interval", "3m")
        optimizer = StrategyOptimizer(root_path=root_path.absolute(),
                                     resolution=backtesting_interval,
                                     db_client=self.config_helper.create_timescale_client(),
                                     storage_name=StrategyOptimizer.get_storage_name("postgres", **self.config),
                                     custom_objective=lambda _, x: x["net_pnl_quote"],  # Focus on profit for directional strategy
                                     backtest_offset=self.config.get("backtest_offset", 0)
                                    )
        logger.info(f"StrategyOptimizer initialized with root_path: {root_path.absolute()}")
        
        selected_pairs = self.config.get("selected_pairs")
        connector_name = self.config.get("connector_name")
        
        for i, trading_pair in enumerate(selected_pairs):
            pair_start_time = time.perf_counter()
            logger.info(f"[{i+1}/{len(selected_pairs)}] Processing {trading_pair}")
            
            start_time, end_time, human_start, human_end = self.config_helper.get_backtesting_time_range()
            
            logger.info(f"Optimizing strategy for {connector_name} {trading_pair} {human_start} {human_end}")

            config_generator = DManV3HyperparameterTuningConfigGenerator(
                start_date=pd.to_datetime(start_time, unit="s"),
                end_date=pd.to_datetime(end_time, unit="s"),
                config={**self.config, "trading_pair": trading_pair}
            )
            
            logger.info(f"Fetching candles for {connector_name} {trading_pair}")
            
            try:
                await optimizer._backtesting_engine.load_candles_cache_for_connector_pair_from_timescale(
                    connector_name=connector_name, 
                    trading_pair=trading_pair,
                    intervals=[candle_interval],
                    start_time=start_time - 60 * 500,  # add 500 minutes buffer for TA calculations
                    end_time=end_time + 60 * 5,
                    timescale_client=self.config_helper.create_timescale_client()
                )
                
                await optimizer._backtesting_engine.load_candles_cache_for_connector_pair_from_timescale(
                    connector_name=connector_name, 
                    trading_pair=trading_pair,
                    intervals=[backtesting_interval],
                    start_time=start_time - 60 * 5,  # add 5 minutes buffer for offset backtesting
                    end_time=end_time + 60 * 5,
                    timescale_client=self.config_helper.create_timescale_client()
                )
                
                optimize_start_time = time.perf_counter()
                study_name_suffix = self.config.get("study_name_suffix", "")
                force_new_study = self.config.get("force_new_study", False)
                logger.info(f"Starting optimization with {self.config['n_trials']} trials for {trading_pair}")
                study_name = f"{self.name.replace(' ', '_').lower()}_{trading_pair}_{backtesting_interval}_{self.config['n_trials']}_{study_name_suffix}"
                if force_new_study:
                    study_name = f"{study_name}_{task_start_time:.0f}"
                
                debug_trial = self.config.get("debug_trial", False)
                if debug_trial:
                    study = await optimizer.repeat_trial(
                        study_name=study_name,
                        trial_number=debug_trial,
                        config_generator=config_generator
                    )
                else:
                    study = await optimizer.optimize(
                        study_name=study_name,
                        config_generator=config_generator, 
                        n_trials=self.config["n_trials"]
                    )
                optimize_duration = time.perf_counter() - optimize_start_time
                logger.info(f"Optimization completed in {optimize_duration:.2f} seconds for {trading_pair}")
                
                # Save the best configuration to YAML
                best_config_path = root_path / "config" / "generated" / f"{study_name}.yml"
                best_config_path.parent.mkdir(parents=True, exist_ok=True)
                
                await optimizer.save_best_config_to_yaml(
                    study_name=study_name, 
                    output_path=str(best_config_path.absolute()),
                    config_generator=config_generator
                )
                
            except Exception as e:
                logger.error(f"Error processing {trading_pair}: {str(e)}")
                logger.error(traceback.format_exc())
            
            pair_duration = time.perf_counter() - pair_start_time
            logger.info(f"Completed {trading_pair} in {pair_duration:.2f} seconds with best_trial value {study.best_trial.value}")
        
        total_duration = time.time() - task_start_time
        logger.info(f"DManV3 hyperparameter tuning task completed in {total_duration:.2f} seconds")


async def main():
    # Force is needed to override other logging configurations
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    # Run from command line with: python -m tasks.backtesting.dman_v3_hyperparameter_tuning_task --config config/dman_v3_hyperparameter_tuning_task.yml
    config = BaseTask.load_single_task_config()
    task = DManV3HyperparameterTuningTask("DManV3 Hyperparameter Tuning", None, config)
    await task.run_once()

if __name__ == "__main__":
    debug = os.environ.get("ASYNC_DEBUG", False)
    asyncio.run(main(), debug=debug) 