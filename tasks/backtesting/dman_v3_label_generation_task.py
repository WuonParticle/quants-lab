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
import fcntl

from controllers.directional_trading.dman_v3 import DManV3Controller, DManV3ControllerConfig
from core.backtesting.optimizer import StrategyOptimizer, BacktestingConfig, BaseStrategyConfigGenerator
from core.task_base import BaseTask, LeaderElectedTask
from core.task_config_helpers import TaskConfigHelper

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DManV3ConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for DManV3 optimization.
    """
    async def generate_config(self, trial: optuna.Trial) -> BacktestingConfig:
        # Controller configuration
        connector_name = self.config["connector_name"]
        trading_pair = self.config["trading_pair"]
        interval = self.config.get("interval", "3m")
        logger.debug(f"Generating config for {connector_name} {trading_pair} in trial {trial.number}")
        
        trial.set_user_attr("connector_name", connector_name)
        trial.set_user_attr("trading_pair", trading_pair)
        trial.set_user_attr("interval", interval)
        
        # Bollinger Bands parameters - These are tuned for each window
        bb_length = trial.suggest_int("bb_length", 20, 200, step=10)
        bb_std = trial.suggest_float("bb_std", 1.0, 3.5, step=0.1)
        bb_long_threshold = trial.suggest_float("bb_long_threshold", 0.0, 0.4, step=0.05)
        bb_short_threshold = trial.suggest_float("bb_short_threshold", 0.6, 1.0, step=0.05)
        
        # Dynamic order parameters
        dynamic_order_spread = trial.suggest_categorical("dynamic_order_spread", [True, False])
        dynamic_target = trial.suggest_categorical("dynamic_target", [True, False])
        
        # Risk management parameters (in %)
        stop_loss = Decimal(trial.suggest_float("stop_loss_pct", 0.5, 10.0, step=0.25) / 100)
        take_profit = Decimal(trial.suggest_float("take_profit_pct", 0.5, 10.0, step=0.25) / 100)
        
        # Time parameters
        time_limit = trial.suggest_int("time_limit_sec", 300, 7200, step=300)
        cooldown_time = trial.suggest_int("cooldown_time_sec", 60, 600, step=60)
        
        # Fixed parameters from the config - These are determined from hyperparameter tuning
        total_amount_quote = self.config.get("total_amount_quote", 100)
        leverage = self.config.get("leverage", None)
        
        # Parse trailing stop
        trailing_stop_str = self.config.get("trailing_stop")
        trailing_stop = None
        if trailing_stop_str and trailing_stop_str != "null":
            parts = trailing_stop_str.split(",")
            if len(parts) == 2:
                trailing_stop = TrailingStop(
                    activation_price=Decimal(parts[0]),
                    trailing_delta=Decimal(parts[1])
                )
        
        # Parse DCA spreads
        dca_spreads_str = self.config.get("dca_spreads")
        dca_spreads = None
        if dca_spreads_str and dca_spreads_str.startswith("[") and dca_spreads_str.endswith("]"):
            dca_spreads_list = dca_spreads_str.strip("[]").split(",")
            dca_spreads = [Decimal(spread.strip()) for spread in dca_spreads_list]
        
        # Parse activation bounds
        activation_bounds_str = self.config.get("activation_bounds")
        activation_bounds = None
        if activation_bounds_str and activation_bounds_str.startswith("[") and activation_bounds_str.endswith("]") and activation_bounds_str != "null":
            activation_bounds_list = activation_bounds_str.strip("[]").split(",")
            activation_bounds = [Decimal(bound.strip()) for bound in activation_bounds_list]

        # Create the strategy configuration
        config = DManV3ControllerConfig(
            connector_name=connector_name,
            trading_pair=trading_pair,
            interval=interval,
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
            leverage=leverage,
        )

        logger.debug(f"Config generated for trial {trial.number}")
        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)


class DManV3LabelGenerationTask(LeaderElectedTask):
    async def task_execute(self):
        task_start_time = time.perf_counter()
        self.config_helper = TaskConfigHelper(self.config)
        random.seed(42)
        filtered_config = {k: v for k, v in self.config.items() if k not in ['timescale_config', 'postgres_config', 'mongo_config']}
        logger.info(f"Starting DManV3LabelGenerationTask at {datetime.datetime.now()} with config: {filtered_config}")
        
        (self.root_path / "data" / "labels").mkdir(parents=True, exist_ok=True)
        # Utilize a throwaway database as we are creating a huge number of studies
        self.config["postgres_config"]["database"] = "throwaway_optuna_db"
        
        backtesting_interval = self.config.get("backtesting_interval", "1s")
        candle_interval = self.config.get("interval", "3m")
        optimizer = StrategyOptimizer(root_path=self.root_path.absolute(),
                                    resolution=backtesting_interval,
                                    db_client=self.config_helper.create_timescale_client(),
                                    storage_name=StrategyOptimizer.get_storage_name("postgres", **self.config),
                                    custom_objective=lambda _, x: x["net_pnl_quote"],  # Focus on profit for directional strategy
                                    backtest_offset=self.config.get("backtest_offset", 0)
                                    )
        
        selected_pairs = self.config.get("selected_pairs")
        connector_name = self.config.get("connector_name")
        study_name_suffix = self.config.get("study_name_suffix", "")
        force_new_study = self.config.get("force_new_study", False)
        debug_study_name = self.config.get("debug_study", None)
        debug_trial = self.config.get("debug_trial", False)
        n_trials = self.config["n_trials"]
        
        for i, trading_pair in enumerate(selected_pairs):
            pair_start_time = time.perf_counter()
            logger.info(f"[{i+1}/{len(selected_pairs)}] Processing {trading_pair}")
            
            start_time, end_time, human_start, human_end, backtest_window_size, backtest_window_step = self.config_helper.get_backtesting_time_range().for_window()
            
            logger.info(f"Optimizing strategy for {connector_name} {trading_pair} {human_start} {human_end}")
            
            # Load all candles for the entire period first
            await optimizer._backtesting_engine.load_candles_cache_for_connector_pair_from_timescale(
                    connector_name=connector_name, 
                    trading_pair=trading_pair,
                    intervals=[backtesting_interval, candle_interval],
                    start_time=start_time - 60 * 60,  # add 1 hour buffer for TA calculations and window overhang
                    end_time=end_time + 60 * 60,
                    timescale_client=optimizer._db_client
                )

            try:
                # Calculate number of windows based on total duration and step size
                total_duration = end_time - start_time
                num_windows = int(total_duration / backtest_window_step)
                logger.info(f"Creating {num_windows} studies for {trading_pair} with step {backtest_window_step}s")
                if debug_study_name is not None:
                    num_windows = 1
                studies = []
                # TODO: have leader precreate all studies so that workers can preload the study list.
                for window_idx in range(num_windows):
                    window_start = start_time + (window_idx * backtest_window_step)
                    window_end = window_start + backtest_window_size
                    window_human_start = datetime.datetime.fromtimestamp(window_start).strftime('%Y-%m-%d %H:%M:%S')
                    window_human_end = datetime.datetime.fromtimestamp(window_end).strftime('%Y-%m-%d %H:%M:%S')
                    
                    logger.info(f"Processing window {window_idx + 1}/{num_windows}: {window_human_start} to {window_human_end}")
                    if debug_study_name is None:
                        study_name = f"{self.name.replace(' ', '_').lower()}_{trading_pair}_{backtesting_interval}_{n_trials}_{study_name_suffix}_{window_start:.0f}"
                        if force_new_study:
                            study_name = f"{study_name}_{task_start_time:.0f}"
                    else:
                        try:
                            study_name = debug_study_name
                            parts = study_name.split('_')
                            window_start = float(parts[-1].split('.')[0])  # Remove any decimal part
                            window_end = window_start + backtest_window_size
                            logger.info(f"Extracted window_start={window_start} and window_end={window_end} from study_name")
                            if force_new_study:
                                study_name = f"{study_name}_{task_start_time:.0f}"
                        except (IndexError, ValueError) as e:
                            logger.error(f"Failed to extract window_start from study_name: {study_name}. Error: {str(e)}")
                            raise
                
                    optimize_start_time = time.perf_counter()
                    config_generator = DManV3ConfigGenerator(
                        start_date=pd.to_datetime(window_start, unit="s"),
                        end_date=pd.to_datetime(window_end, unit="s"),
                        config={**self.config, "trading_pair": trading_pair}
                    )
                    
                    logger.debug(f"Starting optimization with {n_trials} trials for {trading_pair} window {window_idx + 1}")     
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
                            n_trials=n_trials
                        )
                    studies.append((study, window_start, window_idx))
                    optimize_duration = time.perf_counter() - optimize_start_time
                    logger.debug(f"Optimization completed in {optimize_duration:.1f} seconds for {trading_pair} window {window_idx + 1}")
                    
                # Save all optimal configurations for this trading pair to CSV - only if we're the leader
                if self.is_leader:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    labels_file_name = f"{trading_pair.replace('-', '_')}_optimal_configs_{timestamp}.csv"
                    labels_file = self.root_path / "data" / "labels" / labels_file_name
                    logger.info(f"Leader is saving optimal configurations to {labels_file}")
                    
                    try:
                        data = []
                        for study, window_start, window_idx in studies:
                            row = study.best_trial.params.copy()
                            row['value'] = study.best_trial.value
                            row['window_start'] = pd.to_datetime(window_start, unit='s')
                            data.append(row)
                        df = pd.DataFrame(data)
                        df.set_index('window_start', inplace=True)
                        df.to_csv(labels_file)
                        logger.info(f"Saved {len(studies)} optimal configurations to {labels_file}")
                    except Exception as e:
                        logger.error(f"Error saving configurations: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error processing {trading_pair}: {str(e)}")
                logger.error(traceback.format_exc())
            finally:
                await optimizer._db_client.close()
            
            pair_duration = time.perf_counter() - pair_start_time
            logger.info(f"Completed {trading_pair} in {pair_duration:.2f} seconds with best_trial value {study.best_trial.value}")


async def main():
    # Force is needed to override other logging configurations
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    # Run from command line with: python -m tasks.backtesting.dman_v3_label_generation_task --config config/dman_v3_label_generation_task.yml
    config = BaseTask.load_single_task_config()
    task = DManV3LabelGenerationTask("DManV3 Label Generation", None, config)
    await task.run_once()

if __name__ == "__main__":
    debug = os.environ.get("ASYNC_DEBUG", False)
    asyncio.run(main(), debug=debug) 