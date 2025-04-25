import asyncio
import datetime
import logging
import os
import random
import time
from decimal import Decimal
import traceback
import optuna
import pandas as pd
from dotenv import load_dotenv

from controllers.market_making.pmm_simple import PMMSimpleConfig
from core.backtesting.optimizer import StrategyOptimizer, BacktestingConfig, BaseStrategyConfigGenerator
from core.task_base import BaseTask, ParallelWorkerTask
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
        # Generate buy and sell spreads
        # Buy spreads in bips to help with step size
        buy_0_bips = trial.suggest_float("buy_0_bips", 1, 200, step=1)
        buy_1_step = trial.suggest_float("buy_1_step", 0, 200, step=5)
        sell_0_bips = trial.suggest_float("sell_0_bips", 1, 200, step=1)
        sell_1_step = trial.suggest_float("sell_1_step", 0, 200, step=5)
        buy_spreads = [buy_0_bips / 10000, (buy_0_bips + buy_1_step) / 10000]
        sell_spreads = [sell_0_bips / 10000, (sell_0_bips + sell_1_step) / 10000]
        
        # Risk management parameters (in %)
        take_profit = trial.suggest_float("take_profit_pct", 0.125, 10, step=0.125) / 100
        stop_loss = trial.suggest_float("stop_loss_pct", 0.5, 20, step=0.125) / 100
        # trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.005, 0.02, step=0.005)
        # trailing_delta_ratio = trial.suggest_float("trailing_delta_ratio", 0.1, 0.5, step=0.1)
        # trailing_stop_trailing_delta = trailing_stop_activation_price * trailing_delta_ratio
        
        # fixing per pmm_simple_ADA-USDT_1s_2000_round2fixed   
        time_limit = 90 
        executor_refresh_time = 60
        cooldown_time = 10

        # Create the strategy configuration
        total_amount_quote = self.config.get("total_amount_quote", 100)
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


class PMMLabelGenerationTask(ParallelWorkerTask):
    async def task_execute(self):
        task_start_time = time.perf_counter()
        self.config_helper = TaskConfigHelper(self.config)
        random.seed(42)
        filtered_config = {k: v for k, v in self.config.items() if k not in ['timescale_config', 'postgres_config', 'mongo_config']}
        logger.info(f"Starting PMMLabelGenerationTask at {datetime.datetime.now()} with config: {filtered_config}")
        
        (self.root_path / "data" / "labels").mkdir(parents=True, exist_ok=True)
        # Utilze a throwaway database as we are creating a huge number of studies
        self.config["postgres_config"]["database"] = "throwaway_optuna_db"
        
        backtesting_interval = self.config.get("backtesting_interval", "1s")
        candle_interval = self.config.get("candle_interval", "1m")
        optimizer = StrategyOptimizer(root_path=self.root_path.absolute(),
                                    resolution=backtesting_interval,
                                    db_client=self.config_helper.create_timescale_client(),
                                    storage_name=StrategyOptimizer.get_storage_name("postgres", create_db_if_not_exists=False, **self.config),
                                    # custom_objective= lambda _, x: x["total_volume"] if x["net_pnl_quote"] > 0 else 0.0,
                                    custom_objective= lambda _, x: [x["net_pnl_quote"], x["total_volume"]],
                                    directions=["maximize", "maximize"],
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
            # TODO: only cache 1 day of candles at a time as it takes 1 call to timescale per 2 days
            await optimizer._backtesting_engine.load_candles_cache_for_connector_pair_from_timescale(
                    connector_name=connector_name, 
                    trading_pair=trading_pair,
                    intervals=[backtesting_interval, candle_interval],
                    start_time = start_time - 60 * 60, # add 1 hour buffer for TA calculations and window overhang
                    end_time = end_time + 60 * 60,
                    timescale_client=optimizer._db_client
                )

            try:
                # Calculate number of windows based on total duration and step size
                study_name_prefix = f"{self.name.replace(' ', '_').lower()}_{trading_pair}_{backtesting_interval}_{n_trials}_{study_name_suffix}"
                await optimizer.set_study_name_prefix(study_name_prefix)
                total_duration = end_time - start_time
                num_windows = int(total_duration / backtest_window_step)
                logger.info(f"Creating {num_windows} studies for {trading_pair} with step {backtest_window_step}s")
                
                if debug_study_name is not None:
                    num_windows = 1
                
                # Distribute windows among workers
                worker_windows = [idx for idx in range(num_windows) if self.should_process_item(idx)]
                logger.info(f"Worker {self.worker_id+1}/{self.total_workers} will process {len(worker_windows)} windows")
                # TODO: skip all windows that have already been completed
                
                # Leader needs all windows to collect results
                if self.is_leader:
                    worker_windows.extend([idx for idx in range(num_windows) if not self.should_process_item(idx)])
                    # study_names = optimizer.storage_wrapper.get_study_names_by_prefix(study_name_prefix)
                should_clear_study_cache = self.is_leader
                studies = []
                for window_idx in worker_windows:
                    window_start = start_time + (window_idx * backtest_window_step)
                    window_end = window_start + backtest_window_size
                    window_human_start = datetime.datetime.fromtimestamp(window_start).strftime('%Y-%m-%d %H:%M:%S')
                    window_human_end = datetime.datetime.fromtimestamp(window_end).strftime('%Y-%m-%d %H:%M:%S')
                    
                    logger.info(f"Processing window {window_idx + 1}/{num_windows}: {window_human_start} to {window_human_end}")
                    if debug_study_name is None:
                        study_name = f"{study_name_prefix}_{window_start:.0f}"
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
                    config_generator = PMMSimpleConfigGenerator(
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
                    
                    if should_clear_study_cache and not self.should_process_item(window_idx):
                        # reset the study cache so that the leader doesn't attempt to create a study for every study it's collecting 
                        optimizer.reset_study_cache()
                        should_clear_study_cache = False

                # Save all optimal configurations for this trading pair to CSV - only if we're the leader
                if self.is_leader:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    labels_file_name = f"{trading_pair.replace('-', '_')}_optimal_configs_{timestamp}.csv"
                    labels_file = self.root_path / "data" / "labels" / labels_file_name
                    logger.info(f"Leader is saving optimal configurations to {labels_file}")
                    
                    try:
                        data = []
                        studies.sort(key=lambda x: x[1])
                        for study, window_start, window_idx in studies:
                            best_trial = max(study.best_trials, key=lambda t: t.values[0])
                            row = best_trial.params.copy()
                            row['value'] = best_trial.values
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
                await optimizer.dispose()
            
            pair_duration = time.perf_counter() - pair_start_time
            logger.info(f"Completed {trading_pair} in {pair_duration:.2f}")


async def main():
    # Force is needed to override other logging configurations
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    # Run from command line with: python -m tasks.backtesting.pmm_simple_backtesting_task --config config/pmm_simple_backtesting_task.yml
    config = BaseTask.load_single_task_config()
    task = PMMLabelGenerationTask("PMM Simple Label Generation", None, config)
    await task.run_once()

if __name__ == "__main__":
    debug = os.environ.get("ASYNC_DEBUG", False)
    asyncio.run(main(), debug=debug) 