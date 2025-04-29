import asyncio
import datetime
import logging
import os
import random
import time
from dataclasses import dataclass
from decimal import Decimal
import traceback
import optuna
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Tuple
from core.utils.time_utils import perf_log_text

from controllers.market_making.pmm_simple import PMMSimpleConfig
from core.backtesting.optimizer import StrategyOptimizer, BacktestingConfig, BaseStrategyConfigGenerator
from core.task_base import BaseTask, ParallelWorkerTask
from core.task_config_helpers import TaskConfigHelper

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerWindow:
    idx: int
    start: float
    end: float
    study_name: str
    
    @property
    def human_start(self) -> str:
        return datetime.datetime.fromtimestamp(self.start).strftime('%Y-%m-%d %H:%M:%S')
    
    @property
    def human_end(self) -> str:
        return datetime.datetime.fromtimestamp(self.end).strftime('%Y-%m-%d %H:%M:%S')


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
        # Buy spreads in bps to help with step size
        #TODO: rename to bps instead of bps
        buy_0_bps = trial.suggest_float("buy_0_bps", 1, 200, step=1)
        buy_1_step = trial.suggest_float("buy_1_step", 0, 200, step=5)
        sell_0_bps = trial.suggest_float("sell_0_bps", 1, 200, step=1)
        sell_1_step = trial.suggest_float("sell_1_step", 0, 200, step=5)
        buy_spreads = [buy_0_bps / 10000, (buy_0_bps + buy_1_step) / 10000]
        sell_spreads = [sell_0_bps / 10000, (sell_0_bps + sell_1_step) / 10000]
        
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


class PMMVMLLabelGenerationTask(ParallelWorkerTask):
    def __init__(self, name, config, **kwargs):
        super().__init__(name, config, **kwargs)
        # Move config parameters to __init__
        self.config_helper = TaskConfigHelper(config)
        self.selected_pairs = config.get("selected_pairs")
        self.connector_name = config.get("connector_name")
        self.study_name_suffix = config.get("study_name_suffix", "")
        self.force_new_study = config.get("force_new_study", False)
        self.debug_study_name = config.get("debug_study", None)
        self.debug_trial = config.get("debug_trial", False)
        self.n_trials = config["n_trials"]
        self.backtesting_interval = config.get("backtesting_interval", "1s")
        self.backtest_offset = config.get("backtest_offset", 0)
        # self.candle_interval = self.config.get("candle_interval", None)
        self.required_candle_intervals = [self.backtesting_interval]
        
        # Set time range parameters
        self.start_time, self.end_time, self.human_start, self.human_end, self.backtest_window_size, self.backtest_window_step = self.config_helper.get_backtesting_time_range().for_window()
        self.task_start_time = time.perf_counter()
        
    def create_worker_window(self, idx, study_name_prefix):
        window_start = self.start_time + (idx * self.backtest_window_step)
        window_end = window_start + self.backtest_window_size
        if self.debug_study_name is None:
            study_name = f"{study_name_prefix}_{window_start:.0f}"
            if self.force_new_study:
                study_name = f"{study_name}_{self.task_start_time:.0f}"
        else:
            study_name = self.debug_study_name
            try:
                parts = study_name.split('_')
                window_start = float(parts[-1].split('.')[0])  # Remove any decimal part
                window_end = window_start + self.backtest_window_size
                logger.info(f"Extracted window_start={window_start} and window_end={window_end} from study_name")
                if self.force_new_study:
                    study_name = f"{study_name}_{self.task_start_time:.0f}"
            except (IndexError, ValueError) as e:
                logger.error(f"Failed to extract window_start from study_name: {study_name}. Error: {str(e)}")
                raise
        
        return WorkerWindow(
            idx=idx,
            start=window_start,
            end=window_end,
            study_name=study_name
        )
        
    async def task_execute(self):
        random.seed(42)
        filtered_config = {k: v for k, v in self.config.items() if k not in ['timescale_config', 'postgres_config', 'mongo_config']}
        logger.info(f"Starting PMMVMLLabelGenerationTask at {datetime.datetime.now()} with config: {filtered_config}")
        
        (self.root_path / "data" / "labels").mkdir(parents=True, exist_ok=True)
        # Utilze a throwaway database as we are creating a huge number of studies
        self.config["postgres_config"]["database"] = "throwaway_optuna_db"
        
        optimizer = StrategyOptimizer(root_path=self.root_path.absolute(),
                                    resolution=self.backtesting_interval,
                                    db_client=self.config_helper.create_timescale_client(),
                                    storage_name=StrategyOptimizer.get_storage_name("postgres", create_db_if_not_exists=False, **self.config),
                                    # custom_objective= lambda _, x: x["total_volume"] if x["net_pnl_quote"] > 0 else 0.0,
                                    custom_objective= lambda _, x: [x["net_pnl_quote"], x["total_volume"]],
                                    directions=["maximize", "maximize"],
                                    backtest_offset=self.backtest_offset
                                    )
        
        # Replace local variables with class attributes
        for i, trading_pair in enumerate(self.selected_pairs):
            pair_start_time = time.perf_counter()
            logger.info(f"[{i+1}/{len(self.selected_pairs)}] Processing {self.connector_name} {trading_pair} from {self.human_start} to {self.human_end}")
            
            try:
                self.study_name_prefix = f"{self.name.replace(' ', '_').lower()}_{trading_pair}_{self.backtesting_interval}_{self.n_trials}_{self.study_name_suffix}"
                await optimizer.set_study_name_prefix(self.study_name_prefix)
                total_duration = self.end_time - self.start_time
                num_windows = int(total_duration / self.backtest_window_step)
                
                if self.debug_study_name is not None:
                    num_windows = 1
                
                # Distribute windows among workers
                worker_windows = [self.create_worker_window(idx, self.study_name_prefix) for idx in range(num_windows) if self.should_process_item(idx)]
                if self.is_leader:
                    # Leader will follow up to be sure all windows are completed
                    leader_windows = [self.create_worker_window(idx, self.study_name_prefix) for idx in range(num_windows) if not self.should_process_item(idx)]
                    worker_windows.extend(leader_windows)
                # TODO: make a study_has_completed wrapper function for this
                windows_to_skip = [window for window in worker_windows if optimizer.storage_wrapper.count_trial_state(window.study_name, optuna.trial.TrialState.COMPLETE, self.study_name_prefix) >= self.n_trials]
                if not self.debug_trial and len(windows_to_skip) > 0:
                    logger.info(f"Skipping {len(windows_to_skip)} already completed windows for {trading_pair}")
                    worker_windows = [window for window in worker_windows if window not in windows_to_skip]
                
                # TODO: only cache 2 days of candles at a time as it takes 1 call to timescale per 2 days
                if len(worker_windows) > 0:
                    logger.info(f"Worker {self.worker_id+1}/{self.total_workers} will process {len(worker_windows)}/{num_windows} windows")
                    await optimizer._backtesting_engine.load_candles_cache_for_connector_pair_from_timescale(
                        connector_name=self.connector_name, 
                        trading_pair=trading_pair,
                        intervals=self.required_candle_intervals,
                        # TODO: make buffer part of configuration 
                        start_time = min(window.start for window in worker_windows) - 60 * 5, # add 1 hour buffer for TA calculations and window overhang
                        end_time = max(window.end for window in worker_windows) + 60 * 5,
                        timescale_client=optimizer._db_client
                    )
                
                should_clear_study_cache = self.is_leader
                studies: list[Tuple[optuna.Study, WorkerWindow]] = []
                for window in worker_windows:
                    logger.info(f"Processing window {window.idx + 1}/{num_windows}: {window.human_start} to {window.human_end}")
                    
                    optimize_start_time = time.perf_counter()
                    config_generator = PMMSimpleConfigGenerator(
                        start_date=pd.to_datetime(window.start, unit="s"),
                        end_date=pd.to_datetime(window.end, unit="s"),
                        config={**self.config, "trading_pair": trading_pair}
                    )
                    
                    logger.debug(f"Starting optimization with {self.n_trials} trials for {trading_pair} window {window.idx + 1}")     
                    if self.debug_trial:
                        study = await optimizer.repeat_trial(
                            study_name=window.study_name,
                            trial_number=self.debug_trial,
                            config_generator=config_generator
                        )
                    else:
                        study = await optimizer.optimize(
                            study_name=window.study_name,
                            config_generator=config_generator, 
                            n_trials=self.n_trials
                        )
                    studies.append((study, window))
                    logger.debug(f"Optimization completed in {perf_log_text(optimize_start_time)} for {trading_pair} window {window.idx + 1}")
                    
                    if should_clear_study_cache and not self.should_process_item(window.idx):
                        # reset the study cache so that the leader doesn't attempt to create a study for every study it's collecting 
                        optimizer.reset_study_cache()
                        should_clear_study_cache = False

                # Save all optimal configurations for this trading pair to CSV - only if we're the leader
                if self.is_leader:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    labels_file_name = f"{trading_pair.replace('-', '_')}_optimal_configs_{timestamp}.csv"
                    labels_file = self.root_path / "data" / "labels" / labels_file_name
                    
                    try:
                        data = []
                        label_save_start_time = time.perf_counter()
                        # TODO: load the skipped studies in the same query as the trials or at least all studies at once
                        skipped_studies = [(optimizer.get_study(window.study_name), window) for window in windows_to_skip]
                        logger.info(f"Leader is retrieving {len(skipped_studies)} skipped studies after {perf_log_text(label_save_start_time)} getting studies")
                        studies.extend(skipped_studies)
                        studies.sort(key=lambda x: x[1].start)
                        # NOTE: Got this down to 2 minutes per 1500 100 trial studies 
                        # TODO?: create SQL queries to summarize the data directly 
                        optimizer.storage_wrapper.populate_study_trial_caches([study for study, _ in studies])
                        for study, window in studies:
                            # Get the best trial from the study using the standard method
                            best_trial = max(study.best_trials, key=lambda t: t.values[0])
                            row = best_trial.params.copy()
                            row['t'] = pd.to_datetime(window.start, unit='s')
                            row['value'] = best_trial.values
                            # row['study'] = window.study_name
                            # row['best_trial_number'] = best_trial.number
                            data.append(row)
                        df = pd.DataFrame(data)
                        df.set_index('t', inplace=True)
                        df.to_csv(labels_file)
                        logger.info(f"Saved {len(studies)} optimal configurations to {labels_file} in {perf_log_text(label_save_start_time)}")
                    except Exception as e:
                        logger.error(f"Error saving configurations: {str(e)}")
                        logger.error(traceback.format_exc())
                    
            except Exception as e:
                logger.error(f"Error processing {trading_pair}: {str(e)}")
                logger.error(traceback.format_exc())
            finally:
                await optimizer.dispose()
            
            logger.info(f"Completed {trading_pair} in {perf_log_text(pair_start_time)}")


async def main():
    # Force is needed to override other logging configurations
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    # Run from command line with: python -m tasks.pmm_vml.pmm_vml_label_generation_task --config config/pmm_vml_label_generation_task.yml
    config = BaseTask.load_single_task_config()
    task = PMMVMLLabelGenerationTask("PMM VML Label Generation", config)
    await task.run_once()

if __name__ == "__main__":
    debug = os.environ.get("ASYNC_DEBUG", False)
    asyncio.run(main(), debug=debug) 