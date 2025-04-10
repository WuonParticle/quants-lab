import datetime
from decimal import Decimal
import logging
import math
import os.path
from pathlib import Path
import subprocess
import time
import traceback
import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Type, Dict

import optuna
from dotenv import load_dotenv
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
from hummingbot.strategy_v2.controllers import ControllerConfigBase
from pydantic import BaseModel

from core.backtesting import BacktestingEngine
from core.data_structures.backtesting_result import BacktestingResult
from core.services.timescale_client import TimescaleClient
from core.services.postgres_client import PostgresClient

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestingConfig(BaseModel):
    """
    A simple data structure to hold the backtesting configuration.
    """
    config: ControllerConfigBase
    start: int
    end: int


class BaseStrategyConfigGenerator(ABC):
    """
    Base class for generating strategy configurations for optimization.
    Subclasses should implement the method to provide specific strategy configurations.
    """


    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime,
                 config: Optional[Dict] = None):
        """
        Initialize with common parameters for backtesting.

        Args:
            start_date (datetime.datetime): The start date of the backtesting period.
            end_date (datetime.datetime): The end date of the backtesting period.
        """
        self.start = int(start_date.timestamp())
        self.end = int(end_date.timestamp())
        if config:
            self.config = config
        else:
            self.config = {}

    def update_config(self, config):
        self.config.update(config)

    @abstractmethod
    async def generate_config(self, trial) -> BacktestingConfig:
        """
        Generate the configuration for a given trial.
        This method must be implemented by subclasses.

        Args:
            trial (optuna.Trial): The trial object containing hyperparameters to optimize.

        Returns:
            BacktestingConfig: An object containing the configuration, start time, and end time.
        """
        pass

    async def generate_custom_configs(self) -> List[BacktestingConfig]:
        """
        Generate custom configurations for optimization.
        This method must be implemented by subclasses.

        Returns:
            List[BacktestingConfig]: A list of objects containing the configuration, start time, and end time.
        """
        pass


class StrategyOptimizer:
    """
    Class for optimizing trading strategies using Optuna and a backtesting engine.
    """

    def __init__(self, storage_name: Optional[str] = None, root_path: str = "",
                 load_cached_data: bool = False, resolution: str = "1m", db_client: Optional[TimescaleClient] = None,
                 custom_backtester: Optional[BacktestingEngineBase] = None):
        """
        Initialize the optimizer with a backtesting engine and database configuration.

        Args:
            engine (str): "sqlite" or "postgres".
            root_path (str): Root path for storing database files.
            database_name (str): Name of the SQLite database for storing optimization results.
            load_cached_data (bool): Whether to load cached backtesting data.
            resolution (str): The resolution or time frame of the data (e.g., '1h', '1d').
            db_host (str): Database Host
            db_port (int): Database Port
            db_user (str): Database User
            db_pass (str): Database Password
        """
        self._backtesting_engine = BacktestingEngine(load_cached_data=load_cached_data, root_path=root_path,
                                                     custom_backtester=custom_backtester)
        self._db_client = db_client
        self.resolution = resolution
        self.root_path = root_path
        self._storage_name = storage_name if storage_name else self.get_storage_name(engine="sqlite", root_path=root_path)
        self.dashboard_process = None
        self._custom_objective = None
    
    @classmethod
    def get_storage_name(cls, engine, create_db_if_not_exists: bool = True, **kwargs):
        """
        Get the storage name for the optimization database.
        
        Args:
            engine (str): The database engine to use ("sqlite" or "postgres").
            create_db_if_not_exists (bool): Whether to create the database if it doesn't exist.
            **kwargs: Additional arguments for database configuration.
                      For sqlite: root_path, database_name
                      For postgres: either postgres_config dict
        
        Returns:
            str: The storage name (connection string) for the database.
        """
        if engine == "sqlite":
            root_path = kwargs.get("root_path", "")
            db_name = kwargs.get("database_name", "optimization_database")
            path = os.path.join(root_path, "data", "backtesting", f"{db_name}.db")
            return f"sqlite:///{path}"
        elif engine == "postgres":
            # Let PostgresClient.from_config handle all configuration and db creation
            _, connection_string = PostgresClient.from_config(
                create_db_if_not_exists=create_db_if_not_exists,
                **kwargs  # Pass all kwargs to handle both configuration styles
            )
            return connection_string

    def load_candles_cache_by_connector_pair(self, connector_name: str, trading_pair: str):
        """
        Load the cached candles data for a given connector and trading pair.

        Args:
            connector_name (str): The name of the connector.
            trading_pair (str): The trading pair.
        """
        self._backtesting_engine.load_candles_cache_by_connector_pair(connector_name, trading_pair, root_path=self.root_path)

    async def load_candles_cache_for_connector_pair_from_timescale(self, connector_name: str, trading_pair: str, 
                                                                  intervals: list = None, start_time: int = None, 
                                                                  end_time: int = None):
        """
        Load candles directly from TimescaleDB for a specific connector and trading pair.

        Args:
            connector_name (str): Name of the connector
            trading_pair (str): The trading pair to load candles for
            intervals (list, optional): List of intervals to load (e.g., ["1m", "5m", "15m", "1h"])
            start_time (int, optional): Start timestamp for data retrieval
            end_time (int, optional): End timestamp for data retrieval

        Returns:
            bool: True if any candles were loaded, False otherwise
        """
        if not self._db_client:
            logger.error("TimescaleDB client not initialized. Cannot load candles from database.")
            return False
            
        return await self._backtesting_engine.load_candles_cache_for_connector_pair_from_timescale(
            connector_name=connector_name,
            trading_pair=trading_pair,
            intervals=intervals,
            start_time=start_time,
            end_time=end_time,
            timescale_client=self._db_client
        )

    def get_all_study_names(self):
        """
        Get all the study names available in the database.

        Returns:
            List[str]: A list of study names.
        """
        return optuna.get_all_study_names(self._storage_name)

    def get_study(self, study_name: str):
        """
        Get the study object for a given study name.

        Args:
            study_name (str): The name of the study.

        Returns:
            optuna.Study: The study object.
        """
        return optuna.load_study(study_name=study_name, storage=self._storage_name)

    def get_study_trials_df(self, study_name: str):
        """
        Get the trials data frame for a given study name.

        Args:
            study_name (str): The name of the study.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the trials data.
        """
        study = self.get_study(study_name)
        df = study.trials_dataframe()
        df.dropna(inplace=True)
        # Renaming the columns that start with 'user_attrs_'
        df.rename(columns={col: col.replace('user_attrs_', '') for col in df.columns if col.startswith('user_attrs_')},
                  inplace=True)
        df.rename(columns={col: col.replace('params_', '') for col in df.columns if col.startswith('params_')}, )
        return df

    def get_study_best_params(self, study_name: str):
        """
        Get the best parameters for a given study name.

        Args:
            study_name (str): The name of the study.

        Returns:
            Dict[str, Any]: A dictionary containing the best parameters.
        """
        study = self.get_study(study_name)
        return study.best_params

    def _create_study(self, study_name: str, direction: str = "maximize", load_if_exists: bool = True) -> optuna.Study:
        """
        Create or load an Optuna study for optimization.

        Args:
            study_name (str): The name of the study.
            direction (str): Direction of optimization ("maximize" or "minimize").
            load_if_exists (bool): Whether to load an existing study if available.

        Returns:
            optuna.Study: The created or loaded study.
        """
        return optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=self._storage_name,
            sampler=optuna.samplers.TPESampler(),
            load_if_exists=load_if_exists
        )

    async def optimize(self, study_name: str, config_generator: Type[BaseStrategyConfigGenerator], n_trials: int = 100,
                       load_if_exists: bool = True):
        """
        Run the optimization process asynchronously.

        Args:
            study_name (str): The name of the study.
            config_generator (Type[BaseStrategyConfigGenerator]): A configuration generator class instance.
            n_trials (int): Number of trials to run for optimization.
            load_if_exists (bool): Whether to load an existing study if available.
            num_parallel_trials (int): Number of trials being run in parallel. Just helps with preventing running all trials if some are failing.
        """
        study = self._create_study(study_name, load_if_exists=load_if_exists)
        logger.info(f"About to start optimizing {study_name} with {n_trials} trials.")
        await self._optimize_async(study, config_generator, n_trials=n_trials)

    async def optimize_custom_configs(self, study_name: str, config_generator: Type[BaseStrategyConfigGenerator],
                                      load_if_exists: bool = True):
        """
        Run the optimization process asynchronously using custom configurations.

        Args:
            study_name (str): The name of the study.
            config_generator (Type[BaseStrategyConfigGenerator]): A configuration generator class instance.
            load_if_exists (bool): Whether to load an existing study if available.
        """
        study = self._create_study(study_name, load_if_exists=load_if_exists)
        await self._optimize_async_custom_configs(study, config_generator)

    async def _optimize_async(self, study: optuna.Study, config_generator: Type[BaseStrategyConfigGenerator],
                              n_trials: int):

        trial_attempts = 0
        num_completed_trials = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
        if num_completed_trials >= n_trials:
            logger.warning(f"{study.study_name} study already completed with {num_completed_trials} trials")
            return
        # Only attempt 1 more trial than responsible for to avoid looping if failing
        while (num_completed_trials < n_trials and trial_attempts < math.ceil((n_trials + 1))):
            start_time = time.perf_counter()
            trial = study.ask()
            logger.info(f"Starting {study.study_name} trial {trial.number}/{n_trials}")
            trial_attempts += 1
            try:
                # Run the async objective function and get the result
                value = await self._async_objective(trial, config_generator)
                duration = time.perf_counter() - start_time
                logger.info(f"Trial {trial.number} completed with value: {value} in {duration} seconds")

                # Report the result back to the study
                study.tell(trial, value)

            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {str(e)}")
                traceback.print_exc()
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            finally:
                num_completed_trials = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
        
        logger.info(f"{study.study_name} study completed after {trial_attempts} trials")


    async def _optimize_async_custom_configs(self, study: optuna.Study,
                                             config_generator: Type[BaseStrategyConfigGenerator]):
        """
        Asynchronously optimize using the provided study and configuration generator.

        Args:
            study (optuna.Study): The study to use for optimization.
            config_generator (Type[BaseStrategyConfigGenerator]): A configuration generator class instance.
            n_trials (int): Number of trials to run for optimization.
        """
        backtesting_configs = config_generator.generate_custom_configs()
        await self._db_client.connect()
        for bt_config in backtesting_configs:
            trial = study.ask()
            try:
                connector_name = bt_config.config.connector_name
                trading_pair = bt_config.config.trading_pair
                start = bt_config.start
                end = bt_config.end

                trial.set_user_attr("config", bt_config.config.json())
                trial.set_user_attr("start_bt", start)
                trial.set_user_attr("end_bt", end)
                candles = await self._db_client.get_candles(connector_name,
                                                            trading_pair,
                                                            self.resolution, start, end)
                self._backtesting_engine._dt_bt.backtesting_data_provider.candles_feeds[
                    f"{connector_name}_{trading_pair}_{self.resolution}"] = candles.data
                self._backtesting_engine._mm_bt.backtesting_data_provider.candles_feeds[
                    f"{connector_name}_{trading_pair}_{self.resolution}"] = candles.data
                config_generator.backtester.backtesting_data_provider.candles_feeds[
                    f"{connector_name}_{trading_pair}_{self.resolution}"] = candles.data
                start = candles.data["timestamp"].min()
                end = candles.data["timestamp"].max()
                # Generate configuration using the config generator
                backtesting_result = await self._backtesting_engine.run_backtesting(
                    config=bt_config.config,
                    start=start,
                    end=end,
                    backtesting_resolution=self.resolution,
                )
                strategy_analysis = backtesting_result.results

                for key, value in strategy_analysis.items():
                    trial.set_user_attr(key, value)
                executors_df = backtesting_result.executors_df.copy()
                trial.set_user_attr("executors", executors_df.to_json())
                executors_df["close_type"] = executors_df["close_type"].apply(lambda x: x.name)
                executors_df["status"] = executors_df["status"].apply(lambda x: x.name)

                # Return the value you want to optimize
                value = strategy_analysis["net_pnl"]
            except Exception as e:
                print(f"An error occurred during optimization: {str(e)}")
                traceback.print_exc()
                value = float('-inf')  # Return a very low value to indicate failure

            # Report the result back to the study
            study.tell(trial, value)

    def set_custom_objective(self, objective_function):
        """
        Set a custom objective function for optimization.
        
        Args:
            objective_function: A callable that takes (trial, results) and returns a float 
                              representing the objective value to optimize.
        """
        self._custom_objective = objective_function

    def get_study_best_trial(self, study_name: str):
        """
        Get the best trial for a given study name.
        
        Args:
            study_name (str): The name of the study.
            
        Returns:
            optuna.Trial: The best trial from the study.
        """
        study = self.get_study(study_name)
        if study and len(study.trials) > 0:
            return study.best_trial
        return None

    async def _async_objective(self, trial: optuna.Trial, config_generator: Type[BaseStrategyConfigGenerator]) -> float:
        """
        The asynchronous objective function for a given trial.

        Args:
            trial (optuna.Trial): The trial object containing hyperparameters.
            config_generator (Type[BaseStrategyConfigGenerator]): A configuration generator class instance.

        Returns:
            float: The objective value to be optimized.
        """
        try:
            # Generate configuration using the config generator
            backtesting_result = await self._async_run_backtesting(trial, config_generator)
            strategy_analysis = backtesting_result.results
            # Use custom objective function if provided, otherwise use default
            if self._custom_objective:
                return self._custom_objective(trial, strategy_analysis)
            
            # Default objective: sharpe ratio
            return strategy_analysis["sharpe_ratio"]
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            traceback.print_exc()
            return float('-inf')  # Return a very low value to indicate failure

    async def _async_run_backtesting(self, trial: optuna.Trial, config_generator: Type[BaseStrategyConfigGenerator]) -> BacktestingResult:
        backtesting_config = await config_generator.generate_config(trial)
        # Await the backtesting result
        backtesting_result = await self._backtesting_engine.run_backtesting(
            config=backtesting_config.config,
            start=backtesting_config.start,
            end=backtesting_config.end,
            backtesting_resolution=self.resolution,
        )
        strategy_analysis = backtesting_result.results

        for key, value in strategy_analysis.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("config", backtesting_result.controller_config.json())
        executors_df = backtesting_result.executors_df.copy()
        executors_df["close_type"] = executors_df["close_type"].apply(lambda x: x.name)
        executors_df["status"] = executors_df["status"].apply(lambda x: x.name)
        executors_df.drop(columns=["config"], inplace=True)
        trial.set_user_attr("executors", executors_df.to_json())

        return backtesting_result
    
    def launch_optuna_dashboard(self):
        """
        Launch the Optuna dashboard for visualization.
        """
        self.dashboard_process = subprocess.Popen(["optuna-dashboard", self._storage_name])

    def kill_optuna_dashboard(self):
        """
        Kill the Optuna dashboard process.
        """
        if self.dashboard_process and self.dashboard_process.poll() is None:
            self.dashboard_process.terminate()  # Graceful termination
            self.dashboard_process.wait()  # Wait for process to terminate
            self.dashboard_process = None  # Reset process handle
        else:
            print("Dashboard is not running or already terminated.")
            
    async def repeat_trial(self, study_name: str, trial_number: int, config_generator: Type[BaseStrategyConfigGenerator]):
        """
        Repeat a specific trial multiple times for debugging purposes.
        
        This is useful when a trial returns inconsistent or invalid results (like Inf values)
        and you want to debug the issue by running the same parameters multiple times.
        
        Args:
            study_name (str): The name of the study containing the trial.
            trial_number (int): The trial number to repeat.
            config_generator (Type[BaseStrategyConfigGenerator]): Configuration generator for the trial.
            
        Returns:
            list: A list containing the results of each trial repetition.
        """
        study = self.get_study(study_name)
        if not study:
            logger.error(f"Study {study_name} not found")
            return None
            
        target_trial = next((trial for trial in study.trials if trial.number == trial_number), None)
                
        if not target_trial:
            logger.error(f"Trial {trial_number} not found in study {study_name}")
            return None
            
        # Create a dummy trial with the same parameters for testing
        params = target_trial.params
        logger.info(f"Repeating trial {trial_number} with parameters: {params}")
        
        try:
            # Run the objective function for this trial
            backtesting_result = await self._async_run_backtesting(target_trial, config_generator)
            strategy_analysis = backtesting_result.results
            
            
            logger.info(f"Debug trial returned results: {strategy_analysis}")
            return backtesting_result
            
        except Exception as e:
            logger.error(f"Error in debug trial {trial_number}: {str(e)}")
            traceback.print_exc()
            
    async def save_best_config_to_yaml(self, study_name: str, output_path: str, config_generator: Type[BaseStrategyConfigGenerator]):
        """
        Save the best configuration from a study to a YAML file.
        
        Args:
            study_name (str): The name of the study to extract best parameters from.
            output_path (str): The file path where to save the YAML file.
            config_generator: The configuration generator instance that contains the logic to generate configs.
        
        Returns:
            bool: True if the configuration was saved successfully, False otherwise.
        """
        try:
            import yaml
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
            # Get the best trial from the study
            best_trial = self.get_study_best_trial(study_name)
            backtesting_config = await config_generator.generate_config(best_trial)
            config = backtesting_config.config.dict()
            
            # Generate a unique ID
            trading_pair = config.get("trading_pair", "unknown")
            trading_pair_id = trading_pair.replace("-", "_")
            config_id = f"{config.get('controller_name', 'strategy')}_{trading_pair_id}_{study_name}"
            config["id"] = config_id
            performance = best_trial.user_attrs.get("performance", {})
            # Remove performance metrics if present
            # TODO: check if this is required.
            if "performance" in config:
                del config["performance"]
            
            # Ensure the output directory exists
            # TODO use Path library
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save to YAML file
            with open(output_path, "w") as f:
                def decimal_representer(dumper, data):
                    return dumper.represent_float(float(data))
                yaml.add_representer(Decimal, decimal_representer)
                yaml.dump(config, f, default_flow_style=False)
                
            logger.info(f"Saved best configuration to {output_path}")
            return performance
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            logger.error(traceback.format_exc())
            return False