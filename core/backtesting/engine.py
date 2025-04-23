import logging
import os
import random
from typing import Dict, Optional
import traceback
import asyncpg
import asyncio

import pandas as pd

from core.data_structures.backtesting_result import BacktestingResult
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
from hummingbot.strategy_v2.controllers import ControllerConfigBase
from core.services.timescale_client import TimescaleClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestingEngine:
    def __init__(self, load_cached_data: bool = True, root_path: str = "", custom_backtester: Optional[BacktestingEngineBase] = None):
        self._bt_engine = custom_backtester if custom_backtester is not None else BacktestingEngineBase()
        self.root_path = root_path
        if load_cached_data:
            self._load_candles_cache(root_path)

    def _load_candles_cache(self, root_path: str):
        all_files = os.listdir(os.path.join(root_path, "data", "candles"))
        for file in all_files:
            if file == ".gitignore":
                continue
            try:
                connector_name, trading_pair, interval = file.split(".")[0].split("|")
                candles = pd.read_parquet(os.path.join(root_path, "data", "candles", file))
                candles.index = pd.to_datetime(candles.timestamp, unit='s')
                candles.index.name = None
                columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                           'n_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']
                for column in columns:
                    candles[column] = pd.to_numeric(candles[column])
                self._bt_engine.backtesting_data_provider.candles_feeds[
                    f"{connector_name}_{trading_pair}_{interval}"] = candles
                # TODO: evaluate start and end time for each feed
                start_time = candles["timestamp"].min()
                end_time = candles["timestamp"].max()
                self._bt_engine.backtesting_data_provider.start_time = start_time
                self._bt_engine.backtesting_data_provider.end_time = end_time
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

    def load_candles_cache_by_connector_pair(self, connector_name: str, trading_pair: str, root_path: str = ""):
            all_files = os.listdir(os.path.join(root_path, "data", "candles"))
            for file in all_files:
                if file == ".gitignore":
                    continue
                try:
                    if connector_name in file and trading_pair in file:
                        connector_name, trading_pair, interval = file.split(".")[0].split("|")
                        candles = pd.read_parquet(os.path.join(root_path, "data", "candles", file))
                        candles.index = pd.to_datetime(candles.timestamp, unit='s')
                        candles.index.name = None
                        columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                                   'n_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']
                        for column in columns:
                            candles[column] = pd.to_numeric(candles[column])
                        self._bt_engine.backtesting_data_provider.candles_feeds[
                            f"{connector_name}_{trading_pair}_{interval}"] = candles
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")

    async def load_candles_cache_for_connector_pair_from_timescale(self, connector_name: str,
                                                                trading_pair: str,
                                                                intervals: list = None,
                                                                start_time: int = None,
                                                                end_time: int = None,
                                                                timescale_client: TimescaleClient = None):
        """
        Load candles directly from TimescaleDB for a specific connector and trading pair.
        
        Args:
            connector_name: Name of the connector
            trading_pair: The trading pair to load candles for
            intervals: List of intervals to load (e.g., ["1m", "5m", "15m", "1h"])
            start_time: Start time to load candles from
            end_time: End time to load candles to
            timescale_client: TimescaleDB client
            
        Returns:
            bool: True if any candles were loaded, False otherwise
        """
        if not intervals:
            intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
            
        try:
            # Connect to database with retry
            for attempt in range(3):
                try:
                    await timescale_client.connect()
                    break
                except (asyncpg.exceptions.InsufficientResourcesError, 
                       asyncpg.exceptions.PostgresConnectionError) as e:
                    if attempt == 2:  # Last attempt
                        raise
                    jitter = random.uniform(0, 0.5)
                    await asyncio.sleep((2 ** attempt) + jitter)  # Exponential backoff with jitter
            
            loaded_any = False
            
            # Process each interval
            for interval in intervals:
                for attempt in range(3):
                    try:
                        # Get candles from database
                        candles = await timescale_client.get_candles(
                            connector_name=connector_name,
                            trading_pair=trading_pair,
                            interval=interval,
                            start_time=start_time,
                            end_time=end_time
                        )
                        
                        if candles.data.empty:
                            logger.warning(f"No {interval} candles found in TimescaleDB for {connector_name}_{trading_pair}")
                            continue  # No point retrying if data is empty
                        
                        # Format candles for backtesting
                        candles_df = candles.data.copy()
                        candles_df.index = pd.to_datetime(candles_df.timestamp, unit='s')
                        candles_df.index.name = None
                        
                        # Process numeric columns
                        columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                                  'n_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']
                        for column in columns:
                            if column in candles_df.columns:
                                candles_df[column] = pd.to_numeric(candles_df[column])
                        
                        feed_key = f"{connector_name}_{trading_pair}_{interval}"
                        self._bt_engine.backtesting_data_provider.candles_feeds[feed_key] = candles_df
                        
                        logger.info(f"Loaded {len(candles_df)} {interval} candles for {connector_name}_{trading_pair}")
                        loaded_any = True
                        break  # Success, exit retry loop
                    except (asyncpg.exceptions.InsufficientResourcesError, 
                           asyncpg.exceptions.PostgresConnectionError) as e:
                        if attempt == 2:  # Last attempt
                            logger.error(f"Failed to load {interval} candles for {connector_name}_{trading_pair} after 3 attempts: {e}")
                            raise
                        jitter = random.uniform(0, 0.5)
                        await asyncio.sleep((2 ** attempt) + jitter)  # Exponential backoff with jitter
                    except Exception as e:
                        logger.error(f"Error loading {interval} candles for {connector_name}_{trading_pair}: {e}")
                        logger.error(traceback.format_exc())
                        raise  # Propagate other exceptions immediately
            
            # Close database connection
            await timescale_client.close()
            
            return loaded_any
            
        except Exception as e:
            logger.error(f"Error loading candles from TimescaleDB: {e}")
            logger.error(traceback.format_exc())
            raise  # Propagate the exception instead of returning False

    def get_controller_config_instance_from_dict(self, config: Dict):
        return BacktestingEngineBase.get_controller_config_instance_from_dict(
            config_data=config,
            controllers_module="controllers",
        )

    async def run_backtesting(self, config: ControllerConfigBase, start: int,
                              end: int, backtesting_resolution: str, trade_cost: float = 0.0006,
                              backtest_offset: int = 0) -> BacktestingResult:
        bt_result = await self._bt_engine.run_backtesting(config, start, end, backtesting_resolution, trade_cost, backtest_offset)
        return BacktestingResult(bt_result, config)

    async def backtest_controller_from_yml(self,
                                           config_file: str,
                                           controllers_conf_dir_path: str,
                                           start: int,
                                           end: int,
                                           backtesting_resolution: str = "1m",
                                           trade_cost: float = 0.0006,
                                           backtester: Optional[BacktestingEngineBase] = None):
        config = self._bt_engine.get_controller_config_instance_from_yml(config_file, controllers_conf_dir_path)
        return await self.run_backtesting(config, start, end, backtesting_resolution, trade_cost, backtester)
