import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from functools import reduce
import operator

import pandas as pd
from dotenv import load_dotenv

from core.data_sources import CLOBDataSource
from core.services.timescale_client import TimescaleClient
from core.task_base import BaseTask
from core.data_structures.trading_rules import TradingRules
from core.task_config_helpers import TaskConfigHelper

logging.basicConfig(level=logging.INFO)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
load_dotenv()


class CandlesDownloaderTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.connector_name = config["connector_name"]
        self.days_data_retention = config.get("days_data_retention", 7)
        self.intervals = config.get("intervals", ["1m"])
        self.min_notional_size = Decimal(str(config.get("min_notional_size", 0)))
        self.base_asset_filter = config.get("base_asset_filter", None)
        self.quote_asset_filter = config.get("quote_asset_filter", None)
        self.cleanup_old_data = config.get("cleanup_old_data", False)
        
        # Initialize helpers
        self.config_helper = TaskConfigHelper(config)
        self.clob = CLOBDataSource()

    async def execute(self):
        date_format = "%Y-%m-%d %H:%M:%S UTC"
        logging.info(f"Starting {self.__class__.__name__} for {self.connector_name}")
        
        # Calculate time range for data retention
        end_time = datetime.now(timezone.utc)
        start_time = pd.Timestamp(time.time() - self.days_data_retention * 24 * 60 * 60,
                                  unit="s").tz_localize(timezone.utc)
        logging.info(f"Start date: {start_time.strftime(date_format)}, End date: {end_time.strftime(date_format)}")
        if self.cleanup_old_data:
            logging.info(f"Old data cleanup is enabled. Data older than {self.days_data_retention} days will be deleted.")

        # Create database client using helper
        timescale_client = self.config_helper.create_timescale_client()
        logging.info(f"TimescaleDB Client Properties:")
        logging.info(vars(timescale_client))  # vars() shows all instance variables
        await timescale_client.connect()

        try:
            # Get all trading rules
            trading_rules = await self.clob.get_trading_rules(self.connector_name)
            
            # Apply filters using the helper method
            trading_rules = self.config_helper.filter_trading_rules(trading_rules, logger=logging)
            
            # Get final trading pairs
            trading_pairs = trading_rules.get_all_trading_pairs()
            logging.info(f"Final trading pairs: {trading_pairs}")
            
            # Process each trading pair and interval
            for i, trading_pair in enumerate(trading_pairs):
                for interval in self.intervals:
                    logging.info(f"Fetching candles for {trading_pair} [{i+1} from {len(trading_pairs)}]")
                    try:
                        table_name = timescale_client.get_ohlc_table_name(self.connector_name, trading_pair, interval)
                        await timescale_client.create_candles_table(table_name)
                        last_candle_timestamp = await timescale_client.get_last_candle_timestamp(
                            connector_name=self.connector_name,
                            trading_pair=trading_pair,
                            interval=interval)
                        from_time = last_candle_timestamp if last_candle_timestamp else start_time.timestamp()
                        
                        # TODO grab both first and last candle timestamp in case days_data_retention is increased
                        #       when this is the case then we need to do an extra query to get the missing beginning candles
                        # Get candles
                        candles = await self.clob.get_candles(
                            self.connector_name,
                            trading_pair,
                            interval,
                            int(from_time),
                            # int(start_time.timestamp()),
                            int(end_time.timestamp()),
                        )

                        if candles.data.empty:
                            logging.info(f"No new trades for {trading_pair}")
                            continue

                        # Store candles in database
                        await timescale_client.append_candles(table_name=table_name,
                                                            candles=candles.data.values.tolist())
                        
                        # Cleanup old data if enabled
                        if self.cleanup_old_data:
                            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                            cutoff_timestamp = (today_start - timedelta(days=self.days_data_retention)).timestamp()
                            await timescale_client.delete_candles(
                                connector_name=self.connector_name,
                                trading_pair=trading_pair,
                                interval=interval,
                                timestamp=cutoff_timestamp
                            )
                            logging.info(f"Cleaned up data older than {datetime.fromtimestamp(cutoff_timestamp).strftime(date_format)} for {trading_pair} @ {interval}")
                        
                        await asyncio.sleep(1)
                    except Exception as e:
                        logging.exception(
                            f"An error occurred during the data load for trading pair {trading_pair}:\n {e}")
                        continue
                logging.info(f"Done Fetching candles for {trading_pair}")
        except Exception as e:
            logging.exception(f"Error executing task: {str(e)}")
            raise
        finally:
            if timescale_client:
                await timescale_client.close()


async def main(config):
    candles_downloader_task = CandlesDownloaderTask(
        name="Candles Downloader",
        frequency=timedelta(hours=1),
        config=config
    )
    await candles_downloader_task.execute()

if __name__ == "__main__":
    timescale_config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": os.getenv("TIMESCALE_PORT", 5432),
        "user": os.getenv("TIMESCALE_USER", "admin"),
        "password": os.getenv("TIMESCALE_PASSWORD", "admin"),
        "database": os.getenv("TIMESCALE_DB", "timescaledb")
    }
    config = {
<<<<<<< HEAD
        "connector_name": "binance_perpetual",
        "quote_asset": "USDT",
=======
        "connector_name": "hyperliquid_perpetual",
>>>>>>> 12fbdc7 (extend candle loader task with additional config options)
        "intervals": ["15m", "1h"],
        "days_data_retention": 30,
        "min_notional_size": 10,
        "timescale_config": timescale_config
    }
    asyncio.run(main(config))
