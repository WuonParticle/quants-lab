from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, List
import pandas as pd
import time
import logging
from dataclasses import dataclass

from core.services.timescale_client import TimescaleClient
from core.services.mongodb_client import MongoClient
from core.data_structures.trading_rules import TradingRules

logger = logging.getLogger(__name__)

@dataclass
class BacktestTimeRange:
    start_date: float
    end_date: float
    human_start: str
    human_end: str
    backtest_window_step: Optional[int] = None
    backtest_window_size: Optional[int] = None

    def __iter__(self):
        # For backward compatibility, allow unpacking the first 4 values
        return iter((self.start_date, self.end_date, self.human_start, self.human_end))

    def for_window(self):
        return iter((self.start_date, self.end_date, self.human_start, self.human_end, self.backtest_window_size, self.backtest_window_step))

class TaskConfigHelper:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def create_timescale_client(self) -> TimescaleClient:
        """Create a TimescaleClient from the configuration"""
        return TimescaleClient(
            host=self.config["timescale_config"].get("host", "localhost"),
            port=self.config["timescale_config"].get("port", 5432),
            user=self.config["timescale_config"].get("user", "admin"),
            password=self.config["timescale_config"].get("password", "admin"),
            database=self.config["timescale_config"].get("database", "timescaledb")
        )
    
    def create_mongo_client(self) -> MongoClient:
        """Create a MongoClient from the configuration"""
        return MongoClient(**self.config.get("db_config", {}))
    
    def filter_trading_rules(self, trading_rules: TradingRules, logger=None) -> TradingRules:
        """
        Filter trading rules based on base assets, quote assets, and min notional size from config.
        
        Args:
            trading_rules: The TradingRules object to filter
            logger: Optional logger to log filtering progress
            
        Returns:
            Filtered TradingRules object
        """
        date_format = "%Y-%m-%d %H:%M:%S UTC"
        now = datetime.now().strftime(date_format)
        
        # Get filter parameters from config
        base_asset_filter = self.config.get("base_asset_filter", None)
        quote_asset_filter = self.config.get("quote_asset_filter", None)
        min_notional_size = float(self.config.get("min_notional_size", 0))
        
        initial_count = len(trading_rules.data)
        if logger:
            logger.info(f"{now} - Initial trading rules count: {initial_count}")
        
        # Filter by base assets if specified
        if base_asset_filter:
            # Create a list of filtered rules data and flatten it
            all_rules_data = sum([trading_rules.filter_by_base_asset(asset).data for asset in base_asset_filter], [])
            # Create a new TradingRules with deduplicated pairs
            trading_rules = TradingRules(list({pair.trading_pair: pair for pair in all_rules_data}.values()))
            if logger:
                logger.info(f"{now} - After base asset filter: {len(trading_rules.data)} pairs")
        
        # Filter by quote assets if specified
        if quote_asset_filter:
            # Create a list of filtered rules data and flatten it
            all_rules_data = sum([trading_rules.filter_by_quote_asset(asset).data for asset in quote_asset_filter], [])
            # Create a new TradingRules with deduplicated pairs
            trading_rules = TradingRules(list({pair.trading_pair: pair for pair in all_rules_data}.values()))
            if logger:
                logger.info(f"{now} - After quote asset filter: {len(trading_rules.data)} pairs")
        
        # Apply minimum notional size filter if specified
        if min_notional_size > 0:
            trading_rules = trading_rules.filter_by_min_notional_size(min_notional_size)
            if logger:
                logger.info(f"{now} - After min notional filter: {len(trading_rules.data)} pairs")
        
        return trading_rules
    
    def _get_end_time(self) -> float:
        """Get end time from config or current time"""
        if "end_time" in self.config:
            # Parse ISO format
            if isinstance(self.config["end_time"], str):
                try:
                    return datetime.fromisoformat(self.config["end_time"].replace('Z', '+00:00')).timestamp()
                except ValueError:
                    pass
            # Use raw timestamp
            return float(self.config["end_time"])
        
        # Default to current time
        return datetime.now().timestamp()
    
    def _get_start_time(self, end_time: float) -> float:
        """Get start time from config or calculate from days_to_analyze"""
        if "start_time" in self.config:
            # Parse ISO format
            if isinstance(self.config["start_time"], str):
                try:
                    return datetime.fromisoformat(self.config["start_time"].replace('Z', '+00:00')).timestamp()
                except ValueError:
                    pass
            # Use raw timestamp
            return float(self.config["start_time"])
        
        # Calculate from days_to_analyze
        days_to_analyze = self.config.get("days_to_analyze", 30)
        return end_time - (days_to_analyze * 24 * 60 * 60)

    def get_backtesting_time_range(self) -> BacktestTimeRange:
        """
        Get the backtesting time range based on configuration.
        Returns a BacktestTimeRange object containing start/end timestamps and human-readable times.
        """
        # Try to get absolute time range first
        start_time_str = self.config.get("start_time")
        end_time_str = self.config.get("end_time")
       
        # TODO: support end_time + lookback_days
        if start_time_str and end_time_str:
            try:
                start_date = pd.to_datetime(start_time_str).timestamp()
                end_date = pd.to_datetime(end_time_str).timestamp()
                logger.info("Using absolute time range from config")
            except Exception as e:
                raise ValueError(f"Invalid datetime format in config. Please use format 'YYYY-MM-DD HH:MM:SS'. Error: {str(e)}")
        else:
            # Fall back to relative time range
            end_time_buffer_hours = self.config.get("end_time_buffer_hours", 6)
            lookback_days = self.config.get("lookback_days", 1)
            
            end_date = time.time() - end_time_buffer_hours * 3600
            start_date = end_date - lookback_days * 24 * 60 * 60
            logger.info(f"Using relative time range: {lookback_days} days lookback with {end_time_buffer_hours} hours buffer")
        
        human_start = datetime.fromtimestamp(start_date).strftime('%Y-%m-%d %H:%M:%S')
        human_end = datetime.fromtimestamp(end_date).strftime('%Y-%m-%d %H:%M:%S')
       
        return BacktestTimeRange(
            start_date=start_date,
            end_date=end_date,
            human_start=human_start,
            human_end=human_end,
            backtest_window_step=self.config.get("backtest_window_step", None),
            backtest_window_size=self.config.get("backtest_window_size", None)
        )