import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, List

from core.services.timescale_client import TimescaleClient
from core.services.mongodb_client import MongoClient
from core.data_structures.trading_rules import TradingRules


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
    
    async def get_candles(self, trading_pair: str, interval: str):
        """Get candles for a trading pair and interval using config parameters"""
        # Create client
        client = self.create_timescale_client()
        
        try:
            # Connect to the database
            await client.connect()
            
            # Get connector name from config
            connector_name = self.config["connector_name"]
            
            # Calculate start and end times
            end_time = self._get_end_time()
            start_time = self._get_start_time(end_time)
            
            # Get candles
            candles = await client.get_candles(
                connector_name=connector_name,
                trading_pair=trading_pair,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
            
            return candles
        finally:
            # Clean up client
            if client:
                await client.close()
    
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