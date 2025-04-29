import asyncio
import datetime
import logging
import os
import re
import time
import ast
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler

from core.task_base import BaseTask
from core.task_config_helpers import TaskConfigHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMMVMLFeatureGenerationTask(BaseTask):
    """
    Task that generates features and prepares labels for training a ML model
    to predict optimal parameters for the pmm_simple market making strategy.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        frequency = None  # We'll use run_once for this task
        super().__init__(name, frequency, config)
        
        # Configuration parameters
        self.connector_name = config.get("connector_name", "WTF")
        self.trading_pairs = config.get("trading_pairs", [])
        self.label_data_path = Path(config.get("label_data_path", "data/labels/"))
        self.feature_output_path = Path(config.get("feature_output_path", "data/features_labels/"))
        self.candle_intervals = config.get("candle_intervals", ["1m"])
        self.label_lookback_minutes = config.get("label_lookback_minutes", 1440)
        self.feature_generation_config = config.get("feature_generation_config", {})
        self.smoothing_config = config.get("smoothing_config", {
            "max_smoothing_gap_minutes": 30,
            "edge_case_fill_method": "ffill"
        })
        
        # Create output directory if it doesn't exist
        os.makedirs(self.feature_output_path, exist_ok=True)
        
        # Initialize timescale client
        self.config_helper = TaskConfigHelper(config)
        self.timescale_client = self.config_helper.create_timescale_client()
    
    def get_latest_label_file(self, trading_pair: str) -> Optional[Path]:
        """Get the latest label file for a trading pair"""
        normalized_pair = trading_pair.replace('-', '_')
        pattern = f"{normalized_pair}_optimal_configs_*.csv"
        
        label_files = list(self.label_data_path.glob(pattern))
        if not label_files:
            logger.warning(f"No label files found for {trading_pair} matching pattern {pattern}")
            return None
            
        # Sort by file modification time (most recent first)
        label_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        logger.info(f"Found {len(label_files)} label files for {trading_pair}, using most recent: {label_files[0].name}")
        return label_files[0]
    
    def parse_values_column(self, value_str: str) -> Tuple[float, float]:
        """Parse the 'value' column from string representation to tuple of (pnl, volume)"""
        try:
            # Parse the string representation of list [pnl, volume]
            values = ast.literal_eval(value_str)
            return (values[0], values[1])
        except (ValueError, SyntaxError, IndexError) as e:
            logger.error(f"Error parsing value column: {e}, value: {value_str}")
            return (0.0, 0.0)
    
    def smooth_labels(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fuzzy logic smoothing to label parameters where objective value is [0.0, 0.0]
        """
        logger.info(f"Smoothing labels for dataframe with shape {labels_df.shape}")
        
        # Parse the 'value' column to extract pnl and volume
        if 'value' in labels_df.columns:
            labels_df[['pnl', 'volume']] = pd.DataFrame(
                labels_df['value'].apply(self.parse_values_column).tolist(), 
                index=labels_df.index
            )
            
            # Identify rows where both pnl and volume are 0.0
            zero_mask = (labels_df['pnl'] == 0.0) & (labels_df['volume'] == 0.0)
            
            # Columns to smooth (the 6 parameters)
            param_columns = ['buy_0_bps', 'buy_1_step', 'sell_0_bps', 'sell_1_step', 
                            'take_profit_pct', 'stop_loss_pct']
            
            # Create a copy to avoid SettingWithCopyWarning
            smoothed_df = labels_df.copy()
            
            # Get the max gap to interpolate across (in minutes)
            max_gap = self.smoothing_config.get("max_smoothing_gap_minutes", 30)
            max_time_diff = pd.Timedelta(minutes=max_gap)
            
            # Edge case handling method
            edge_case_method = self.smoothing_config.get("edge_case_fill_method", "ffill")
            
            # Count of rows that need smoothing
            rows_to_smooth = zero_mask.sum()
            logger.info(f"Found {rows_to_smooth} rows that need smoothing")
            
            if rows_to_smooth > 0:
                # For each row with zero objective values
                for idx in labels_df[zero_mask].index:
                    current_time = idx
                    
                    # Find the nearest previous valid row
                    prev_valid_mask = (~zero_mask) & (labels_df.index < current_time)
                    if prev_valid_mask.any():
                        prev_valid_idx = labels_df[prev_valid_mask].index.max()
                        prev_time_diff = current_time - prev_valid_idx
                    else:
                        prev_valid_idx = None
                        prev_time_diff = None
                    
                    # Find the nearest next valid row
                    next_valid_mask = (~zero_mask) & (labels_df.index > current_time)
                    if next_valid_mask.any():
                        next_valid_idx = labels_df[next_valid_mask].index.min()
                        next_time_diff = next_valid_idx - current_time
                    else:
                        next_valid_idx = None
                        next_time_diff = None
                    
                    # Check if we can interpolate between valid neighbors
                    if (prev_valid_idx is not None and next_valid_idx is not None and
                        prev_time_diff <= max_time_diff and next_time_diff <= max_time_diff):
                        # Calculate interpolation weight
                        total_diff = (next_valid_idx - prev_valid_idx).total_seconds()
                        current_diff = (current_time - prev_valid_idx).total_seconds()
                        weight = current_diff / total_diff
                        
                        # Interpolate each parameter
                        for param in param_columns:
                            prev_val = labels_df.loc[prev_valid_idx, param]
                            next_val = labels_df.loc[next_valid_idx, param]
                            smoothed_df.loc[idx, param] = prev_val * (1 - weight) + next_val * weight
                    
                    # Handle edge cases where interpolation isn't possible
                    elif edge_case_method != "drop":
                        if edge_case_method == "ffill" and prev_valid_idx is not None:
                            # Forward fill from previous valid row
                            for param in param_columns:
                                smoothed_df.loc[idx, param] = labels_df.loc[prev_valid_idx, param]
                        elif edge_case_method == "bfill" and next_valid_idx is not None:
                            # Backward fill from next valid row
                            for param in param_columns:
                                smoothed_df.loc[idx, param] = labels_df.loc[next_valid_idx, param]
                
                # Drop rows that couldn't be smoothed if required
                if edge_case_method == "drop":
                    # Recalculate zero mask to find rows still with zero objective values
                    remain_zero_mask = (smoothed_df['pnl'] == 0.0) & (smoothed_df['volume'] == 0.0)
                    smoothed_df = smoothed_df[~remain_zero_mask]
                    logger.info(f"Dropped {remain_zero_mask.sum()} rows that couldn't be smoothed")
            
            return smoothed_df
        else:
            logger.warning("No 'value' column found in labels dataframe")
            return labels_df
    
    def generate_features(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical analysis features for the given candle data
        """
        logger.info(f"Generating features for dataframe with shape {candles_df.shape}")
        
        # Create a copy to work with
        df_with_indicators = candles_df.copy()
        
        # Log the columns available for debugging
        logger.info(f"Available columns in candles data: {df_with_indicators.columns.tolist()}")
        
        # Make sure required columns exist for TA indicators
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df_with_indicators.columns:
                logger.error(f"Required column '{col}' not found in candles data")
                raise ValueError(f"Required column '{col}' not found in candles data")
                
        # Bollinger Bands with different lengths
        bb_length_1 = self.feature_generation_config.get("bbands_length_1", 20)
        bb_std_1 = self.feature_generation_config.get("bbands_std_1", 2.0)
        bb_length_2 = self.feature_generation_config.get("bbands_length_2", 50)
        bb_std_2 = self.feature_generation_config.get("bbands_std_2", 2.0)
        
        df_with_indicators.ta.bbands(length=bb_length_1, std=bb_std_1, append=True)
        df_with_indicators.ta.bbands(length=bb_length_2, std=bb_std_2, append=True)
        
        # MACD with different parameters
        df_with_indicators.ta.macd(fast=12, slow=26, signal=9, append=True)  # Standard MACD
        df_with_indicators.ta.macd(fast=8, slow=21, signal=5, append=True)  # Faster MACD
        
        # RSI with different lengths
        df_with_indicators.ta.rsi(length=14, append=True)  # Standard RSI
        df_with_indicators.ta.rsi(length=21, append=True)  # Longer RSI
        
        # Moving averages
        df_with_indicators.ta.sma(length=20, append=True)  # Short MA
        df_with_indicators.ta.sma(length=50, append=True)  # Medium MA
        df_with_indicators.ta.ema(length=20, append=True)  # Short EMA
        df_with_indicators.ta.ema(length=50, append=True)  # Medium EMA
        
        # Volatility and momentum indicators
        df_with_indicators.ta.atr(length=14, append=True)  # ATR
        df_with_indicators.ta.stoch(k=14, d=3, append=True)  # Stochastic
        df_with_indicators.ta.adx(length=14, append=True)  # ADX
        
        # Process further to match expected format
        df_processed = df_with_indicators.copy()
        
        # Convert prices to returns
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df_processed[f'{col}_ret'] = df_processed[col].pct_change()
        df_processed = df_processed.drop(columns=price_columns)
        
        # Create volume-based features with just the available 'volume' column
        if 'volume' in df_processed.columns:
            # Volume change rate
            df_processed['volume_change'] = df_processed['volume'].pct_change()
            
            # Volume SMA
            df_processed['volume_sma_10'] = df_processed['volume'].rolling(window=10).mean() / df_processed['volume']
            
            # Relative volume (compared to moving average)
            df_processed['rel_volume'] = df_processed['volume'] / df_processed['volume'].rolling(window=20).mean()
        else:
            logger.warning("Volume column not found, using placeholders for volume features")
            df_processed['volume_change'] = 0
            df_processed['volume_sma_10'] = 1
            df_processed['rel_volume'] = 1
            df_processed['buy_volume_ratio'] = 0.5
        
        # Additional unnecessary columns to drop if they exist
        columns_to_drop = [
            'taker_buy_base_volume', 'volume', 'close_time', 
            'taker_buy_volume', 'number_of_trades', 'buy_volume',
            'sell_volume', 'unused', 'quote_asset_volume', 'taker_buy_quote_volume'
        ]
        # Only drop columns that actually exist
        cols_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        if cols_to_drop:
            df_processed = df_processed.drop(columns=cols_to_drop)
        
        # Drop any rows with NaN values
        df_processed = df_processed.dropna()
        
        # Don't standardize numeric columns as random forest does better with unstandardized data
        # # Standardize numeric columns
        # numeric_columns = df_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # if 'timestamp' in numeric_columns:
        #     numeric_columns.remove('timestamp')  # Don't standardize timestamp
            
        # scaler = StandardScaler()
        # df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
        
        return df_processed
    
    async def process_trading_pair(self, trading_pair: str) -> Optional[pd.DataFrame]:
        """Process a single trading pair to generate features and align with labels"""
        start_time = time.perf_counter()
        logger.info(f"Processing {trading_pair}")
        
        # Step 1: Load label data
        label_file = self.get_latest_label_file(trading_pair)
        if label_file is None:
            logger.error(f"No label file found for {trading_pair}, skipping")
            return None
        
        # Load the label file
        try:
            labels_df = pd.read_csv(label_file)
            # Convert timestamp to datetime index
            labels_df['t'] = pd.to_datetime(labels_df['t'])
            labels_df.set_index('t', inplace=True)
            logger.info(f"Loaded label data with shape {labels_df.shape}")
        except Exception as e:
            logger.error(f"Error loading labels for {trading_pair}: {e}")
            return None
        
        # Step 2: Smooth the labels
        try:
            smoothed_labels = self.smooth_labels(labels_df)
            logger.info(f"Smoothed labels, resulting shape: {smoothed_labels.shape}")
        except Exception as e:
            logger.error(f"Error smoothing labels for {trading_pair}: {e}")
            return None
        
        # Step 3: Determine time range for candle data
        label_start_time = smoothed_labels.index.min()
        label_end_time = smoothed_labels.index.max()
        
        # Add buffer for feature generation (some TA indicators need prior data)
        candle_start_time = label_start_time - pd.Timedelta(hours=6)  # 6 hours buffer for TA calculation
        candle_end_time = label_end_time + pd.Timedelta(minutes=5)  # Small buffer at the end
        
        # Step 4: Load candle data directly from timescale
        try:
            candle_interval = self.candle_intervals[0]  # Use the first interval (typically 1m)
            
            # Convert datetime to timestamp seconds
            start_timestamp = int(candle_start_time.timestamp())
            end_timestamp = int(candle_end_time.timestamp())
            
            logger.info(f"Loading candles for {trading_pair} from timescale, from {candle_start_time} to {candle_end_time}")
            
            # Connect to the database
            await self.timescale_client.connect()
            
            # Get candles directly from timescale client
            candles = await self.timescale_client.get_candles(
                connector_name=self.connector_name,
                trading_pair=trading_pair,
                interval=candle_interval,
                start_time=start_timestamp,
                end_time=end_timestamp
            )
            
            if candles.data.empty:
                logger.error(f"No candle data available in timescale for {trading_pair}, skipping")
                return None
                
            # Use the candles data directly
            candles_df = candles.data
            
            # Convert any numeric columns that might be strings
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                           'n_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']
            for col in numeric_cols:
                if col in candles_df.columns and candles_df[col].dtype == 'object':
                    candles_df[col] = pd.to_numeric(candles_df[col], errors='coerce')
                    
            # Ensure index is datetime if not already
            if not isinstance(candles_df.index, pd.DatetimeIndex):
                if 'timestamp' in candles_df.columns:
                    # Convert to datetime and set as index with name='timestamp'
                    candles_df.index = pd.to_datetime(candles_df['timestamp'], unit='s')
                    candles_df.index.name = 'timestamp'
                    # Drop the original timestamp column to avoid duplication
                    candles_df = candles_df.drop(columns=['timestamp'], errors='ignore')
                else:
                    candles_df.index = pd.to_datetime(candles_df.index, unit='s')
                    candles_df.index.name = 'timestamp'
            else:
                # Ensure the index has a name even if it's already a DatetimeIndex
                candles_df.index.name = 'timestamp'
            
            logger.info(f"Loaded candle data with shape {candles_df.shape}")
            
        except Exception as e:
            logger.error(f"Error loading candles from timescale for {trading_pair}: {e}")
            logger.error(traceback.format_exc())
            return None
        finally:
            # Close the database connection
            await self.timescale_client.close()
        
        # Step 5: Generate features
        try:
            features_df = self.generate_features(candles_df)
            logger.info(f"Generated features, resulting shape: {features_df.shape}")
        except Exception as e:
            logger.exception(f"Error generating features for {trading_pair}: {e}")
            return None
        
        # Step 6: Align and merge features with labels
        try:
            # Extract the parameter columns we want to use as labels
            label_columns = ['buy_0_bps', 'buy_1_step', 'sell_0_bps', 'sell_1_step', 
                           'take_profit_pct', 'stop_loss_pct']
            
            target_labels = smoothed_labels[label_columns].copy()
            
            # Get the timestamps from the label data to use as reference
            label_timestamps = target_labels.index.to_list()
            logger.info(f"Number of unique label timestamps: {len(label_timestamps)}")
            
            # Reset index for features
            features_reset = features_df.reset_index()
            
            # Convert features timestamp to datetime if it's numeric
            if pd.api.types.is_numeric_dtype(features_reset['timestamp']):
                features_reset['timestamp'] = pd.to_datetime(features_reset['timestamp'], unit='s')
                logger.info(f"Converted features timestamp from numeric to datetime")
            
            # Filter features to only include rows with timestamps that match label timestamps
            # This ensures we only have features for which we have labels
            features_reset['timestamp_key'] = features_reset['timestamp'].dt.floor('min')  # Floor to minute
            
            # Create a similar timestamp key for labels
            target_labels_reset = target_labels.reset_index()
            target_labels_reset['t_key'] = pd.to_datetime(target_labels_reset['t']).dt.floor('min')
            
            logger.info(f"Features timestamp range: {features_reset['timestamp_key'].min()} to {features_reset['timestamp_key'].max()}")
            logger.info(f"Labels timestamp range: {target_labels_reset['t_key'].min()} to {target_labels_reset['t_key'].max()}")
            
            # Make sure we're merging one-to-one with no duplicate timestamps that could cause averaging
            features_reset = features_reset.drop_duplicates('timestamp_key')
            target_labels_reset = target_labels_reset.drop_duplicates('t_key')
            
            # Inner join to only keep rows with exact timestamp matches
            merged_df = pd.merge(
                features_reset,
                target_labels_reset,
                left_on='timestamp_key',
                right_on='t_key',
                how='inner'
            )
            
            # Log merge result
            logger.info(f"Merged features and labels with exact matches, shape before dropna: {merged_df.shape}")
            
            # Drop temporary timestamp columns
            merged_df = merged_df.drop(columns=['timestamp_key', 't_key'])
            
            # Drop rows with NaN in either features or labels
            merged_df = merged_df.dropna(subset=label_columns)
            logger.info(f"Merged features and labels, final shape: {merged_df.shape}")
            
            # Set timestamp as index
            if 'timestamp' in merged_df.columns:
                merged_df.set_index('timestamp', inplace=True)
            
            # Drop the redundant 't' column from the merge
            if 't' in merged_df.columns:
                merged_df = merged_df.drop(columns=['t'])
                
            return merged_df
            
        except Exception as e:
            logger.error(f"Error aligning features and labels for {trading_pair}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    async def execute(self):
        """Execute the feature generation and label preparation task"""
        overall_start_time = time.perf_counter()
        logger.info(f"Starting PMMVMLFeatureGenerationTask at {datetime.datetime.now()}")
        logger.info(f"Processing {len(self.trading_pairs)} trading pairs")
        
        # Create output directory with full permissions
        try:
            # Ensure the directory exists with proper permissions
            os.makedirs(self.feature_output_path, exist_ok=True)
            # Make sure we have write permissions (0o777 = full permissions for all)
            os.chmod(self.feature_output_path, 0o777)
        except Exception as e:
            logger.error(f"Error creating output directory: {e}")
            logger.error(traceback.format_exc())
        
        # Process each trading pair
        for trading_pair in self.trading_pairs:
            try:
                pair_start_time = time.perf_counter()
                result_df = await self.process_trading_pair(trading_pair)
                
                if result_df is not None and not result_df.empty:
                    try:
                        # Save to output file
                        output_file = self.feature_output_path / f"{trading_pair.replace('-', '_')}_features_labels.csv"
                        result_df.to_csv(output_file)
                        logger.info(f"Saved features and labels for {trading_pair} to {output_file}")
                        
                        # Also save as parquet (more efficient)
                        parquet_file = self.feature_output_path / f"{trading_pair.replace('-', '_')}_features_labels.parquet"
                        result_df.to_parquet(
                            parquet_file,
                            engine='pyarrow',
                            compression='snappy',
                            index=True
                        )
                        logger.info(f"Saved features and labels for {trading_pair} to {parquet_file}")
                    except PermissionError as pe:
                        logger.error(f"Permission error saving files: {pe}")
                        logger.error(f"Trying to save to user home directory instead")
                        # Try saving to a different location as fallback
                        home_dir = os.path.expanduser("~")
                        alt_output_file = Path(home_dir) / f"{trading_pair.replace('-', '_')}_features_labels.csv"
                        result_df.to_csv(alt_output_file)
                        logger.info(f"Saved features and labels for {trading_pair} to {alt_output_file}")
                
                pair_duration = time.perf_counter() - pair_start_time
                logger.info(f"Completed processing {trading_pair} in {pair_duration:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error processing {trading_pair}: {e}")
                logger.error(traceback.format_exc())
        
        overall_duration = time.perf_counter() - overall_start_time
        logger.info(f"Completed PMMVMLFeatureGenerationTask in {overall_duration:.2f} seconds")


async def main():
    # Force is needed to override other logging configurations
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    # Run from command line with: python -m tasks.feature_generation.pmm_vml_feature_generation_task --config config/pmm_vml_feature_generation_task.yml
    config = BaseTask.load_single_task_config()
    task = PMMVMLFeatureGenerationTask("PMM VML Feature Generation", config)
    await task.run_once()

if __name__ == "__main__":
    debug = os.environ.get("ASYNC_DEBUG", False)
    asyncio.run(main(), debug=debug) 