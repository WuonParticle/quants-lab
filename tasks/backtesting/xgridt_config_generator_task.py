import asyncio
import datetime
import logging
import os
import time
import yaml
from datetime import timedelta
from typing import Any, Dict, List
from decimal import Decimal

import pandas as pd
import optuna
from dotenv import load_dotenv
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop

from controllers.directional_trading.xgridt import XGridTControllerConfig
from core.backtesting.optimizer import StrategyOptimizer, BacktestingConfig, BaseStrategyConfigGenerator
from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class XGridTSpotConfigGenerator(BaseStrategyConfigGenerator):
    """
    Strategy configuration generator for XGridT optimization for spot markets.
    """
    async def generate_config(self, trial) -> BacktestingConfig:
        # Controller configuration
        connector_name = self.config.get("connector_name", "okx")
        trading_pair = self.config.get("trading_pair", "ADA-USDT")
        interval = self.config.get("interval", "1m")
        fee_pct = self.config.get("fee_pct", 0.1)  # Default 0.1% fee
        trial.set_user_attr("connector_name", connector_name)
        trial.set_user_attr("trading_pair", trading_pair)
        trial.set_user_attr("interval", interval)
        trial.set_user_attr("fee_pct", fee_pct)
        
        # EMA parameters - optimize for shorter periods to increase trading activity
        ema_short = trial.suggest_int("ema_short", 3, 20)
        ema_medium = trial.suggest_int("ema_medium", ema_short + 5, 50)
        ema_long = trial.suggest_int("ema_long", ema_medium + 5, 100)
        
        # Donchian channel parameters - affects trend detection
        donchian_channel_length = trial.suggest_int("donchian_channel_length", 20, 100, step=10)
        
        # NATR parameters - affects order placement 
        natr_length = trial.suggest_int("natr_length", 50, 150, step=10)
        natr_multiplier = trial.suggest_float("natr_multiplier", 0.8, 2.5, step=0.1)
        
        # Take profit parameters - important for profitability
        # Higher values could lead to fewer completed trades but better per-trade profit
        # Lower values could lead to more completed trades, higher volume
        tp_default = trial.suggest_float("tp_default", 0.008, 0.04, step=0.004)  # Narrowed range for better optimization
        
        # Grid parameters - directly affects trading volume
        total_amount_quote = trial.suggest_float("total_amount_quote", 100, 1000, step=100)
        max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 3)
        
        # Time parameters - adjust for higher trading frequency
        time_limit = 60 * 60 * 24  # 24 hours
        cooldown_time = trial.suggest_int("cooldown_time", 60, 900, step=60)  # 1-15 minutes, lower for more frequent trading
        
        # Peak detection parameters - affects order placement
        prominence_pct_peaks = trial.suggest_float("prominence_pct_peaks", 0.005, 0.05, step=0.005)
        distance_between_peaks = trial.suggest_int("distance_between_peaks", 20, 150, step=10)
        
        # Create the strategy configuration
        config = XGridTControllerConfig(
            connector_name=connector_name,
            trading_pair=trading_pair,
            interval=interval,
            total_amount_quote=Decimal(total_amount_quote),
            time_limit=time_limit,
            max_executors_per_side=max_executors_per_side,
            cooldown_time=cooldown_time,
            ema_short=ema_short,
            ema_medium=ema_medium,
            ema_long=ema_long,
            donchian_channel_length=donchian_channel_length,
            natr_length=natr_length,
            natr_multiplier=natr_multiplier,
            tp_default=tp_default,
            prominence_pct_peaks=prominence_pct_peaks,
            distance_between_peaks=distance_between_peaks,
            position_mode="ONE_WAY",  # Spot markets use ONE_WAY position mode
            leverage=1,  # Spot markets use leverage of 1
            close_position_on_signal_change=True,
            take_profit_mode="original",
            take_profit_step_multiplier=1,
            global_stop_loss=0.1,
            executor_activation_bounds=0.001,
            general_activation_bounds=0.001,
            max_ranges_by_signal=1,
            min_spread_between_orders=0.001,
            min_order_amount=1,
            max_open_orders=5,
            extra_balance_base_usd=10
        )

        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)


class VolumeWithBreakevenObjective:
    """
    Custom objective function class that prioritizes high volume while ensuring profit is at least break-even after fees.
    """
    def __init__(self, fee_pct=0.1, target_min_volume=500, profit_weight=1.0, volume_weight=2.0):
        self.fee_pct = fee_pct
        self.target_min_volume = target_min_volume
        self.profit_weight = profit_weight
        self.volume_weight = volume_weight
        
    def __call__(self, trial, results):
        # Extract metrics from backtesting results
        net_pnl = results.get("net_pnl", 0)
        volume = results.get("volume", 0)
        num_trades = results.get("num_trades", 0)
        win_rate = results.get("win_rate", 0)
        
        # Calculate estimated fees based on volume
        estimated_fees = (volume * self.fee_pct) / 100
        
        # Adjusted profit after fees
        adjusted_pnl = net_pnl - estimated_fees
        
        # Objectives:
        # 1. Maximize volume (main objective)
        # 2. Ensure profit is at least break-even after fees
        # 3. Encourage higher win rate for strategy stability
        
        # Volume score (higher is better)
        volume_score = min(volume / self.target_min_volume, 3.0)  # Cap at 3x target
        
        # Profit score (higher is better, negative if below break-even)
        profit_score = max(-1.0, min(adjusted_pnl / 10, 2.0))  # Normalize to [-1, 2] range
        
        # Win rate bonus (0 to 0.5)
        win_rate_bonus = 0.5 * (win_rate / 100) if win_rate > 0 else 0
        
        # Trade count bonus (to encourage activity)
        trade_count_bonus = min(num_trades / 50, 0.5)  # Up to 0.5 for 50+ trades
        
        # Combined objective (higher is better)
        objective = (self.volume_weight * volume_score) + (self.profit_weight * profit_score) + win_rate_bonus + trade_count_bonus
        
        # Log the metrics for visibility
        logger.info(f"Trial {trial.number}: Volume={volume:.2f}, Net PnL={net_pnl:.2f}, "
                    f"Est. Fees={estimated_fees:.2f}, Adjusted PnL={adjusted_pnl:.2f}, "
                    f"Objective={objective:.4f}")
        
        # Store important metrics for later analysis
        trial.set_user_attr("volume", volume)
        trial.set_user_attr("net_pnl", net_pnl)
        trial.set_user_attr("estimated_fees", estimated_fees)
        trial.set_user_attr("adjusted_pnl", adjusted_pnl)
        trial.set_user_attr("objective_score", objective)
        
        # Return objective value (maximizing)
        return objective


class XGridTConfigGeneratorTask(BaseTask):
    """
    Task for generating optimized XGridT configurations for multiple token pairs.
    """
    async def execute(self):
        optimizer = StrategyOptimizer(root_path=self.config["root_path"])
        
        # Get configuration parameters
        selected_pairs = self.config.get("selected_pairs", ["ADA-USDT"])
        connector_name = self.config.get("connector_name", "okx")
        output_dir = self.config.get("output_dir", "configs")
        lookback_days = self.config.get("lookback_days", 30)
        n_trials = self.config.get("n_trials", 100)
        fee_pct = self.config.get("fee_pct", 0.1)
        target_min_volume = self.config.get("target_min_volume", 500)
        profit_weight = self.config.get("profit_weight", 1.0)
        volume_weight = self.config.get("volume_weight", 2.0)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create custom objective function
        objective_function = VolumeWithBreakevenObjective(
            fee_pct=fee_pct,
            target_min_volume=target_min_volume,
            profit_weight=profit_weight,
            volume_weight=volume_weight
        )
        
        # Create a summary file for all results
        summary_file = os.path.join(output_dir, f"xgridt_optimization_summary_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
        summary_cols = ["trading_pair", "ema_short", "ema_medium", "ema_long", "donchian_channel_length", 
                        "natr_length", "natr_multiplier", "tp_default", "cooldown_time", "prominence_pct_peaks",
                        "distance_between_peaks", "total_amount_quote", "max_executors_per_side",
                        "volume", "net_pnl", "adjusted_pnl", "num_trades", "win_rate", "objective_score"]
        
        summary_df = pd.DataFrame(columns=summary_cols)
        
        # Process each trading pair
        for trading_pair in selected_pairs:
            logger.info(f"Optimizing strategy for {connector_name} {trading_pair}")
            
            # Calculate date range for backtesting
            end_date = time.time() - self.config.get("end_time_buffer_hours", 6) * 3600
            start_date = end_date - lookback_days * 24 * 60 * 60
            
            # Create config generator
            config_generator = XGridTSpotConfigGenerator(
                start_date=pd.to_datetime(start_date, unit="s"), 
                end_date=pd.to_datetime(end_date, unit="s"),
                config={
                    "connector_name": connector_name, 
                    "trading_pair": trading_pair,
                    "fee_pct": fee_pct
                }
            )
            
            # Load candles cache
            logger.info(f"Fetching candles for {connector_name} {trading_pair}")
            optimizer.load_candles_cache_by_connector_pair(connector_name=connector_name, trading_pair=trading_pair)
            
            # Run optimization
            study_name = f"xgridt_spot_{trading_pair.replace('-', '_')}_{datetime.datetime.now().strftime('%Y%m%d')}"
            try:
                # Set custom objective function
                optimizer.set_custom_objective(objective_function)
                
                # Run optimization with pruning to speed up process
                pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
                await optimizer.optimize(
                    study_name=study_name,
                    config_generator=config_generator, 
                    n_trials=n_trials,
                    pruner=pruner
                )
                
                # Get best parameters
                best_params = optimizer.get_study_best_params(study_name)
                best_trial = optimizer.get_study_best_trial(study_name)
                
                if not best_params or not best_trial:
                    logger.warning(f"No best parameters found for {trading_pair}")
                    continue
                
                # Log best results
                logger.info(f"Best parameters for {trading_pair}:")
                for key, value in best_params.items():
                    logger.info(f"  {key}: {value}")
                
                logger.info(f"Performance metrics:")
                volume = best_trial.user_attrs.get("volume", 0)
                net_pnl = best_trial.user_attrs.get("net_pnl", 0)
                adjusted_pnl = best_trial.user_attrs.get("adjusted_pnl", 0)
                objective_score = best_trial.user_attrs.get("objective_score", 0)
                num_trades = best_trial.user_attrs.get("num_trades", 0)
                win_rate = best_trial.user_attrs.get("win_rate", 0)
                
                logger.info(f"  Volume: {volume:.2f}")
                logger.info(f"  Net PnL: {net_pnl:.2f}")
                logger.info(f"  Adjusted PnL (after fees): {adjusted_pnl:.2f}")
                logger.info(f"  Number of trades: {num_trades}")
                logger.info(f"  Win rate: {win_rate:.2f}%")
                logger.info(f"  Objective score: {objective_score:.4f}")
                
                # Add to summary dataframe
                row_data = {
                    "trading_pair": trading_pair,
                    **best_params,
                    "volume": volume,
                    "net_pnl": net_pnl,
                    "adjusted_pnl": adjusted_pnl,
                    "num_trades": num_trades,
                    "win_rate": win_rate,
                    "objective_score": objective_score
                }
                summary_df = pd.concat([summary_df, pd.DataFrame([row_data])], ignore_index=True)
                
                # Only create config files for profitable strategies
                if adjusted_pnl >= 0:
                    # Create configuration file
                    config = self._create_config_from_params(
                        connector_name=connector_name,
                        trading_pair=trading_pair,
                        best_params=best_params
                    )
                    
                    # Save configuration to file
                    output_file = os.path.join(output_dir, f"xgridt_spot_{trading_pair.replace('-', '_')}.yaml")
                    with open(output_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    
                    logger.info(f"Saved configuration to {output_file}")
                else:
                    logger.warning(f"Not generating config for {trading_pair} as adjusted profit is negative: {adjusted_pnl:.2f}")
            
            except Exception as e:
                logger.error(f"Error optimizing {trading_pair}: {str(e)}")
                continue
        
        # Save summary dataframe
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved optimization summary to {summary_file}")
        
        # Print final summary of profitable configurations
        profitable_pairs = summary_df[summary_df["adjusted_pnl"] >= 0]
        logger.info(f"\nSummary of profitable configurations ({len(profitable_pairs)}/{len(selected_pairs)} pairs):")
        for _, row in profitable_pairs.iterrows():
            logger.info(f"  {row['trading_pair']}: Volume={row['volume']:.2f}, Adjusted PnL={row['adjusted_pnl']:.2f}")
    
    def _create_config_from_params(self, connector_name: str, trading_pair: str, best_params: Dict) -> Dict:
        """
        Create a configuration dictionary from the best parameters.
        """
        # Extract parameters
        ema_short = best_params.get("ema_short", 8)
        ema_medium = best_params.get("ema_medium", 29)
        ema_long = best_params.get("ema_long", 31)
        donchian_channel_length = best_params.get("donchian_channel_length", 50)
        natr_length = best_params.get("natr_length", 100)
        natr_multiplier = best_params.get("natr_multiplier", 2.0)
        tp_default = best_params.get("tp_default", 0.05)
        total_amount_quote = best_params.get("total_amount_quote", 100)
        max_executors_per_side = best_params.get("max_executors_per_side", 1)
        cooldown_time = best_params.get("cooldown_time", 900)
        prominence_pct_peaks = best_params.get("prominence_pct_peaks", 0.05)
        distance_between_peaks = best_params.get("distance_between_peaks", 100)
        
        # Create configuration dictionary
        config = {
            "id": f"xgridt_spot_{trading_pair.replace('-', '_')}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "controller_name": "xgridt",
            "controller_type": "generic",
            "total_amount_quote": total_amount_quote,
            "manual_kill_switch": None,
            "candles_config": [],
            "connector_name": connector_name,
            "trading_pair": trading_pair,
            "candles_connector": connector_name,
            "candles_trading_pair": trading_pair,
            "interval": "1m",
            "ema_short": ema_short,
            "ema_medium": ema_medium,
            "ema_long": ema_long,
            "position_mode": "ONE_WAY",
            "leverage": 1,
            "close_position_on_signal_change": True,
            "grid_update_interval": None,
            "take_profit_mode": "original",
            "take_profit_step_multiplier": 1,
            "global_stop_loss": 0.1,
            "time_limit": 7200,
            "executor_activation_bounds": 0.001,
            "general_activation_bounds": 0.001,
            "max_ranges_by_signal": 1,
            "min_spread_between_orders": 0.001,
            "min_order_amount": 1,
            "max_open_orders": 5,
            "order_frequency": 0,
            "extra_balance_base_usd": 10,
            "donchian_channel_length": donchian_channel_length,
            "natr_length": natr_length,
            "natr_multiplier": natr_multiplier,
            "tp_default": tp_default,
            "prominence_pct_peaks": prominence_pct_peaks,
            "distance_between_peaks": distance_between_peaks,
            "cooldown_time": cooldown_time
        }
        
        return config


async def main():
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
        "connector_name": "okx",
        "lookback_days": 30,
        "end_time_buffer_hours": 6,
        "n_trials": 100,
        "output_dir": "configs",
        "selected_pairs": ["ADA-USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"],
        "fee_pct": 0.1,
        "target_min_volume": 500,
        "profit_weight": 1.0,
        "volume_weight": 2.0
    }

    task = XGridTConfigGeneratorTask("XGridTConfigGenerator", timedelta(hours=24), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main()) 