import os
import random
import datetime
import yaml
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
FEE_PCT = 0.1  # 0.1% trading fee
OUTPUT_DIR = "config/xgridt"
TRADING_PAIRS = ["ADA-USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"]
N_TRIALS = 50  # Number of optimization trials per trading pair


def objective(trial, trading_pair):
    """
    Objective function for Optuna optimization.
    This simulates the performance of an XGridT strategy with the given parameters.
    """
    # Generate parameters to optimize
    ema_short = trial.suggest_int("ema_short", 3, 20)
    ema_medium = trial.suggest_int("ema_medium", ema_short + 5, 50)
    ema_long = trial.suggest_int("ema_long", ema_medium + 5, 100)
    donchian_channel_length = trial.suggest_int("donchian_channel_length", 20, 100, step=10)
    natr_length = trial.suggest_int("natr_length", 50, 150, step=10)
    natr_multiplier = trial.suggest_float("natr_multiplier", 0.8, 2.5, step=0.1)
    tp_default = trial.suggest_float("tp_default", 0.008, 0.04, step=0.004)
    total_amount_quote = trial.suggest_float("total_amount_quote", 100, 1000, step=100)
    max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 3)
    cooldown_time = trial.suggest_int("cooldown_time", 60, 900, step=60)
    prominence_pct_peaks = trial.suggest_float("prominence_pct_peaks", 0.005, 0.05, step=0.005)
    distance_between_peaks = trial.suggest_int("distance_between_peaks", 20, 150, step=10)
    
    # Set a random seed based on trading pair to get different but repeatable results per pair
    seed_value = sum(ord(c) for c in trading_pair) + trial.number
    random.seed(seed_value)
    
    # Generate simulated results based on parameters
    # We create a simple model of how these parameters might affect performance
    
    # Trade frequency factors:
    # - Lower ema_short → more trades → higher volume
    # - Higher tp_default → better per-trade profit but fewer trades
    # - Lower cooldown_time → more trades → higher volume
    # - Higher max_executors_per_side → more trades → higher volume
    trade_frequency_factor = (
        (20 - ema_short) / 17 + 
        (1 - tp_default / 0.04) + 
        (900 - cooldown_time) / 840 +
        max_executors_per_side / 2
    ) / 3
    
    # Make sure it's in a reasonable range
    trade_frequency_factor = max(0.5, min(3.0, trade_frequency_factor))
    
    # Different pairs have different baseline performance 
    # (this simulates market characteristics)
    pair_factor = {
        "ADA-USDT": 1.2,  # More volatile, higher volume potential
        "BTC-USDT": 0.9,  # Less volatile, harder to generate volume
        "ETH-USDT": 1.0,  # Base case
        "SOL-USDT": 1.3,  # More volatile, higher volume potential
        "XRP-USDT": 1.1,  # Moderate volatility
    }.get(trading_pair, 1.0)
    
    # Generate simulated metrics
    base_volume = 500 * pair_factor
    base_trades = 50 * pair_factor
    base_win_rate = 52
    
    # Add some randomness to simulate real-world variation
    randomness = 0.9 + 0.2 * random.random()
    
    # Calculate volume and trade count
    volume = base_volume * trade_frequency_factor * randomness
    num_trades = int(base_trades * trade_frequency_factor * randomness)
    
    # Profit is affected by:
    # - tp_default (take profit level)
    # - natr_multiplier (risk management)
    # - EMA settings (entry timing)
    profit_factor = (
        tp_default / 0.02 +  # Higher tp_default → higher profit per trade
        natr_multiplier / 2.0 * 0.5 +  # Better volatility management
        (ema_medium - ema_short) / 25 * 0.3  # Better trend confirmation
    ) / 1.8
    
    # Win rate is affected by several parameters
    win_rate = base_win_rate * (0.8 + profit_factor * 0.4) * randomness
    win_rate = min(65, max(45, win_rate))  # Keep in realistic range
    
    # Calculate profit
    avg_profit_per_win = tp_default * 100 * 0.8  # Convert to percentage, assume we don't always reach full TP
    avg_loss_per_loss = -avg_profit_per_win * 0.7  # Losses are smaller than wins due to risk management
    
    # Net profit calculation
    win_count = int(num_trades * win_rate / 100)
    loss_count = num_trades - win_count
    
    # Scale based on trading pair
    pair_profit_scale = {
        "ADA-USDT": 1.0,
        "BTC-USDT": 1.2,  # Higher spread, better profit potential
        "ETH-USDT": 1.1,
        "SOL-USDT": 0.9,  # Higher volatility, less predictable
        "XRP-USDT": 1.0,
    }.get(trading_pair, 1.0)
    
    # Calculate net profit and fees
    net_pnl = (win_count * avg_profit_per_win + loss_count * avg_loss_per_loss) * 0.1 * pair_profit_scale
    estimated_fees = (volume * FEE_PCT) / 100
    adjusted_pnl = net_pnl - estimated_fees
    
    # Calculate objective components for optimization
    volume_score = min(volume / 500, 3.0)  # Cap at 3x target
    profit_score = max(-1.0, min(adjusted_pnl / 10, 2.0))  # Normalize to [-1, 2] range
    win_rate_bonus = 0.5 * (win_rate / 100) if win_rate > 0 else 0
    trade_count_bonus = min(num_trades / 50, 0.5)  # Up to 0.5 for 50+ trades
    
    # Combined objective (higher is better)
    # We prioritize volume (weight 2.0) over profit (weight 1.0)
    objective_score = (2.0 * volume_score) + (1.0 * profit_score) + win_rate_bonus + trade_count_bonus
    
    # Store attributes for later analysis
    trial.set_user_attr("trading_pair", trading_pair)
    trial.set_user_attr("volume", volume)
    trial.set_user_attr("net_pnl", net_pnl)
    trial.set_user_attr("estimated_fees", estimated_fees)
    trial.set_user_attr("adjusted_pnl", adjusted_pnl)
    trial.set_user_attr("num_trades", num_trades)
    trial.set_user_attr("win_rate", win_rate)
    trial.set_user_attr("objective_score", objective_score)
    
    # Log trial results
    logger.info(f"Trial {trial.number} ({trading_pair}): Vol={volume:.2f}, Trades={num_trades}, "
                f"Win%={win_rate:.1f}, Net={net_pnl:.2f}, Fees={estimated_fees:.2f}, "
                f"Adj.PnL={adjusted_pnl:.2f}, Score={objective_score:.4f}")
    
    return objective_score


def create_config_from_params(connector_name, trading_pair, params):
    """Create a configuration file from optimized parameters."""
    config = {
        "id": f"xgridt_spot_{trading_pair.replace('-', '_')}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "controller_name": "xgridt",
        "controller_type": "generic",
        "total_amount_quote": params.get("total_amount_quote", 100),
        "manual_kill_switch": None,
        "candles_config": [],
        "connector_name": connector_name,
        "trading_pair": trading_pair,
        "candles_connector": connector_name,
        "candles_trading_pair": trading_pair,
        "interval": "1m",
        "ema_short": params.get("ema_short", 8),
        "ema_medium": params.get("ema_medium", 29),
        "ema_long": params.get("ema_long", 31),
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
        "donchian_channel_length": params.get("donchian_channel_length", 50),
        "natr_length": params.get("natr_length", 100),
        "natr_multiplier": params.get("natr_multiplier", 2.0),
        "tp_default": params.get("tp_default", 0.05),
        "prominence_pct_peaks": params.get("prominence_pct_peaks", 0.05),
        "distance_between_peaks": params.get("distance_between_peaks", 100),
        "cooldown_time": params.get("cooldown_time", 900),
        "max_executors_per_side": params.get("max_executors_per_side", 1)
    }
    return config


def main():
    """Main function to run the XGridT configuration generator."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Prepare summary dataframe
    summary_cols = ["trading_pair", "ema_short", "ema_medium", "ema_long", "donchian_channel_length",
                    "natr_length", "natr_multiplier", "tp_default", "cooldown_time", "prominence_pct_peaks",
                    "distance_between_peaks", "total_amount_quote", "max_executors_per_side",
                    "volume", "net_pnl", "adjusted_pnl", "num_trades", "win_rate", "objective_score"]
    summary_df = pd.DataFrame(columns=summary_cols)
    
    # Process each trading pair
    for trading_pair in TRADING_PAIRS:
        logger.info(f"Optimizing strategy for okx {trading_pair}...")
        
        # Create a study for this trading pair
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            study_name=f"xgridt_spot_{trading_pair.replace('-', '_')}"
        )
        
        # Run the optimization
        study.optimize(lambda trial: objective(trial, trading_pair), n_trials=N_TRIALS)
        
        # Get best parameters and results
        best_params = study.best_params
        best_trial = study.best_trial
        best_attrs = best_trial.user_attrs
        
        # Log best results
        logger.info(f"\nBest parameters for {trading_pair}:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
            
        logger.info(f"Performance metrics:")
        logger.info(f"  Volume: {best_attrs['volume']:.2f}")
        logger.info(f"  Net PnL: {best_attrs['net_pnl']:.2f}")
        logger.info(f"  Adjusted PnL (after fees): {best_attrs['adjusted_pnl']:.2f}")
        logger.info(f"  Number of trades: {best_attrs['num_trades']}")
        logger.info(f"  Win rate: {best_attrs['win_rate']:.2f}%")
        logger.info(f"  Objective score: {best_attrs['objective_score']:.4f}")
        
        # Add to summary dataframe
        row_data = {
            "trading_pair": trading_pair,
            **best_params,
            "volume": best_attrs["volume"],
            "net_pnl": best_attrs["net_pnl"],
            "adjusted_pnl": best_attrs["adjusted_pnl"],
            "num_trades": best_attrs["num_trades"],
            "win_rate": best_attrs["win_rate"],
            "objective_score": best_attrs["objective_score"]
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Create and save configuration file if profitable
        if best_attrs["adjusted_pnl"] >= 0:
            config = create_config_from_params("okx", trading_pair, best_params)
            output_file = os.path.join(OUTPUT_DIR, f"xgridt_spot_{trading_pair.replace('-', '_')}.yaml")
            
            with open(output_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            logger.info(f"Saved configuration to {output_file}")
        else:
            logger.warning(f"Not generating config for {trading_pair} as adjusted profit is negative: {best_attrs['adjusted_pnl']:.2f}")
    
    # Save summary dataframe
    summary_file = os.path.join(OUTPUT_DIR, f"xgridt_optimization_summary_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved optimization summary to {summary_file}")
    
    # Print final summary
    profitable_pairs = summary_df[summary_df["adjusted_pnl"] >= 0]
    logger.info(f"\nSummary of profitable configurations ({len(profitable_pairs)}/{len(TRADING_PAIRS)} pairs):")
    for _, row in profitable_pairs.iterrows():
        logger.info(f"  {row['trading_pair']}: Volume={row['volume']:.2f}, Adjusted PnL={row['adjusted_pnl']:.2f}")


if __name__ == "__main__":
    main() 