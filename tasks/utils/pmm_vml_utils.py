import logging
import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401

logger = logging.getLogger(__name__)


def generate_features(candles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technical analysis features for the given candle data.
    This function is used by both the feature generation task and prediction service
    to ensure consistent feature engineering between training and inference.
    
    Args:
        candles_df: DataFrame containing candle data
        
    Returns:
        DataFrame with features added
    """
    logger.debug(f"Generating features for dataframe with shape {candles_df.shape}")
    
    # Create a copy to work with
    df_with_indicators = candles_df.copy()
    
    # Make sure required columns exist for TA indicators
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in df_with_indicators.columns:
            logger.error(f"Required column '{col}' not found in candles data")
            raise ValueError(f"Required column '{col}' not found in candles data")
    
    # Bollinger Bands with different lengths
    df_with_indicators.ta.bbands(length=20, std=2.0, append=True)
    df_with_indicators.ta.bbands(length=50, std=2.0, append=True)
    
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
    if all(col in df_processed.columns for col in ['taker_buy_quote_volume', 'quote_asset_volume']):
        # Create buy/sell volume ratio
        df_processed['buy_volume_ratio'] = df_processed['taker_buy_quote_volume'] / df_processed['quote_asset_volume']
        df_processed = df_processed.drop(columns=['taker_buy_quote_volume'])
    elif 'volume' in df_processed.columns:
        # If we don't have taker_buy_quote_volume but have volume, use placeholders
        logger.warning("taker_buy_quote_volume column not found, using placeholders for volume features")
        df_processed['volume_change'] = df_processed['volume'].pct_change()
        df_processed['volume_sma_10'] = df_processed['volume'].rolling(window=10).mean() / df_processed['volume']
        df_processed['rel_volume'] = df_processed['volume'] / df_processed['volume'].rolling(window=20).mean()
        df_processed['buy_volume_ratio'] = 0.5  # Placeholder
    else:
        logger.warning("Volume columns not found, using placeholders for volume features")
        df_processed['buy_volume_ratio'] = 0.5  # Placeholder
    
    # Additional unnecessary columns to drop if they exist
    columns_to_drop = [
        'taker_buy_base_volume', 'volume', 'close_time', 
        'taker_buy_volume', 'number_of_trades', 'buy_volume',
        'sell_volume', 'unused', 'quote_asset_volume'
    ]
    # Only drop columns that actually exist
    cols_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
    if cols_to_drop:
        df_processed = df_processed.drop(columns=cols_to_drop)
    
    # Replace any infinite values with NaN
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df_processed


def preprocess_features(features_df: pd.DataFrame, imputer=None, scaler=None) -> pd.DataFrame:
    """
    Handle missing values and scale features consistently between training and inference
    
    Args:
        features_df: DataFrame with features
        imputer: Optional SimpleImputer to use for handling missing values
        scaler: Optional StandardScaler to use for scaling features
        
    Returns:
        Preprocessed features DataFrame
    """
    # Handle NaN values
    if features_df.isna().any().any():
        if imputer is not None:
            # Use provided imputer
            features_imputed = pd.DataFrame(
                imputer.transform(features_df),
                columns=features_df.columns,
                index=features_df.index
            )
        else:
            # Default to dropping NaN rows
            features_imputed = features_df.dropna()
    else:
        features_imputed = features_df
    
    # Apply scaling if scaler is provided
    if scaler is not None and not features_imputed.empty:
        features_scaled = pd.DataFrame(
            scaler.transform(features_imputed),
            columns=features_imputed.columns,
            index=features_imputed.index
        )
        return features_scaled
    
    return features_imputed 