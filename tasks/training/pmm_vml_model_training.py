import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMMVMLModelTrainingTask(BaseTask):
    """
    Task that trains machine learning models to predict the 6 optimal parameters
    for the pmm_simple market making strategy based on previously generated features.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        frequency = None  # We'll use run_once for this task
        super().__init__(name, frequency, config)
        
        # Configuration parameters
        self.feature_label_data_path = Path(config.get("feature_label_data_path", "data/features_labels/"))
        self.model_output_path = Path(config.get("model_output_path", "models/pmm_vml/"))
        self.trading_pairs = config.get("trading_pairs", [])
        self.test_size = config.get("test_size", 0.2)
        self.random_state = config.get("random_state", 42)
        self.model_params = config.get("model_params", {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        })
        
        # Create output directory if it doesn't exist
        os.makedirs(self.model_output_path, exist_ok=True)
    
    async def train_model_for_pair(self, trading_pair: str) -> bool:
        """
        Train a model for a single trading pair
        
        Args:
            trading_pair: The trading pair to train the model for
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        start_time = time.perf_counter()
        logger.info(f"Training model for {trading_pair}")
        
        # Normalized pair name for file paths
        normalized_pair = trading_pair.replace('-', '_')
        
        # Step 1: Load data
        feature_label_path = self.feature_label_data_path / f"{normalized_pair}_features_labels.csv"
        try:
            data = pd.read_csv(feature_label_path, index_col=0)
            logger.info(f"Loaded data with shape {data.shape}")
        except Exception as e:
            logger.error(f"Error loading data for {trading_pair}: {e}")
            return False
        
        # Step 2: Identify features and target columns
        target_columns = [
            'buy_0_bps', 'buy_1_step', 'sell_0_bps', 
            'sell_1_step', 'take_profit_pct', 'stop_loss_pct'
        ]
        
        # Check if we have the target columns
        missing_targets = [col for col in target_columns if col not in data.columns]
        if missing_targets:
            logger.error(f"Missing target columns for {trading_pair}: {missing_targets}")
            return False
        
        # All columns except target columns are features
        feature_columns = [col for col in data.columns if col not in target_columns]
        
        # Check if we have enough data
        if data.shape[0] < 100:
            logger.warning(f"Not enough data for {trading_pair} (only {data.shape[0]} rows), skipping")
            return False
        
        # Step 3: Handle missing or infinite values
        X = data[feature_columns].copy()
        y = data[target_columns].copy()
        
        # Identify non-numeric columns and datetime columns
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        
        if non_numeric_cols:
            logger.info(f"Found non-numeric columns: {non_numeric_cols}")
            
            # Convert any datetime-like columns to timestamp (seconds since epoch)
            datetime_cols = []
            for col in non_numeric_cols:
                try:
                    # Try to convert to datetime
                    X[f"{col}_ts"] = pd.to_datetime(X[col]).astype(np.int64) // 10**9
                    datetime_cols.append(col)
                except Exception as e:
                    logger.warning(f"Column {col} couldn't be converted to datetime: {e}")
            
            # Drop original datetime columns after converting to timestamp
            if datetime_cols:
                logger.info(f"Converted datetime columns to timestamps: {datetime_cols}")
                X = X.drop(columns=datetime_cols)
                
            # Drop any remaining non-numeric columns that couldn't be converted
            remaining_non_numeric = X.select_dtypes(exclude=['number']).columns.tolist()
            if remaining_non_numeric:
                logger.warning(f"Dropping remaining non-numeric columns: {remaining_non_numeric}")
                X = X.drop(columns=remaining_non_numeric)
        
        # Replace any infinite values with NaN
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Log the percentage of missing values
        missing_pct = X.isna().mean().round(4) * 100
        columns_with_missing = missing_pct[missing_pct > 0].index.tolist()
        if columns_with_missing:
            logger.info(f"Columns with missing values: {len(columns_with_missing)}")
            logger.info(f"Top 5 columns with highest missing %: {missing_pct.sort_values(ascending=False).head(5)}")
        
        # Use SimpleImputer to fill NaN values with the median of each column
        logger.info(f"Imputing missing values with median strategy")
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Update feature columns to match the actual columns we're using
        feature_columns = X.columns.tolist()
        
        # Step 4: Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Step 5: Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Step 6: Initialize and train the model
        base_model = RandomForestRegressor(**self.model_params, random_state=self.random_state)
        model = MultiOutputRegressor(base_model)
        
        try:
            logger.info(f"Training model for {trading_pair}...")
            model.fit(X_train_scaled, y_train)
            logger.info(f"Model training completed for {trading_pair}")
        except Exception as e:
            logger.error(f"Error training model for {trading_pair}: {e}")
            return False
        
        # Step 7: Evaluate the model
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics for each target variable
        logger.info(f"Model evaluation for {trading_pair}:")
        for i, target in enumerate(target_columns):
            mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            logger.info(f"  {target}: MSE={mse:.4f}, R²={r2:.4f}")
        
        # Step 8: Save the model and scaler
        model_file = self.model_output_path / f"{normalized_pair}_model.joblib"
        scaler_file = self.model_output_path / f"{normalized_pair}_scaler.joblib"
        imputer_file = self.model_output_path / f"{normalized_pair}_imputer.joblib"
        
        try:
            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)
            joblib.dump(imputer, imputer_file)
            logger.info(f"Model and preprocessing objects saved for {trading_pair}")
            
            # Also save feature names for future reference
            feature_file = self.model_output_path / f"{normalized_pair}_features.txt"
            with open(feature_file, 'w') as f:
                f.write('\n'.join(feature_columns))
            logger.info(f"Feature list saved for {trading_pair}")
        except Exception as e:
            logger.error(f"Error saving model for {trading_pair}: {e}")
            return False
        
        duration = time.perf_counter() - start_time
        logger.info(f"Completed model training for {trading_pair} in {duration:.2f} seconds")
        return True
    
    async def execute(self):
        """Execute the model training task for all trading pairs"""
        overall_start_time = time.perf_counter()
        logger.info(f"Starting PMMVMLModelTrainingTask for {len(self.trading_pairs)} trading pairs")
        
        # Train models for each trading pair
        success_count = 0
        for trading_pair in self.trading_pairs:
            try:
                success = await self.train_model_for_pair(trading_pair)
                if success:
                    success_count += 1
            except Exception as e:
                logger.exception(f"Unexpected error processing {trading_pair}: {e}")
        
        overall_duration = time.perf_counter() - overall_start_time
        logger.info(f"Completed PMMVMLModelTrainingTask in {overall_duration:.2f} seconds")
        logger.info(f"Successfully trained models for {success_count}/{len(self.trading_pairs)} trading pairs")


async def main():
    # Force is needed to override other logging configurations
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    # Run from command line with: python -m tasks.training.pmm_vml_model_training --config config/pmm_vml_model_training.yml
    config = BaseTask.load_single_task_config()
    task = PMMVMLModelTrainingTask("PMM VML Model Training", config)
    await task.run_once()

if __name__ == "__main__":
    debug = os.environ.get("ASYNC_DEBUG", False)
    asyncio.run(main(), debug=debug) 