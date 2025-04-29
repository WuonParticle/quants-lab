import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Set, Union

import yaml
from dotenv import load_dotenv

from core.task_base import BaseTask
from tasks.deployment.deployment_base_task import DeploymentBaseTask
from tasks.deployment.models import ConfigCandidate

load_dotenv()

class PMMVMLDeploymentTask(DeploymentBaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name=name, frequency=frequency, config=config)
        self.trading_pairs_to_deploy = config.get("trading_pairs_to_deploy", [])
        self.base_controller_config = config.get("base_controller_config", {})

    async def _fetch_controller_configs(self) -> List[ConfigCandidate]:
        """
        Programmatically  ConfigCandidate objects based on the trading_pairs_to_deploy list
        and the base_controller_config template from YAML config.
        """
        generated_candidates = []
        timestamp = int(time.time())
        
        for trading_pair in self.trading_pairs_to_deploy:
            try:
                # Create a copy of the base controller config
                config_copy = self.base_controller_config.copy()
                
                # Set the trading pair specific fields
                config_copy["trading_pair"] = trading_pair
                config_copy["connector_name"] = self.connector_name
                
                # Generate a unique ID - using lowercase and no underscores to match the expected format
                normalized_pair = trading_pair.replace("-", "").lower()
                config_id = f"pmm_vml_{normalized_pair}"
                config_copy["id"] = config_id
                
                # Set controller_name to pmm_vml
                config_copy["controller_name"] = "pmm_vml"
                
                # Adjust MQTT topic if needed
                if "mqtt_topic_prefix" in config_copy:
                    config_copy["mqtt_topic_prefix"] = f"{config_copy.get('mqtt_topic_prefix', 'hbot/pmm_vml_params')}"
                
                # Create a ConfigCandidate object
                extra_info = {"generated_timestamp": timestamp, "trading_pair": trading_pair}
                candidate = ConfigCandidate(config=config_copy, extra_info=extra_info, id=config_id)
                
                generated_candidates.append(candidate)
                logging.info(f"Generated config candidate for {trading_pair} with ID: {config_id}")
            except Exception as e:
                logging.exception(f"Error generating config for {trading_pair}: {str(e)}")
        
        return generated_candidates

    def _extract_trading_pairs(self, config_candidates: List[ConfigCandidate]) -> Set[str]:
        """
        Extracts the set of trading pairs from the config candidates.
        """
        trading_pairs = set()
        for candidate in config_candidates:
            trading_pair = candidate.config.get("trading_pair")
            if trading_pair:
                trading_pairs.add(trading_pair)
        return trading_pairs

    def _filter_configs_by_trading_pair(self, all_config_candidates: List[ConfigCandidate],
                                        trading_pairs: List[str]) -> List[ConfigCandidate]:
        """
        Filters the config candidates by trading pair.
        """
        filtered_candidates = []
        for candidate in all_config_candidates:
            trading_pair = candidate.config.get("trading_pair")
            if trading_pair in trading_pairs:
                filtered_candidates.append(candidate)
            else:
                logging.warning(f"Trading pair {trading_pair} from candidate {candidate.id} "
                               f"is not available in the exchange.")
        return filtered_candidates

    async def _is_candidate_valid(self, candidate: ConfigCandidate, filter_candidate_params: Dict[str, Any]) -> bool:
        """
        Determines if a candidate is valid based on trading rules and last prices.
        """
        trading_pair = candidate.config.get("trading_pair")
        
        # Check if trading pair exists in trading rules
        trading_pair_in_rules = any(rule.trading_pair == trading_pair for rule in self.trading_rules.data)
        if not trading_pair_in_rules:
            logging.warning(f"Trading pair {trading_pair} not found in trading rules")
            return False
        
        # Check if trading pair exists in last prices
        if trading_pair not in self.last_prices:
            logging.warning(f"Trading pair {trading_pair} not found in last prices")
            return False
        
        return True

    def _adjust_config_candidates(self, config_candidates: List[ConfigCandidate]):
        """
        Adjusts configuration parameters for each candidate.
        """
        adjusted_candidates = []
        for candidate in config_candidates:
            config = candidate.config
            trading_pair = config.get("trading_pair")
            
            # Ensure essential fields are correctly set
            config["controller_type"] = "market_making"
            
            # Ensure other required parameters are set
            if "total_amount_quote" not in config:
                config["total_amount_quote"] = float(self.config.get("base_controller_config", {}).get("total_amount_quote", 1000.0))
            
            if "leverage" not in config:
                config["leverage"] = self.config.get("base_controller_config", {}).get("leverage", 20)
            
            # Convert any Decimal objects to float to ensure JSON serialization works
            candidate.config = config
            
            logging.info(f"Adjusted config for {trading_pair} - ID: {candidate.id}")
            adjusted_candidates.append(candidate)
        
        return adjusted_candidates

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load configuration from YAML file in the config directory
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "pmm_vml_deployment_task.yml")
    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    
    # Extract task config from the new format
    config = yaml_data.get("tasks", {}).get("pmm_vml_deployment", {}).get("config", {})
    
    # Extract task frequency
    frequency_seconds = config.get("frequency_seconds", 600)
    
    # Create and run the task
    task = PMMVMLDeploymentTask(
        name="pmm_vml_deployment",
        frequency=timedelta(seconds=frequency_seconds),
        config=config
    )
    
    logging.info("Starting PMM VML deployment task")
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main()) 