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


def convert_decimal_to_float(obj: Any) -> Any:
    """
    Recursively converts all Decimal objects in a dict/list structure to float.
    This is needed because Decimal objects are not JSON serializable.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal_to_float(item) for item in obj]
    return obj


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
            
            # Set triple barrier config if not fully defined
            tb_config = config.get("triple_barrier_config", {})
            if not tb_config:
                # Default triple barrier configuration
                take_profit = float(Decimal(str(config.get("default_take_profit_pct", 2.0))) / Decimal("100"))
                stop_loss = float(Decimal(str(config.get("default_stop_loss_pct", 3.0))) / Decimal("100"))
                
                config["triple_barrier_config"] = {
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "time_limit": self.config.get("control_params", {}).get("global_time_limit", 86400)
                }
            
            # Ensure other required parameters are set
            if "total_amount_quote" not in config:
                config["total_amount_quote"] = float(self.config.get("base_controller_config", {}).get("total_amount_quote", 1000.0))
            elif isinstance(config["total_amount_quote"], Decimal):
                config["total_amount_quote"] = float(config["total_amount_quote"])
            
            if "leverage" not in config:
                config["leverage"] = self.config.get("base_controller_config", {}).get("leverage", 20)
            
            # Convert any Decimal objects to float to ensure JSON serialization works
            config = convert_decimal_to_float(config)
            candidate.config = config
            
            logging.info(f"Adjusted config for {trading_pair} - ID: {candidate.id}")
            adjusted_candidates.append(candidate)
        
        return adjusted_candidates

    async def _prepare_and_launch_bots(self, selected_candidates: List[ConfigCandidate]):
        """
        Override parent method to ensure all configs are JSON serializable and use consistent naming.
        Deploy with all controller configs at once instead of one by one.
        """
        if not selected_candidates:
            logging.info("No candidates to deploy")
            return
            
        # Convert all Decimal objects to float in the configs
        for candidate in selected_candidates:
            candidate.config = convert_decimal_to_float(candidate.config)
        
        # Clean up any existing bots for these trading pairs
        await self._cleanup_existing_bots(selected_candidates)
        
        # Filter all configs to include only valid PMMVMLControllerConfig fields
        filtered_candidates = []
        for candidate in selected_candidates:
            filtered_config = self._filter_pmm_vml_config(candidate.config)
            filtered_candidates.append((candidate, filtered_config))
            await self.backend_api_client.add_controller_config(filtered_config)
            logging.info(f"Added controller config for {filtered_config.get('trading_pair')} with ID: {filtered_config.get('id')}")
        
        # Generate bot name
        trading_pairs_str = "_".join(c[1].get('trading_pair', '').replace('-', '').lower() for c in filtered_candidates)
        bot_name = f"pmm_vml_{self.connector_name}_{trading_pairs_str}"
        
        # Gather all controller configs
        controller_configs = [fc[1]["id"] + ".yml" for fc in filtered_candidates]
        
        # Deploy using the API's deploy_script_with_controllers method
        script_name = self.config["deploy_params"].get("script_name", "v2_with_controllers.py")
        image_name = self.config["deploy_params"].get("image_name", "hummingbot/hummingbot:latest")
        credentials = self.config["deploy_params"].get("credentials", "master_account")
        time_to_cash_out = self.config["deploy_params"].get("time_to_cash_out")
        
        logging.info(f"Deploying bot with controllers: {controller_configs}")
        deploy_resp = await self.backend_api_client.deploy_script_with_controllers(
            bot_name=bot_name,
            controller_configs=controller_configs,
            script_name=script_name,
            image_name=image_name,
            credentials=credentials,
            time_to_cash_out=time_to_cash_out
        )
        
        if deploy_resp.get("success", False):
            instance_name = self.extract_instance_name(deploy_resp)
            logging.info(f"Successfully deployed bot instance: {instance_name}")
            
            # Track the active bot
            self.active_bots[instance_name] = {
                "start_timestamp": time.time(),
                "controller_status": {fc[1]["id"]: "running" for fc in filtered_candidates}
            }
            
            # Mark configs as archived
            self.archived_configs.extend([fc[0].id for fc in filtered_candidates])
        else:
            # TODO: check for {'success': False, 'message': '409 Client Error for http+docker://localhost/v1.47/containers/create?name=hummingbot-pmm_vml_okx_aptusdt-2025.04.28_22.41: Conflict ("Conflict. The container name "/hummingbot-pmm_vml_okx_aptusdt-2025.04.28_22.41" is already in use by container "cc143710c7c760833a7615bbb8b5af7b7edb28aee02413b907c591e21f8bc61e". You have to remove (or rename) that container to be able to reuse that name.")'}
            logging.error(f"Error deploying bot: {deploy_resp.get('error', 'Unknown error')}")

    async def _cleanup_existing_bots(self, candidates: List[ConfigCandidate]):
        """
        Clean up any existing bots that might be running for the trading pairs in the candidates list
        """
        try:
            active_bots_data = await self.backend_api_client.get_active_bots_status()
            active_bots = active_bots_data.get("data", {})
            
            # Extract trading pairs from candidates
            trading_pairs = [c.config.get("trading_pair").replace("-", "").lower() for c in candidates if c.config.get("trading_pair")]
            
            # Find bots that match our naming pattern and contain any of our trading pairs
            for bot_name in list(active_bots.keys()):
                if "pmm_vml" in bot_name and self.connector_name in bot_name:
                    for tp in trading_pairs:
                        if tp in bot_name:
                            logging.info(f"Found existing bot {bot_name} that might conflict, stopping it")
                            # Remove from active bots tracking if it's there
                            if bot_name in self.active_bots:
                                del self.active_bots[bot_name]
                            
                            # Stop and remove the bot
                            try:
                                await self.backend_api_client.stop_bot(bot_name=bot_name)
                                await asyncio.sleep(self.controller_stop_delay)
                                await self.backend_api_client.stop_container(bot_name)
                                await asyncio.sleep(5.0)
                                await self.backend_api_client.remove_container(bot_name, archive_locally=True)
                                logging.info(f"Successfully stopped and removed bot {bot_name}")
                            except Exception as e:
                                logging.warning(f"Error stopping existing bot {bot_name}: {e}")
                            break
        except Exception as e:
            logging.warning(f"Error checking existing bots: {e}")

    def _filter_pmm_vml_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter config to only include fields valid for PMMVMLControllerConfig
        """
        # Valid fields in PMMVMLControllerConfig
        valid_fields = {
            "id", "controller_name", "connector_name", "trading_pair", 
            "candles_config", "mqtt_topic_prefix", 
            "default_buy_0_bps", "default_buy_1_step", 
            "default_sell_0_bps", "default_sell_1_step", 
            "default_take_profit_pct", "default_stop_loss_pct",
            "leverage", "total_amount_quote", "manual_kill_switch"
        }
        
        # Fields inherited from MarketMakingControllerConfigBase
        valid_inherited_fields = {
            "controller_type", "take_profit", "stop_loss"
        }
        
        valid_fields.update(valid_inherited_fields)
        
        # Create filtered config with only valid fields
        filtered_config = {k: v for k, v in config.items() if k in valid_fields}
        
        # Ensure required fields are present
        if "id" not in filtered_config:
            filtered_config["id"] = config.get("id")
        
        if "controller_name" not in filtered_config:
            filtered_config["controller_name"] = "pmm_vml"
        
        # Log what fields were removed
        removed_fields = set(config.keys()) - valid_fields
        if removed_fields:
            logging.info(f"Removed invalid fields from config: {removed_fields}")
        
        return filtered_config


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