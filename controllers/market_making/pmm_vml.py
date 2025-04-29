import asyncio
import logging
from decimal import Decimal
from typing import Dict, List

from pydantic import Field

from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.remote_iface.mqtt import ExternalTopicFactory
from hummingbot.strategy_v2.controllers.market_making_controller_base import (
    MarketMakingControllerBase,
    MarketMakingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig


class PMMVMLControllerConfig(MarketMakingControllerConfigBase):
    controller_name: str = "pmm_vml"
    candles_config: List[CandlesConfig] = Field(default=[])
    
    # MQTT configuration
    mqtt_topic_prefix: str = Field(
        default="hbot/pmm_vml_params", 
        json_schema_extra={"prompt": "Enter the MQTT topic prefix for VML parameters: ", "prompt_on_new": True}
    )
    
    # Default values for predicted parameters (used until MQTT signals are received)
    default_buy_0_bps: Decimal = Field(
        default=Decimal("10"), 
        json_schema_extra={"prompt": "Enter default buy 0 spread in basis points: ", "prompt_on_new": True, "is_updatable": True}
    )
    default_buy_1_step: Decimal = Field(
        default=Decimal("10"), 
        json_schema_extra={"prompt": "Enter default buy 1 step size in basis points: ", "prompt_on_new": True, "is_updatable": True}
    )
    default_sell_0_bps: Decimal = Field(
        default=Decimal("10"), 
        json_schema_extra={"prompt": "Enter default sell 0 spread in basis points: ", "prompt_on_new": True, "is_updatable": True}
    )
    default_sell_1_step: Decimal = Field(
        default=Decimal("10"), 
        json_schema_extra={"prompt": "Enter default sell 1 step size in basis points: ", "prompt_on_new": True, "is_updatable": True}
    )
    default_take_profit_pct: Decimal = Field(
        default=Decimal("2.0"), 
        json_schema_extra={"prompt": "Enter default take profit percentage: ", "prompt_on_new": True, "is_updatable": True}
    )
    default_stop_loss_pct: Decimal = Field(
        default=Decimal("3.0"), 
        json_schema_extra={"prompt": "Enter default stop loss percentage: ", "prompt_on_new": True, "is_updatable": True}
    )


class PMMVMLController(MarketMakingControllerBase):
    def __init__(self, config: PMMVMLControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        
        # Initialize with default values
        self.latest_params = {
            "buy_0_bps": float(self.config.default_buy_0_bps),
            "buy_1_step": float(self.config.default_buy_1_step),
            "sell_0_bps": float(self.config.default_sell_0_bps),
            "sell_1_step": float(self.config.default_sell_1_step),
            "take_profit_pct": float(self.config.default_take_profit_pct),
            "stop_loss_pct": float(self.config.default_stop_loss_pct),
        }
        self.last_parameter_update = 0
        
        # Initialize the ML signal listener
        self._init_parameter_listener()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def _init_parameter_listener(self):
        """Initialize a listener for ML-predicted PMM parameters from the MQTT broker"""
        try:
            normalized_pair = self.config.trading_pair.replace("-", "_").lower()
            topic = f"{self.config.mqtt_topic_prefix}/{normalized_pair}/PMM_PARAMS"
            self._ml_signal_listener = ExternalTopicFactory.create_async(
                topic=topic,
                callback=self._handle_parameter_signal,
                use_bot_prefix=False,
            )
            self.logger().info(f"PMM VML parameter listener initialized successfully on topic: {topic}")
        except Exception as e:
            self.logger().error(f"Failed to initialize PMM VML parameter listener: {str(e)}")
            self._ml_signal_listener = None
    
    def _handle_parameter_signal(self, signal: Dict, topic: str):
        """Handle incoming PMM parameter signal"""
        try:
            self.logger().debug(f"Received PMM VML parameters: {signal}")
            
            # Extract parameters from signal
            parameters = signal.get("parameters", {})
            if not parameters:
                self.logger().warning("Received signal with no parameters")
                return
            
            # Update latest parameters
            self.latest_params["buy_0_bps"] = parameters.get("buy_0_bps", float(self.config.default_buy_0_bps))
            self.latest_params["buy_1_step"] = parameters.get("buy_1_step", float(self.config.default_buy_1_step))
            self.latest_params["sell_0_bps"] = parameters.get("sell_0_bps", float(self.config.default_sell_0_bps))
            self.latest_params["sell_1_step"] = parameters.get("sell_1_step", float(self.config.default_sell_1_step))
            self.latest_params["take_profit_pct"] = parameters.get("take_profit_pct", float(self.config.default_take_profit_pct))
            self.latest_params["stop_loss_pct"] = parameters.get("stop_loss_pct", float(self.config.default_stop_loss_pct))
            
            # Record timestamp
            self.last_parameter_update = signal.get("timestamp", 0)
            
            self.logger().info(f"Updated PMM VML parameters: buy_spreads=[{self.latest_params['buy_0_bps']/10000}, {(self.latest_params['buy_0_bps']+self.latest_params['buy_1_step'])/10000}], "
                              f"sell_spreads=[{self.latest_params['sell_0_bps']/10000}, {(self.latest_params['sell_0_bps']+self.latest_params['sell_1_step'])/10000}], "
                              f"take_profit={self.latest_params['take_profit_pct']/100}, stop_loss={self.latest_params['stop_loss_pct']/100}")
            
            # Update the processed data to reflect new parameters
            self.processed_data.update({
                "latest_params": self.latest_params,
                "last_parameter_update": self.last_parameter_update
            })
        except Exception as e:
            self.logger().error(f"Error processing PMM VML parameters: {str(e)}")
    
    async def update_processed_data(self):
        """Update the processed data based on the market data and latest parameters"""
        # Call the parent method to update reference price, etc.
        await super().update_processed_data()
        
        # Update buy/sell spreads based on the latest parameters
        buy_0_bps = self.latest_params.get("buy_0_bps", float(self.config.default_buy_0_bps))
        buy_1_step = self.latest_params.get("buy_1_step", float(self.config.default_buy_1_step))
        sell_0_bps = self.latest_params.get("sell_0_bps", float(self.config.default_sell_0_bps))
        sell_1_step = self.latest_params.get("sell_1_step", float(self.config.default_sell_1_step))
        
        # Convert to spreads (basis points -> decimal format)
        self.config.buy_spreads = [Decimal(buy_0_bps) / Decimal("10000"), 
                                   Decimal(buy_0_bps + buy_1_step) / Decimal("10000")]
        self.config.sell_spreads = [Decimal(sell_0_bps) / Decimal("10000"), 
                                    Decimal(sell_0_bps + sell_1_step) / Decimal("10000")]
        
        # Update triple barrier config with latest take profit and stop loss
        tp_pct = self.latest_params.get("take_profit_pct", float(self.config.default_take_profit_pct))
        sl_pct = self.latest_params.get("stop_loss_pct", float(self.config.default_stop_loss_pct))
        
        self.config.take_profit = Decimal(tp_pct) / Decimal("100")
        self.config.stop_loss = Decimal(sl_pct) / Decimal("100")
        
        # Add latest parameters to processed data for reference
        self.processed_data.update({
            "latest_params": self.latest_params,
            "last_parameter_update": self.last_parameter_update
        })

    def get_executor_config(self, level_id: str, price: Decimal, amount: Decimal):
        """
        Get the executor config based on the level_id, price and amount.
        """
        trade_type = self.get_trade_type_from_level_id(level_id)
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            level_id=level_id,
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            entry_price=price,
            amount=amount,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage,
            side=trade_type,
        )

    def to_format_status(self) -> List[str]:
        """Format status for display"""
        lines = []
        lines.append(f"PMM VML Controller - {self.config.trading_pair} on {self.config.connector_name}")
        lines.append(f"Reference price: {self.processed_data.get('reference_price', 'N/A')}")
        
        lines.append("\nCurrent Parameters:")
        lines.append(f"Buy spreads: {self.config.buy_spreads}")
        lines.append(f"Sell spreads: {self.config.sell_spreads}")
        lines.append(f"Take profit: {self.config.take_profit}")
        lines.append(f"Stop loss: {self.config.stop_loss}")
        
        lines.append("\nLatest VML Parameters:")
        for key, value in self.latest_params.items():
            lines.append(f"  {key}: {value}")
        
        lines.append(f"\nLast parameter update: {self.last_parameter_update}")
        return lines 