# XGridT Configuration Generator

This tool generates optimized XGridT trading strategy configurations for multiple token pairs. 
It's designed to create profitable configs that prioritize high trading volume while ensuring 
break-even or better profit after accounting for trading fees.

## Features

- **High Volume Focus**: Optimizes configurations to maximize trading volume
- **Fee-Adjusted Profitability**: Ensures strategies are profitable after fee deduction 
- **Multi-Token Support**: Processes multiple tokens in a single batch
- **Performance Summary**: Provides detailed optimization metrics
- **Result Filtering**: Only generates configs for profitable strategies
- **Parameter Optimization**: Fine-tunes all critical strategy parameters

## Usage

### Method 1: Using the Task Runner

1. Create a task configuration file (e.g., `config/xgridt_config_generator_tasks.yml`):

```yaml
# XGridT Configuration Generator Task Configuration
---
task_runner:
  name: "quants-lab-task"
  tasks:
    - name: "XGridTConfigGenerator"
      module: "tasks.backtesting.xgridt_config_generator_task"
      class: "XGridTConfigGeneratorTask"
      frequency: 86400  # 24 hours in seconds
      params:
        root_path: "/app"
        connector_name: "okx"
        lookback_days: 60
        end_time_buffer_hours: 6
        n_trials: 200
        output_dir: "/app/configs/xgridt"
        selected_pairs: 
          - "ADA-USDT"
          - "BTC-USDT" 
          - "ETH-USDT"
          - "SOL-USDT"
        fee_pct: 0.1
        target_min_volume: 500
        profit_weight: 1.0
        volume_weight: 2.0
```

2. Run the task:

```bash
conda activate quants-lab
make run-task config=xgridt_config_generator_tasks.yml
```

### Method 2: Using the Example Script

1. Run the example script directly:

```bash
conda activate quants-lab
python examples/xgridt_config_generator_example.py
```

### Method 3: Using in Custom Code

```python
from tasks.backtesting.xgridt_config_generator_task import XGridTConfigGeneratorTask
from datetime import timedelta

config = {
    "root_path": "/path/to/root",
    "connector_name": "okx",
    "selected_pairs": ["ADA-USDT", "BTC-USDT"],
    "fee_pct": 0.1,
    "target_min_volume": 500
}

task = XGridTConfigGeneratorTask("XGridTConfigGenerator", timedelta(hours=24), config)
await task.execute()
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `root_path` | Root path of the project | Required |
| `connector_name` | Name of the exchange connector | "okx" |
| `selected_pairs` | List of trading pairs to optimize | ["ADA-USDT"] |
| `lookback_days` | Number of days of historical data to use | 30 |
| `end_time_buffer_hours` | Hours to exclude from recent data | 6 |
| `n_trials` | Number of optimization trials per trading pair | 100 |
| `output_dir` | Directory to save configuration files | "configs" |
| `fee_pct` | Trading fee percentage | 0.1 |
| `target_min_volume` | Target minimum trading volume | 500 |
| `profit_weight` | Weight for profit in objective function | 1.0 |
| `volume_weight` | Weight for volume in objective function | 2.0 |

## Optimization Parameters

The task optimizes the following XGridT strategy parameters:

- **EMA Periods**: Short, medium, and long EMAs for trend identification
- **Donchian Channel**: Length for determining price channels
- **NATR**: Length and multiplier for volatility-based order placement
- **Take Profit**: Default take profit level
- **Grid Configuration**: Amount, executors per side
- **Time Parameters**: Cooldown time between trades
- **Peak Detection**: Parameters for identifying significant price levels

## Output

The task generates:

1. **YAML Configuration Files**: One file per profitable trading pair in the specified output directory
2. **Optimization Summary CSV**: A summary of all optimization results including metrics
3. **Console Logs**: Detailed information about the optimization process

## Example Output Configuration

```yaml
id: xgridt_spot_ADA_USDT_20240501123456
controller_name: xgridt
controller_type: generic
total_amount_quote: 300.0
connector_name: okx
trading_pair: ADA-USDT
interval: 1m
ema_short: 7
ema_medium: 21
ema_long: 55
position_mode: ONE_WAY
leverage: 1
donchian_channel_length: 50
natr_length: 90
natr_multiplier: 1.8
tp_default: 0.02
# ... additional parameters ...
``` 