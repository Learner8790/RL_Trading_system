# API Documentation - Enhanced RL Trading System V2

## Overview

This document provides comprehensive API documentation for the Enhanced RL Trading System. The system is designed with modularity and extensibility in mind, allowing researchers and practitioners to easily adapt and extend the framework for their specific needs.

## Core Components

### Environment Module (`src.environment`)

#### Class: `EnhancedTradingEnvironmentV2`

The trading environment simulates realistic market conditions including transaction costs, position limits, and risk constraints.

```python
class EnhancedTradingEnvironmentV2:
    def __init__(self, config=None)
```

**Parameters:**
- `config` (dict, optional): Configuration dictionary containing environment parameters. If None, uses default configuration.

**Key Attributes:**
- `n_assets` (int): Number of tradeable assets
- `state_dim` (int): Dimension of the state vector
- `symbols` (list): List of asset symbols
- `initial_cash` (float): Starting capital amount
- `episode_length` (int): Maximum steps per episode

**Methods:**

##### `reset() -> np.ndarray`
Reset the environment to initial state.

**Returns:**
- `state` (np.ndarray): Initial state vector of shape (state_dim,)

**Example:**
```python
env = EnhancedTradingEnvironmentV2()
initial_state = env.reset()
```

##### `step(actions, confidences) -> Tuple[np.ndarray, float, bool, dict]`
Execute trading actions in the environment.

**Parameters:**
- `actions` (list): List of actions for each asset. Actions are integers: 0=Sell, 1=Hold, 2=Buy
- `confidences` (list): List of confidence scores (0-1) for each action

**Returns:**
- `next_state` (np.ndarray): State after executing actions
- `reward` (float): Immediate reward
- `done` (bool): Whether episode has terminated
- `info` (dict): Additional information including portfolio metrics

**Example:**
```python
actions = [2, 1, 0, 1, 2]  # Buy, Hold, Sell, Hold, Buy
confidences = [0.8, 0.5, 0.9, 0.3, 0.7]
next_state, reward, done, info = env.step(actions, confidences)
```

### Agent Module (`src.agent`)

#### Class: `AttentionTradingAgent`

Neural network agent with attention mechanism for multi-asset trading.

```python
class AttentionTradingAgent(nn.Module):
    def __init__(self, state_dim, n_assets=5, hidden_dim=512)
```

**Parameters:**
- `state_dim` (int): Dimension of input state vector
- `n_assets` (int): Number of assets to trade
- `hidden_dim` (int): Hidden layer dimension

**Methods:**

##### `forward(state) -> Tuple[List[Tensor], List[Tensor], Tensor]`
Forward pass through the network.

**Parameters:**
- `state` (Tensor): State tensor of shape (batch_size, state_dim)

**Returns:**
- `action_logits` (List[Tensor]): List of action logits for each asset
- `confidences` (List[Tensor]): List of confidence scores
- `value` (Tensor): Estimated state value

##### `get_action(state, epsilon=0.05) -> Tuple[List[int], List[float], Tensor]`
Get actions with epsilon-greedy exploration.

**Parameters:**
- `state` (Tensor): Current state
- `epsilon` (float): Exploration rate

**Returns:**
- `actions` (List[int]): Selected actions
- `confidence_values` (List[float]): Confidence scores
- `value` (Tensor): State value estimate

### Trainer Module (`src.trainer`)

#### Class: `ImprovedPPOTrainer`

Implements Proximal Policy Optimization with enhancements for stable training.

```python
class ImprovedPPOTrainer:
    def __init__(self, env, agent, config=None, device=None, logger=None)
```

**Parameters:**
- `env`: Trading environment instance
- `agent`: Neural network agent
- `config` (dict): Training configuration
- `device` (torch.device): Computing device
- `logger` (logging.Logger): Logger instance

**Methods:**

##### `train(n_episodes=1500, start_episode=0)`
Train the agent using PPO.

**Parameters:**
- `n_episodes` (int): Number of training episodes
- `start_episode` (int): Starting episode number (for resuming)

##### `save_checkpoint(filepath)`
Save training checkpoint.

**Parameters:**
- `filepath` (str): Path to save checkpoint

### Indicators Module (`src.indicators`)

#### Class: `AdvancedIndicators`

Technical indicators for feature engineering.

**Static Methods:**

##### `rsi(prices, period=14) -> np.ndarray`
Calculate Relative Strength Index.

##### `macd(prices, fast=12, slow=26, signal=9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
Calculate MACD with signal line.

##### `bollinger_bands(prices, period=20, std_dev=2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
Calculate Bollinger Bands.

##### `atr(high, low, close, period=14) -> np.ndarray`
Calculate Average True Range.

### Position Sizing Module (`src.position_sizing`)

#### Class: `SmartPositionSizer`

Kelly Criterion-based position sizing with risk management.

```python
class SmartPositionSizer:
    def __init__(self, max_position_pct=0.25, confidence_threshold=0.6)
```

**Methods:**

##### `get_position_size(confidence, volatility, available_capital, price) -> int`
Calculate optimal position size.

**Parameters:**
- `confidence` (float): Model confidence (0-1)
- `volatility` (float): Current market volatility
- `available_capital` (float): Available cash
- `price` (float): Current asset price

**Returns:**
- `position_size` (int): Number of shares to trade

## Utility Functions (`src.utils`)

### Training Utilities

##### `set_seeds(seed=42)`
Set random seeds for reproducibility.

##### `setup_logger(name, log_file=None, level=logging.INFO) -> logging.Logger`
Create configured logger instance.

### Evaluation Functions

##### `evaluate_enhanced_agent(env, agent, n_episodes=30, device=None, epsilon=0.05) -> pd.DataFrame`
Comprehensive agent evaluation.

**Parameters:**
- `env`: Trading environment
- `agent`: Trained agent
- `n_episodes` (int): Number of evaluation episodes
- `device` (torch.device): Computing device
- `epsilon` (float): Exploration rate

**Returns:**
- `results_df` (pd.DataFrame): Evaluation results

### Visualization

##### `plot_results(eval_results, trainer=None, save_path=None)`
Create comprehensive performance visualizations.

**Parameters:**
- `eval_results` (pd.DataFrame): Evaluation results
- `trainer` (ImprovedPPOTrainer): Trainer with history
- `save_path` (str): Path to save plots

## Configuration Schema

The system uses YAML configuration with the following structure:

```yaml
environment:
  initial_cash: float  # Starting capital
  episode_length: int  # Days per episode
  max_position_per_stock: int  # Max shares per position
  position_limit_pct: float  # Max position as % of portfolio
  max_drawdown_limit: float  # Max allowed drawdown

assets:
  stocks: List[str]  # List of stock symbols
  data_period: str  # Historical data period (e.g., '2y')

agent:
  hidden_dim: int  # Hidden layer dimension
  n_attention_heads: int  # Number of attention heads
  dropout_rate: float  # Dropout probability

training:
  n_episodes: int  # Training episodes
  learning_rate: float  # Initial learning rate
  batch_size: int  # Batch size for updates
  gamma: float  # Discount factor
  clip_epsilon: float  # PPO clipping parameter
```

## Usage Examples

### Basic Training Loop

```python
from src import EnhancedTradingEnvironmentV2, AttentionTradingAgent, ImprovedPPOTrainer
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
env = EnhancedTradingEnvironmentV2(config)
agent = AttentionTradingAgent(env.state_dim, env.n_assets)
trainer = ImprovedPPOTrainer(env, agent, config['training'])

# Train agent
trainer.train(n_episodes=1500)
```

### Custom Environment Integration

```python
class CustomTradingEnvironment(EnhancedTradingEnvironmentV2):
    def _calculate_reward(self, daily_return, trades_executed, portfolio_value):
        # Implement custom reward function
        custom_reward = super()._calculate_reward(
            daily_return, trades_executed, portfolio_value
        )
        # Add custom logic
        return custom_reward
```

### Advanced Position Sizing

```python
from src.position_sizing import SmartPositionSizer

sizer = SmartPositionSizer(max_position_pct=0.3)

# Update with trade history
sizer.update_history({
    'pnl': 5000,
    'pnl_pct': 2.5
})

# Get position size
size = sizer.get_position_size(
    confidence=0.8,
    volatility=0.02,
    available_capital=100000,
    price=150
)
```

## Error Handling

The system implements comprehensive error handling:

```python
try:
    state = env.reset()
    actions = agent.get_action(state)
    next_state, reward, done, info = env.step(actions)
except ValueError as e:
    # Handle invalid actions or states
    logger.error(f"Invalid operation: {e}")
except RuntimeError as e:
    # Handle computation errors
    logger.error(f"Computation error: {e}")
```

## Performance Considerations

### Memory Management
- Environment pre-loads all historical data
- Agent uses gradient checkpointing for large models
- Trainer implements efficient trajectory collection

### Computational Optimization
- Vectorized operations for technical indicators
- Batch processing in neural networks
- GPU acceleration supported via PyTorch

### Scaling Guidelines
- For more assets: Increase `hidden_dim` proportionally
- For longer episodes: Adjust `n_steps` in trainer
- For larger datasets: Consider data streaming implementation

## Extending the Framework

### Adding New Indicators

```python
class CustomIndicators(AdvancedIndicators):
    @staticmethod
    def custom_indicator(prices, period=10):
        # Implement custom indicator
        return indicator_values
```

### Custom Neural Architectures

```python
class CustomAgent(AttentionTradingAgent):
    def __init__(self, state_dim, n_assets, hidden_dim):
        super().__init__(state_dim, n_assets, hidden_dim)
        # Add custom layers
        self.custom_layer = nn.LSTM(hidden_dim, hidden_dim)
```

### Alternative Reward Functions

```python
def sharpe_focused_reward(daily_return, portfolio_value, window=20):
    # Calculate rolling Sharpe ratio
    sharpe = calculate_rolling_sharpe(daily_return, window)
    return sharpe * scaling_factor
```

## Troubleshooting

### Common Issues

**Issue**: Environment data loading fails
```python
# Solution: Check internet connection and Yahoo Finance availability
# Alternative: Use pre-downloaded data or synthetic data generation
```

**Issue**: GPU out of memory
```python
# Solution: Reduce batch_size or hidden_dim in configuration
# Alternative: Enable gradient accumulation
```

**Issue**: Training instability
```python
# Solution: Reduce learning rate or clip_epsilon
# Alternative: Increase n_epochs for more stable updates
```

## Version History

- **v2.0.0**: Current version with attention mechanism and confidence-based position sizing
- **v1.0.0**: Initial release with basic PPO implementation

## Contributing

See CONTRIBUTING.md for guidelines on extending the framework and submitting improvements.