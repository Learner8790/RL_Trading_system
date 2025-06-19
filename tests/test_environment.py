"""
Tests for the trading environment
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import EnhancedTradingEnvironmentV2


class TestEnhancedTradingEnvironment:
    """Test suite for the trading environment"""
    
    @pytest.fixture
    def env(self):
        """Create a test environment"""
        config = {
            'environment': {
                'initial_cash': 1000000,
                'episode_length': 30,
                'warmup_period': 20,
                'max_position_per_stock': 50,
                'min_trade_value': 5000,
                'position_limit_pct': 0.20,
                'max_drawdown_limit': 0.10,
                'daily_loss_limit': 0.02
            },
            'assets': {
                'stocks': ['RELIANCE.NS', 'TCS.NS'],
                'data_period': '1y',
                'data_interval': '1d'
            }
        }
        return EnhancedTradingEnvironmentV2(config)
    
    def test_environment_initialization(self, env):
        """Test environment initializes correctly"""
        assert env.n_assets == 2
        assert env.initial_cash == 1000000
        assert env.state_dim > 0
        assert len(env.symbols) == 2
    
    def test_reset(self, env):
        """Test environment reset"""
        state = env.reset()
        
        assert isinstance(state, np.ndarray)
        assert len(state) == env.state_dim
        assert env.cash == env.initial_cash
        assert all(env.positions[symbol] == 0 for symbol in env.symbols)
        assert env.trade_count == 0
    
    def test_step_hold_action(self, env):
        """Test step with hold actions"""
        state = env.reset()
        
        # All hold actions
        actions = [1, 1]  # Hold for both assets
        confidences = [0.5, 0.5]
        
        next_state, reward, done, info = env.step(actions, confidences)
        
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert env.cash == env.initial_cash  # No trades executed
    
    def test_step_buy_action(self, env):
        """Test step with buy actions"""
        state = env.reset()
        
        # Buy actions with high confidence
        actions = [2, 2]  # Buy for both assets
        confidences = [0.8, 0.8]
        
        initial_cash = env.cash
        next_state, reward, done, info = env.step(actions, confidences)
        
        # Check that positions were opened
        assert any(env.positions[symbol] > 0 for symbol in env.symbols)
        assert env.cash < initial_cash  # Cash decreased
        assert env.trade_count > 0
    
    def test_state_normalization(self, env):
        """Test that state values are properly normalized"""
        state = env.reset()
        
        # Run several steps
        for _ in range(5):
            actions = np.random.randint(0, 3, size=env.n_assets)
            confidences = np.random.random(size=env.n_assets)
            state, _, done, _ = env.step(actions, confidences)
            if done:
                break
        
        # Check state bounds
        assert np.all(np.isfinite(state))
        assert np.all(state >= -10)
        assert np.all(state <= 10)
    
    def test_position_limits(self, env):
        """Test that position limits are enforced"""
        env.reset()
        
        # Try to buy maximum positions
        for _ in range(10):
            actions = [2] * env.n_assets  # All buy
            confidences = [0.9] * env.n_assets
            env.step(actions, confidences)
        
        # Check position limits
        portfolio_value = env._calculate_portfolio_value()
        for symbol in env.symbols:
            position_value = env.positions[symbol] * env.data[symbol]['close'][env.current_step]
            position_pct = position_value / portfolio_value
            assert position_pct <= env.position_limit_pct + 0.01  # Small tolerance
    
    def test_drawdown_limit(self, env):
        """Test that drawdown limit triggers episode end"""
        env.reset()
        
        # Simulate losses by reducing cash significantly
        env.cash = env.initial_cash * 0.85  # 15% loss
        env.peak_value = env.initial_cash
        
        # Step should detect drawdown breach
        actions = [1] * env.n_assets
        confidences = [0.5] * env.n_assets
        _, reward, done, _ = env.step(actions, confidences)
        
        assert done  # Episode should end
        assert reward < 0  # Should receive negative reward
    
    def test_reward_calculation(self, env):
        """Test reward calculation logic"""
        env.reset()
        
        # Set up a profitable scenario
        env.winning_trades = 15
        env.losing_trades = 5
        env.daily_returns = [0.01, 0.02, -0.005, 0.015, 0.01]
        
        portfolio_value = env._calculate_portfolio_value()
        reward = env._calculate_reward(
            daily_return=0.02,
            trades_executed=2,
            portfolio_value=portfolio_value
        )
        
        assert isinstance(reward, (int, float))
        assert -10 <= reward <= 10  # Reward is clipped
    
    def test_info_metrics(self, env):
        """Test that info dictionary contains expected metrics"""
        env.reset()
        
        # Execute some trades
        for _ in range(10):
            actions = np.random.randint(0, 3, size=env.n_assets)
            confidences = np.random.random(size=env.n_assets)
            _, _, done, info = env.step(actions, confidences)
            if done:
                break
        
        # Check info keys
        expected_keys = [
            'portfolio_value', 'net_return', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'win_rate',
            'trades', 'winning_trades', 'losing_trades'
        ]
        
        for key in expected_keys:
            assert key in info
            assert info[key] is not None