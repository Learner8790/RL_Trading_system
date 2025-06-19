"""
Enhanced Trading Environment V2
"""

import numpy as np
import pandas as pd
import yfinance as yf
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from .indicators import AdvancedIndicators
from .position_sizing import SmartPositionSizer


class EnhancedTradingEnvironmentV2:
    """
    Improved environment with better features and risk management
    """
    
    def __init__(self, config=None):
        # Default configuration
        if config is None:
            config = {
                'environment': {
                    'initial_cash': 2000000,
                    'episode_length': 60,
                    'warmup_period': 50,
                    'max_position_per_stock': 100,
                    'min_trade_value': 10000,
                    'position_limit_pct': 0.20,
                    'max_drawdown_limit': 0.10,
                    'daily_loss_limit': 0.02
                },
                'assets': {
                    'stocks': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'],
                    'data_period': '2y',
                    'data_interval': '1d'
                }
            }
        
        self.config = config
        self.symbols = config['assets']['stocks']
        self.n_assets = len(self.symbols)
        
        # Portfolio settings
        self.initial_cash = config['environment']['initial_cash']
        self.max_position_per_stock = config['environment']['max_position_per_stock']
        self.min_trade_value = config['environment']['min_trade_value']
        self.position_limit_pct = config['environment']['position_limit_pct']
        
        # Risk management
        self.max_drawdown_limit = config['environment']['max_drawdown_limit']
        self.daily_loss_limit = config['environment']['daily_loss_limit']
        
        # Components
        self.indicators = AdvancedIndicators()
        self.position_sizer = SmartPositionSizer(
            max_position_pct=config.get('position_sizing', {}).get('max_position_pct', 0.25)
        )
        
        # Load and prepare data
        self._load_data()
        
        self.episode_length = config['environment']['episode_length']
        self.warmup_period = config['environment']['warmup_period']
        
        # State dimension
        # Per asset: price features(8) + technical(10) + microstructure(4) = 22
        # Portfolio: 8 features
        # Market: 6 features
        self.state_dim = self.n_assets * 22 + 8 + 6
    
    def _load_data(self):
        """Load and prepare market data"""
        print("\nLoading enhanced market data...")
        
        self.data = {}
        min_length = float('inf')
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    period=self.config['assets']['data_period'],
                    interval=self.config['assets']['data_interval']
                )
                
                if len(df) > 252:
                    self.data[symbol] = {
                        'close': df['Close'].values,
                        'open': df['Open'].values,
                        'high': df['High'].values,
                        'low': df['Low'].values,
                        'volume': df['Volume'].values
                    }
                    min_length = min(min_length, len(df))
                    print(f"  Loaded {len(df)} days for {symbol}")
                else:
                    raise ValueError("Insufficient data")
                    
            except Exception as e:
                print(f"  Generating synthetic data for {symbol}: {e}")
                self._generate_synthetic_data(symbol)
                min_length = min(min_length, 500)
        
        self.data_length = int(min_length)
        self._align_and_prepare_data()
    
    def _generate_synthetic_data(self, symbol):
        """Generate realistic synthetic data"""
        n_days = 500
        
        # Realistic market parameters
        params = {
            'RELIANCE.NS': {'mu': 0.0005, 'sigma': 0.022, 'mean_price': 2500},
            'TCS.NS': {'mu': 0.0006, 'sigma': 0.018, 'mean_price': 3500},
            'HDFCBANK.NS': {'mu': 0.0004, 'sigma': 0.020, 'mean_price': 1600},
            'INFY.NS': {'mu': 0.0007, 'sigma': 0.021, 'mean_price': 1500},
            'ICICIBANK.NS': {'mu': 0.0005, 'sigma': 0.023, 'mean_price': 900}
        }
        
        param = params.get(symbol, {'mu': 0.0005, 'sigma': 0.020, 'mean_price': 2000})
        
        # Generate correlated returns with regime switches
        returns = []
        regime = 'normal'
        
        for i in range(n_days):
            # Regime switching
            if np.random.random() < 0.02:
                regime = np.random.choice(['normal', 'bull', 'bear'], p=[0.6, 0.2, 0.2])
            
            # Adjust parameters based on regime
            if regime == 'bull':
                mu = param['mu'] * 2
                sigma = param['sigma'] * 0.8
            elif regime == 'bear':
                mu = -param['mu']
                sigma = param['sigma'] * 1.5
            else:
                mu = param['mu']
                sigma = param['sigma']
            
            # Generate return with fat tails
            if np.random.random() < 0.05:
                ret = np.random.normal(mu, sigma * 3)
            else:
                ret = np.random.normal(mu, sigma)
            
            returns.append(ret)
        
        # Convert to prices
        returns = np.array(returns)
        prices = param['mean_price'] * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        daily_range = np.abs(returns) * prices
        opens = prices - daily_range * np.random.uniform(-0.3, 0.3, size=n_days)
        highs = prices + daily_range * np.random.uniform(0.3, 0.7, size=n_days)
        lows = prices - daily_range * np.random.uniform(0.3, 0.7, size=n_days)
        
        # Volume correlated with volatility
        base_volume = 10000000
        volumes = base_volume * (1 + np.abs(returns) * 50) * np.random.uniform(0.8, 1.2, size=n_days)
        
        self.data[symbol] = {
            'close': prices,
            'open': opens,
            'high': highs,
            'low': lows,
            'volume': volumes.astype(int)
        }
    
    def _align_and_prepare_data(self):
        """Align data and calculate indicators"""
        # Align all data
        for symbol in self.symbols:
            for key in self.data[symbol]:
                self.data[symbol][key] = self.data[symbol][key][-self.data_length:]
        
        # Calculate indicators for each symbol
        self.indicators_data = {}
        
        for symbol in self.symbols:
            prices = self.data[symbol]['close']
            highs = self.data[symbol]['high']
            lows = self.data[symbol]['low']
            volumes = self.data[symbol]['volume']
            
            # Calculate all indicators
            rsi = self.indicators.rsi(prices)
            macd, signal, histogram = self.indicators.macd(prices)
            upper, middle, lower = self.indicators.bollinger_bands(prices)
            atr = self.indicators.atr(highs, lows, prices)
            momentum = self.indicators.momentum(prices)
            vwap = self.indicators.volume_weighted_price(prices, volumes)
            support, resistance = self.indicators.support_resistance_levels(prices)
            
            self.indicators_data[symbol] = {
                'rsi': rsi,
                'macd': macd,
                'macd_signal': signal,
                'macd_histogram': histogram,
                'bb_upper': upper,
                'bb_middle': middle,
                'bb_lower': lower,
                'atr': atr,
                'momentum': momentum,
                'vwap': vwap,
                'support': support,
                'resistance': resistance
            }
        
        # Calculate market-wide indicators
        self._calculate_market_indicators()
    
    def _calculate_market_indicators(self):
        """Calculate market-wide indicators"""
        # Market breadth
        self.market_indicators = {
            'trend': np.zeros(self.data_length),
            'volatility': np.ones(self.data_length) * 0.15,
            'correlation': np.zeros(self.data_length),
            'volume_trend': np.zeros(self.data_length),
            'breadth': np.zeros(self.data_length),
            'momentum': np.zeros(self.data_length)
        }
        
        # Calculate rolling market statistics
        window = 20
        
        for i in range(window, self.data_length):
            # Average returns across all stocks
            returns = []
            for symbol in self.symbols:
                prices = self.data[symbol]['close']
                ret = (prices[i] / prices[i-1] - 1)
                returns.append(ret)
            
            # Market trend (average return)
            self.market_indicators['trend'][i] = np.mean(returns)
            
            # Market volatility
            recent_returns = []
            for j in range(i-window, i):
                day_returns = []
                for symbol in self.symbols:
                    prices = self.data[symbol]['close']
                    ret = (prices[j] / prices[j-1] - 1)
                    day_returns.append(ret)
                recent_returns.append(np.mean(day_returns))
            
            self.market_indicators['volatility'][i] = np.std(recent_returns) * np.sqrt(252)
            
            # Market breadth (% of stocks above 20-day average)
            above_ma = 0
            for symbol in self.symbols:
                prices = self.data[symbol]['close']
                ma20 = np.mean(prices[i-20:i])
                if prices[i] > ma20:
                    above_ma += 1
            self.market_indicators['breadth'][i] = above_ma / len(self.symbols)
    
    def reset(self):
        """Reset environment for new episode"""
        # Random start point
        max_start = self.data_length - self.episode_length - 10
        min_start = self.warmup_period
        
        self.current_step = np.random.randint(min_start, max_start)
        self.start_step = self.current_step
        
        # Portfolio initialization
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.cash = self.initial_cash
        self.avg_buy_prices = {symbol: 0 for symbol in self.symbols}
        
        # Risk tracking
        self.peak_value = self.initial_cash
        self.daily_pnl = 0
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # History tracking
        self.portfolio_values = [self.initial_cash]
        self.trade_history = []
        self.daily_returns = []
        
        # Cost tracking
        self.total_costs = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Get enhanced state representation"""
        state = []
        idx = self.current_step
        
        # Per-asset features (22 each)
        for symbol in self.symbols:
            # Price data
            close = self.data[symbol]['close'][idx]
            open_price = self.data[symbol]['open'][idx]
            high = self.data[symbol]['high'][idx]
            low = self.data[symbol]['low'][idx]
            volume = self.data[symbol]['volume'][idx]
            
            # Price features (8)
            returns = []
            for lookback in [1, 2, 5, 10, 20]:
                if idx >= lookback:
                    ret = (close / self.data[symbol]['close'][idx - lookback] - 1) * 100
                else:
                    ret = 0
                returns.append(np.clip(ret, -10, 10))
            
            # Add OHLC features
            oc_ratio = (close - open_price) / open_price * 100 if open_price > 0 else 0
            hl_ratio = (high - low) / low * 100 if low > 0 else 0
            close_to_high = (high - close) / high * 100 if high > 0 else 0
            
            price_features = returns + [oc_ratio, hl_ratio, close_to_high]
            
            # Technical indicators (10)
            indicators = self.indicators_data[symbol]
            
            rsi_norm = (indicators['rsi'][idx] - 50) / 50
            
            macd_norm = indicators['macd'][idx] / close * 100
            signal_norm = indicators['macd_signal'][idx] / close * 100
            
            bb_width = indicators['bb_upper'][idx] - indicators['bb_lower'][idx]
            bb_position = (close - indicators['bb_lower'][idx]) / bb_width if bb_width > 0 else 0.5
            bb_width_norm = bb_width / close * 100 if close > 0 else 0
            
            atr_norm = indicators['atr'][idx] / close * 100
            momentum_norm = indicators['momentum'][idx] / 10
            
            vwap_diff = (close - indicators['vwap'][idx]) / indicators['vwap'][idx] * 100
            
            # Support/Resistance distance
            if indicators['support']:
                nearest_support = min(indicators['support'], key=lambda x: abs(x - close))
                support_dist = (close - nearest_support) / close * 100
            else:
                support_dist = 0
            
            if indicators['resistance']:
                nearest_resistance = min(indicators['resistance'], key=lambda x: abs(x - close))
                resistance_dist = (nearest_resistance - close) / close * 100
            else:
                resistance_dist = 0
            
            technical_features = [
                rsi_norm, macd_norm, signal_norm, bb_position, bb_width_norm,
                atr_norm, momentum_norm, vwap_diff, support_dist, resistance_dist
            ]
            
            # Microstructure features (4)
            if idx >= 20:
                avg_volume = np.mean(self.data[symbol]['volume'][idx-20:idx])
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            else:
                volume_ratio = 1
            
            position_ratio = self.positions[symbol] / self.max_position_per_stock
            position_value = self.positions[symbol] * close
            position_pct = position_value / self._calculate_portfolio_value()
            
            if self.positions[symbol] > 0 and self.avg_buy_prices[symbol] > 0:
                unrealized_pnl = (close / self.avg_buy_prices[symbol] - 1) * 100
            else:
                unrealized_pnl = 0
            
            microstructure_features = [
                np.log10(volume_ratio + 1), position_ratio, position_pct * 10, unrealized_pnl / 10
            ]
            
            # Combine all features for this asset
            asset_features = price_features + technical_features + microstructure_features
            asset_features = [np.clip(f, -10, 10) if np.isfinite(f) else 0 for f in asset_features]
            state.extend(asset_features)
        
        # Portfolio features (8)
        portfolio_value = self._calculate_portfolio_value()
        
        total_return = (portfolio_value / self.initial_cash - 1) * 100
        cash_ratio = self.cash / portfolio_value
        
        # Calculate current drawdown
        self.peak_value = max(self.peak_value, portfolio_value)
        current_drawdown = (portfolio_value - self.peak_value) / self.peak_value * 100
        
        # Recent performance
        if len(self.daily_returns) >= 10:
            recent_returns = self.daily_returns[-10:]
            recent_mean = np.mean(recent_returns) * 252
            recent_std = np.std(recent_returns) * np.sqrt(252)
            recent_sharpe = recent_mean / (recent_std + 0.001)
            win_rate_recent = sum(1 for r in recent_returns if r > 0) / len(recent_returns)
        else:
            recent_mean = 0
            recent_std = 0.15
            recent_sharpe = 0
            win_rate_recent = 0.5
        
        n_positions = sum(1 for p in self.positions.values() if p > 0)
        position_concentration = max(self.positions.values()) / sum(self.positions.values()) if sum(self.positions.values()) > 0 else 0
        
        portfolio_features = [
            total_return / 10, cash_ratio, current_drawdown / 10, recent_mean * 10,
            recent_std * 10, recent_sharpe, win_rate_recent, n_positions / self.n_assets
        ]
        
        # Market features (6)
        market_features = [
            self.market_indicators['trend'][idx] * 100,
            self.market_indicators['volatility'][idx] * 10,
            self.market_indicators['breadth'][idx],
            self.market_indicators['correlation'][idx],
            self.market_indicators['volume_trend'][idx],
            self.market_indicators['momentum'][idx] * 10
        ]
        
        # Combine all features
        state.extend(portfolio_features)
        state.extend(market_features)
        
        # Ensure all values are finite
        state = np.array(state, dtype=np.float32)
        state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return state
    
    def _calculate_portfolio_value(self):
        """Calculate current portfolio value"""
        value = self.cash
        idx = self.current_step
        
        for symbol in self.symbols:
            if self.positions[symbol] > 0:
                current_price = self.data[symbol]['close'][idx]
                value += self.positions[symbol] * current_price
        
        return value
    
    def step(self, actions, confidences):
        """Execute actions with enhanced logic"""
        old_portfolio_value = self._calculate_portfolio_value()
        idx = self.current_step
        
        # Risk checks
        current_drawdown = (old_portfolio_value - self.peak_value) / self.peak_value
        if current_drawdown < -self.max_drawdown_limit:
            # Force close all positions if max drawdown hit
            self._close_all_positions()
            self.current_step += 1
            return self._get_state(), -10, True, self._get_info()
        
        # Execute trades
        trades_executed = 0
        
        for i, symbol in enumerate(self.symbols):
            action = actions[i]
            confidence = confidences[i]
            
            current_price = self.data[symbol]['close'][idx]
            current_volume = self.data[symbol]['volume'][idx]
            
            # Get current volatility for position sizing
            atr = self.indicators_data[symbol]['atr'][idx]
            volatility = atr / current_price
            
            if action == 2 and confidence > 0.5:  # Buy with confidence threshold
                # Check if we have room for position
                position_value = self.positions[symbol] * current_price
                portfolio_value = self._calculate_portfolio_value()
                
                if position_value / portfolio_value < self.position_limit_pct:
                    # Get position size
                    position_size = self.position_sizer.get_position_size(
                        confidence, volatility, self.cash, current_price
                    )
                    
                    # Check technical conditions
                    rsi = self.indicators_data[symbol]['rsi'][idx]
                    momentum = self.indicators_data[symbol]['momentum'][idx]
                    
                    if 25 < rsi < 70 and momentum > -5:  # Not extremely oversold/overbought
                        trade_value = position_size * current_price
                        
                        if self.cash >= trade_value * 1.01:  # Include buffer for costs
                            # Execute buy
                            self._execute_buy(symbol, position_size, current_price)
                            trades_executed += 1
            
            elif action == 0 and self.positions[symbol] > 0:  # Sell
                # Calculate current P&L
                pnl_pct = (current_price / self.avg_buy_prices[symbol] - 1) * 100
                
                # Dynamic exit conditions
                rsi = self.indicators_data[symbol]['rsi'][idx]
                momentum = self.indicators_data[symbol]['momentum'][idx]
                
                # Adaptive stop-loss and take-profit based on volatility
                atr_pct = atr / current_price * 100
                stop_loss = -max(2, min(5, atr_pct * 2))  # Dynamic stop loss
                take_profit = max(3, min(10, atr_pct * 3))  # Dynamic take profit
                
                should_sell = (
                    pnl_pct < stop_loss or
                    pnl_pct > take_profit or
                    (rsi > 75 and momentum < 0) or  # Overbought and losing momentum
                    confidence > 0.7  # High confidence in sell signal
                )
                
                if should_sell:
                    # Determine sell quantity
                    if pnl_pct < stop_loss or confidence > 0.8:
                        sell_quantity = self.positions[symbol]  # Full exit
                    else:
                        sell_quantity = max(self.positions[symbol] // 2, 10)  # Partial exit
                    
                    self._execute_sell(symbol, sell_quantity, current_price)
                    trades_executed += 1
        
        # Update step
        self.current_step += 1
        
        # Calculate new portfolio value and metrics
        new_portfolio_value = self._calculate_portfolio_value()
        self.portfolio_values.append(new_portfolio_value)
        
        # Daily return
        daily_return = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        self.daily_returns.append(daily_return)
        
        # Calculate reward
        reward = self._calculate_reward(daily_return, trades_executed, new_portfolio_value)
        
        # Check if episode is done
        done = (
            self.current_step >= self.start_step + self.episode_length or
            self.current_step >= self.data_length - 1 or
            new_portfolio_value < self.initial_cash * 0.75  # 25% loss limit
        )
        
        return self._get_state(), reward, done, self._get_info()
    
    def _execute_buy(self, symbol, quantity, price):
        """Execute buy order with costs"""
        # Calculate costs (simplified)
        trade_value = quantity * price
        costs = trade_value * 0.001  # 0.1% total costs
        
        total_cost = trade_value + costs
        
        if self.cash >= total_cost:
            self.cash -= total_cost
            
            # Update position
            if self.positions[symbol] == 0:
                self.avg_buy_prices[symbol] = price
            else:
                total_shares = self.positions[symbol] + quantity
                self.avg_buy_prices[symbol] = (
                    (self.avg_buy_prices[symbol] * self.positions[symbol] + price * quantity) / total_shares
                )
            
            self.positions[symbol] += quantity
            self.total_costs += costs
            self.trade_count += 1
            
            # Record trade
            self.trade_history.append({
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'cost': total_cost
            })
    
    def _execute_sell(self, symbol, quantity, price):
        """Execute sell order with costs"""
        quantity = min(quantity, self.positions[symbol])
        
        if quantity > 0:
            # Calculate proceeds and costs
            trade_value = quantity * price
            costs = trade_value * 0.0015  # 0.15% total costs (including STT)
            
            proceeds = trade_value - costs
            
            # Calculate P&L
            cost_basis = self.avg_buy_prices[symbol] * quantity
            gross_pnl = trade_value - cost_basis
            net_pnl = gross_pnl - costs
            
            # Update portfolio
            self.cash += proceeds
            self.positions[symbol] -= quantity
            self.total_costs += costs
            self.trade_count += 1
            
            # Track wins/losses
            if net_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Update position sizer
            self.position_sizer.update_history({
                'pnl': net_pnl,
                'pnl_pct': net_pnl / cost_basis * 100
            })
            
            # Record trade
            self.trade_history.append({
                'symbol': symbol,
                'action': 'SELL',
                'quantity': quantity,
                'price': price,
                'pnl': net_pnl,
                'pnl_pct': net_pnl / cost_basis * 100
            })
    
    def _close_all_positions(self):
        """Emergency close all positions"""
        idx = self.current_step
        
        for symbol in self.symbols:
            if self.positions[symbol] > 0:
                price = self.data[symbol]['close'][idx]
                self._execute_sell(symbol, self.positions[symbol], price)
    
    def _calculate_reward(self, daily_return, trades_executed, portfolio_value):
        """Enhanced reward function"""
        # 1. Return component with risk adjustment
        if len(self.daily_returns) > 5:
            recent_vol = np.std(self.daily_returns[-5:])
            if recent_vol > 0:
                risk_adjusted_return = daily_return / recent_vol
            else:
                risk_adjusted_return = daily_return * 20
        else:
            risk_adjusted_return = daily_return * 10
        
        # 2. Drawdown penalty
        current_drawdown = (portfolio_value - self.peak_value) / self.peak_value
        drawdown_penalty = max(0, -current_drawdown * 20)  # Heavily penalize drawdowns
        
        # 3. Win rate bonus
        if self.winning_trades + self.losing_trades > 10:
            win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
            win_rate_bonus = (win_rate - 0.5) * 5  # Bonus for > 50% win rate
        else:
            win_rate_bonus = 0
        
        # 4. Consistency bonus
        if len(self.daily_returns) > 20:
            positive_days = sum(1 for r in self.daily_returns[-20:] if r > 0)
            consistency_bonus = (positive_days / 20 - 0.5) * 2
        else:
            consistency_bonus = 0
        
        # 5. Cost efficiency
        if trades_executed > 0:
            avg_trade_size = (portfolio_value - self.cash) / max(1, sum(p > 0 for p in self.positions.values()))
            if avg_trade_size > 50000:  # Reward larger, more efficient trades
                efficiency_bonus = 0.5
            else:
                efficiency_bonus = -0.2
        else:
            efficiency_bonus = 0
        
        # Combined reward
        reward = (
            risk_adjusted_return * 40 +
            win_rate_bonus * 20 -
            drawdown_penalty * 30 +
            consistency_bonus * 10 +
            efficiency_bonus * 10
        )
        
        return np.clip(reward, -10, 10)
    
    def _get_info(self):
        """Get comprehensive episode information"""
        portfolio_value = self._calculate_portfolio_value()
        gross_pnl = portfolio_value - self.initial_cash
        net_pnl = gross_pnl - self.total_costs
        
        # Calculate metrics
        if len(self.daily_returns) > 0:
            avg_return = np.mean(self.daily_returns)
            volatility = np.std(self.daily_returns)
            
            if volatility > 0:
                sharpe = avg_return / volatility * np.sqrt(252)
            else:
                sharpe = 0
            
            # Max drawdown
            cumulative = np.array(self.portfolio_values)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdowns) * 100
            
            # Sortino ratio (downside deviation)
            negative_returns = [r for r in self.daily_returns if r < 0]
            if negative_returns:
                downside_dev = np.std(negative_returns) * np.sqrt(252)
                sortino = avg_return * 252 / downside_dev if downside_dev > 0 else 0
            else:
                sortino = sharpe * 1.5  # If no negative returns, approximate
        else:
            sharpe = 0
            sortino = 0
            max_drawdown = 0
        
        win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)
        
        return {
            'portfolio_value': portfolio_value,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'total_costs': self.total_costs,
            'positions': self.positions.copy(),
            'trades': self.trade_count,
            'win_rate': win_rate,
            'gross_return': gross_pnl / self.initial_cash * 100,
            'net_return': net_pnl / self.initial_cash * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }