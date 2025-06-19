"""
Smart Position Sizing with Kelly Criterion
"""

import numpy as np
from collections import deque


class SmartPositionSizer:
    """
    Kelly Criterion-based position sizing with risk management
    """
    
    def __init__(self, max_position_pct=0.25, confidence_threshold=0.6):
        """
        Initialize position sizer
        
        Args:
            max_position_pct: Maximum position size as fraction of capital
            confidence_threshold: Minimum confidence for full position size
        """
        self.max_position_pct = max_position_pct
        self.confidence_threshold = confidence_threshold
        self.trade_history = deque(maxlen=100)
        
    def calculate_kelly_fraction(self):
        """
        Calculate Kelly fraction from recent trades
        
        Returns:
            kelly_fraction: Optimal betting fraction
        """
        if len(self.trade_history) < 20:
            return 0.02  # Conservative default
        
        wins = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.trade_history if t['pnl'] < 0]
        
        if not wins or not losses:
            return 0.02
        
        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 0.02
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b
        
        # Apply Kelly fraction with safety factor
        return max(0.01, min(kelly * 0.25, self.max_position_pct))
    
    def get_position_size(self, confidence, volatility, available_capital, price):
        """
        Get optimal position size based on multiple factors
        
        Args:
            confidence: Model confidence in the trade (0-1)
            volatility: Current market volatility
            available_capital: Available cash for trading
            price: Current asset price
            
        Returns:
            position_size: Number of shares to trade
        """
        # Base size from Kelly
        kelly_fraction = self.calculate_kelly_fraction()
        
        # Adjust for confidence
        if confidence < self.confidence_threshold:
            kelly_fraction *= confidence / self.confidence_threshold
        
        # Adjust for volatility (inverse relationship)
        vol_adjustment = min(1.0, 0.02 / volatility) if volatility > 0 else 1.0
        kelly_fraction *= vol_adjustment
        
        # Calculate position value
        position_value = available_capital * kelly_fraction
        shares = int(position_value / price)
        
        # Ensure minimum trade size
        min_shares = max(10, int(10000 / price))  # At least ₹10,000
        
        return max(min_shares, shares)
    
    def update_history(self, trade_result):
        """
        Update trade history for Kelly calculation
        
        Args:
            trade_result: Dictionary with trade outcome
        """
        self.trade_history.append(trade_result)
    
    def get_portfolio_heat(self):
        """
        Calculate current portfolio heat (risk exposure)
        
        Returns:
            heat: Current risk level (0-1)
        """
        if not self.trade_history:
            return 0.5
        
        recent_trades = list(self.trade_history)[-10:]
        recent_losses = sum(1 for t in recent_trades if t['pnl'] < 0)
        
        # Heat increases with recent losses
        heat = recent_losses / len(recent_trades)
        
        return heat
    
    def adjust_for_correlation(self, kelly_fraction, correlation):
        """
        Adjust position size based on portfolio correlation
        
        Args:
            kelly_fraction: Base Kelly fraction
            correlation: Average correlation with existing positions
            
        Returns:
            adjusted_fraction: Correlation-adjusted fraction
        """
        # Reduce position size for highly correlated assets
        if abs(correlation) > 0.7:
            kelly_fraction *= (1 - abs(correlation)) / 0.3
        
        return kelly_fraction
    
    def get_risk_metrics(self):
        """
        Calculate current risk metrics
        
        Returns:
            metrics: Dictionary of risk metrics
        """
        if len(self.trade_history) < 10:
            return {
                'win_rate': 0.5,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 1.0,
                'kelly_fraction': 0.02
            }
        
        wins = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.trade_history if t['pnl'] < 0]
        
        win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'kelly_fraction': self.calculate_kelly_fraction()
        }