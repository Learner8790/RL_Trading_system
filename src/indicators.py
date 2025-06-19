"""
Advanced Technical Indicators for Trading
"""

import numpy as np
import pandas as pd


class AdvancedIndicators:
    """
    Enhanced technical indicators for better signals
    """
    
    @staticmethod
    def rsi(prices, period=14):
        """
        Relative Strength Index with smoothing
        
        Args:
            prices: Array of prices
            period: RSI period
            
        Returns:
            rsi: RSI values
        """
        if len(prices) < period + 1:
            return np.full_like(prices, 50.0)
            
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            rs = 100
        else:
            rs = up / down
            
        rsi = np.zeros_like(prices)
        rsi[:period] = 50.0
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            if down == 0:
                rsi[i] = 100
            else:
                rs = up / down
                rsi[i] = 100. - 100. / (1. + rs)
        
        return np.clip(rsi, 0, 100)
    
    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        """
        MACD with signal line
        
        Args:
            prices: Array of prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal EMA period
            
        Returns:
            macd: MACD line
            signal_line: Signal line
            histogram: MACD histogram
        """
        if len(prices) < slow + signal:
            return np.zeros(len(prices)), np.zeros(len(prices)), np.zeros(len(prices))
            
        prices_series = pd.Series(prices)
        exp1 = prices_series.ewm(span=fast, adjust=False).mean()
        exp2 = prices_series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        macd[:slow] = 0
        signal_line[:slow+signal] = 0
        histogram[:slow+signal] = 0
        
        return macd.values, signal_line.values, histogram.values
    
    @staticmethod
    def bollinger_bands(prices, period=20, std_dev=2):
        """
        Bollinger Bands with squeeze detection
        
        Args:
            prices: Array of prices
            period: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            upper_band: Upper band
            middle_band: Middle band (SMA)
            lower_band: Lower band
        """
        if len(prices) < period:
            return prices, prices, prices
            
        sma = pd.Series(prices).rolling(window=period, min_periods=1).mean()
        std = pd.Series(prices).rolling(window=period, min_periods=1).std()
        std = std.fillna(0.01)
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band.values, sma.values, lower_band.values
    
    @staticmethod
    def atr(high, low, close, period=14):
        """
        Average True Range
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            atr: Average True Range values
        """
        if len(high) < 2:
            return np.ones_like(high) * 0.01
            
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        atr = pd.Series(tr).rolling(window=period, min_periods=1).mean()
        return atr.values
    
    @staticmethod
    def momentum(prices, period=10):
        """
        Price momentum
        
        Args:
            prices: Array of prices
            period: Momentum period
            
        Returns:
            momentum: Momentum values
        """
        if len(prices) < period + 1:
            return np.zeros_like(prices)
        
        momentum = np.zeros_like(prices)
        momentum[period:] = (prices[period:] / prices[:-period] - 1) * 100
        
        return momentum
    
    @staticmethod
    def volume_weighted_price(prices, volumes, period=20):
        """
        Volume Weighted Average Price (VWAP)
        
        Args:
            prices: Array of prices
            volumes: Array of volumes
            period: VWAP period
            
        Returns:
            vwap: VWAP values
        """
        if len(prices) < period:
            return prices
        
        vwap = pd.Series(prices * volumes).rolling(window=period).sum() / pd.Series(volumes).rolling(window=period).sum()
        vwap = vwap.fillna(prices[0])
        
        return vwap.values
    
    @staticmethod
    def support_resistance_levels(prices, window=50, num_levels=3):
        """
        Identify support and resistance levels
        
        Args:
            prices: Array of prices
            window: Window for finding local extrema
            num_levels: Number of levels to return
            
        Returns:
            support: Support levels
            resistance: Resistance levels
        """
        if len(prices) < window:
            return [], []
        
        # Find local maxima and minima
        highs = []
        lows = []
        
        for i in range(window, len(prices) - window):
            if prices[i] == max(prices[i-window:i+window]):
                highs.append(prices[i])
            if prices[i] == min(prices[i-window:i+window]):
                lows.append(prices[i])
        
        # Cluster levels
        resistance = sorted(highs, reverse=True)[:num_levels] if highs else []
        support = sorted(lows)[:num_levels] if lows else []
        
        return support, resistance
    
    @staticmethod
    def adx(high, low, close, period=14):
        """
        Average Directional Index
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period
            
        Returns:
            adx: ADX values
        """
        if len(high) < period + 1:
            return np.ones_like(high) * 25.0
        
        # Calculate +DM and -DM
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_move = high[i] - high[i-1]
            minus_move = low[i-1] - low[i]
            
            if plus_move > minus_move and plus_move > 0:
                plus_dm[i] = plus_move
            if minus_move > plus_move and minus_move > 0:
                minus_dm[i] = minus_move
        
        # Calculate ATR
        atr_values = AdvancedIndicators.atr(high, low, close, period)
        
        # Smooth DM values
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).sum()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).sum()
        
        # Calculate DI values
        plus_di = 100 * plus_dm_smooth / atr_values
        minus_di = 100 * minus_dm_smooth / atr_values
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        adx = adx.fillna(25.0)
        
        return adx.values
    
    @staticmethod
    def rolling_volatility(prices, window=21):
        """
        Calculate rolling volatility
        
        Args:
            prices: Array of prices
            window: Volatility window
            
        Returns:
            volatility: Rolling volatility values
        """
        if len(prices) < window + 1:
            return np.ones_like(prices) * 0.02
        
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] / prices[:-1] - 1)
        
        volatility = pd.Series(returns).rolling(window=window).std() * np.sqrt(252)
        volatility = volatility.fillna(0.02)
        
        return volatility.values