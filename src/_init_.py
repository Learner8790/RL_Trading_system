"""
Enhanced RL Trading System - Source Package
"""

from .environment import EnhancedTradingEnvironmentV2
from .agent import AttentionTradingAgent
from .trainer import ImprovedPPOTrainer
from .indicators import AdvancedIndicators
from .utils import (
    set_seeds,
    setup_logger,
    evaluate_enhanced_agent,
    plot_results
)

__version__ = "2.0.0"
__all__ = [
    'EnhancedTradingEnvironmentV2',
    'AttentionTradingAgent',
    'ImprovedPPOTrainer',
    'AdvancedIndicators',
    'set_seeds',
    'setup_logger',
    'evaluate_enhanced_agent',
    'plot_results'
]