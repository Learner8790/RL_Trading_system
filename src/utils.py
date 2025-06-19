"""
Utility functions for the Enhanced RL Trading System
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import random
from datetime import datetime
from pathlib import Path


def set_seeds(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        logger: Configured logger
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def evaluate_enhanced_agent(env, agent, n_episodes=30, device=None, epsilon=0.05):
    """
    Comprehensive evaluation of the trading agent
    
    Args:
        env: Trading environment
        agent: Trained agent
        n_episodes: Number of evaluation episodes
        device: Torch device
        epsilon: Exploration rate during evaluation
        
    Returns:
        results_df: DataFrame with evaluation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n=== Evaluating Enhanced Agent V2 ({n_episodes} episodes) ===")
    
    results = []
    
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        
        actions_taken = []
        confidences_recorded = []
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                actions, confidences, _ = agent.get_action(state_tensor, epsilon=epsilon)
            
            actions_taken.append(actions)
            confidences_recorded.append(confidences)
            
            state, reward, done, info = env.step(actions, confidences)
        
        # Add average confidence to info
        info['avg_confidence'] = np.mean(confidences_recorded)
        info['confidence_std'] = np.std(confidences_recorded)
        
        results.append(info)
        
        if (ep + 1) % 10 == 0:
            print(f"\nEpisode {ep + 1}:")
            print(f"  Net Return: {info['net_return']:.2f}%")
            print(f"  Sharpe Ratio: {info['sharpe_ratio']:.2f}")
            print(f"  Sortino Ratio: {info['sortino_ratio']:.2f}")
            print(f"  Max Drawdown: {info['max_drawdown']:.2f}%")
            print(f"  Win Rate: {info['win_rate']:.1%}")
            print(f"  Trades: {info['trades']}")
            print(f"  Avg Confidence: {info['avg_confidence']:.2f}")
    
    return pd.DataFrame(results)


def plot_results(eval_results, trainer=None, save_path=None):
    """
    Create comprehensive visualization of results
    
    Args:
        eval_results: DataFrame with evaluation results
        trainer: Trainer object with training history (optional)
        save_path: Path to save the plot (optional)
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Returns Distribution
    ax1 = plt.subplot(3, 3, 1)
    returns = eval_results['net_return']
    ax1.hist(returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(returns.mean(), color='red', linestyle='--', 
                label=f'Mean: {returns.mean():.2f}%')
    ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title('Returns Distribution')
    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # 2. Sharpe Ratio Evolution
    ax2 = plt.subplot(3, 3, 2)
    sharpes = eval_results['sharpe_ratio']
    ax2.plot(sharpes, marker='o', markersize=4)
    ax2.axhline(sharpes.mean(), color='red', linestyle='--', 
                label=f'Mean: {sharpes.mean():.2f}')
    ax2.axhline(2.0, color='green', linestyle=':', alpha=0.5, label='Target: 2.0')
    ax2.set_title('Sharpe Ratio by Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Win Rate vs Returns
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(eval_results['win_rate'] * 100, eval_results['net_return'], 
                alpha=0.6, s=50, c=eval_results['trades'], cmap='viridis')
    ax3.set_xlabel('Win Rate (%)')
    ax3.set_ylabel('Net Return (%)')
    ax3.set_title('Win Rate vs Returns')
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Number of Trades')
    ax3.grid(True, alpha=0.3)
    
    # 4. Training Progress
    ax4 = plt.subplot(3, 3, 4)
    if trainer and len(trainer.episode_rewards) > 0:
        window = 50
        rewards_smooth = pd.Series(trainer.episode_rewards).rolling(
            window=window, min_periods=1).mean()
        ax4.plot(rewards_smooth, label='Reward (MA50)', color='blue')
        ax4.set_title('Training Progress')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Reward')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Risk-Return Scatter
    ax5 = plt.subplot(3, 3, 5)
    volatilities = []
    for idx, row in eval_results.iterrows():
        if row['sharpe_ratio'] != 0 and not np.isnan(row['sharpe_ratio']):
            vol = abs(row['net_return'] / row['sharpe_ratio']) * np.sqrt(252/60)
        else:
            vol = 20
        volatilities.append(vol)
    
    ax5.scatter(volatilities, eval_results['net_return'], alpha=0.6, s=50)
    ax5.set_xlabel('Volatility (%)')
    ax5.set_ylabel('Return (%)')
    ax5.set_title('Risk-Return Profile')
    ax5.grid(True, alpha=0.3)
    
    # Add efficient frontier reference
    x = np.linspace(0, max(volatilities), 100)
    y = 2 * x  # Sharpe ratio = 2 line
    ax5.plot(x, y, 'r--', alpha=0.5, label='Sharpe = 2.0')
    ax5.legend()
    
    # 6. Drawdown Analysis
    ax6 = plt.subplot(3, 3, 6)
    drawdowns = eval_results['max_drawdown']
    ax6.hist(drawdowns, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax6.axvline(drawdowns.mean(), color='black', linestyle='--', 
                label=f'Mean: {drawdowns.mean():.2f}%')
    ax6.set_title('Maximum Drawdown Distribution')
    ax6.set_xlabel('Max Drawdown (%)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    
    # 7. Trading Activity
    ax7 = plt.subplot(3, 3, 7)
    trades = eval_results['trades']
    returns = eval_results['net_return']
    
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax7.scatter(trades, returns, c=colors, alpha=0.6, s=50)
    ax7.set_xlabel('Number of Trades')
    ax7.set_ylabel('Net Return (%)')
    ax7.set_title('Trading Activity vs Performance')
    ax7.grid(True, alpha=0.3)
    
    # 8. Confidence Analysis
    ax8 = plt.subplot(3, 3, 8)
    if 'avg_confidence' in eval_results.columns:
        confidence = eval_results['avg_confidence']
        returns = eval_results['net_return']
        
        conf_bins = np.linspace(confidence.min(), confidence.max(), 6)
        bin_returns = []
        bin_labels = []
        
        for i in range(len(conf_bins)-1):
            mask = (confidence >= conf_bins[i]) & (confidence < conf_bins[i+1])
            if mask.any():
                bin_returns.append(returns[mask].mean())
                bin_labels.append(f'{conf_bins[i]:.2f}-{conf_bins[i+1]:.2f}')
        
        ax8.bar(range(len(bin_returns)), bin_returns, alpha=0.7)
        ax8.set_xticks(range(len(bin_labels)))
        ax8.set_xticklabels(bin_labels, rotation=45)
        ax8.set_xlabel('Confidence Level')
        ax8.set_ylabel('Average Return (%)')
        ax8.set_title('Returns by Confidence Level')
        ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Performance Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create summary text
    summary_text = f"""Performance Summary:
    
    Average Return: {eval_results['net_return'].mean():.2f}% ± {eval_results['net_return'].std():.2f}%
    Success Rate: {(eval_results['net_return'] > 0).mean():.1%}
    
    Sharpe Ratio: {eval_results['sharpe_ratio'].mean():.2f}
    Sortino Ratio: {eval_results['sortino_ratio'].mean():.2f}
    Max Drawdown: {eval_results['max_drawdown'].mean():.2f}%
    
    Win Rate: {eval_results['win_rate'].mean():.1%}
    Avg Trades/Episode: {eval_results['trades'].mean():.1f}
    
    Best Episode: {eval_results['net_return'].max():.2f}%
    Worst Episode: {eval_results['net_return'].min():.2f}%
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.show()


def calculate_portfolio_metrics(returns, risk_free_rate=0.06):
    """
    Calculate comprehensive portfolio metrics
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        metrics: Dictionary of portfolio metrics
    """
    returns = np.array(returns)
    
    # Basic statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Annualized metrics (assuming daily returns)
    annual_return = mean_return * 252
    annual_vol = std_return * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_dev = np.std(downside_returns) * np.sqrt(252)
        sortino = (annual_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0
    else:
        sortino = sharpe * 1.5
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    win_rate = np.mean(returns > 0)
    
    # Value at Risk (95%)
    var_95 = np.percentile(returns, 5)
    
    # Conditional Value at Risk
    cvar_95 = np.mean(returns[returns <= var_95])
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'var_95': var_95,
        'cvar_95': cvar_95
    }