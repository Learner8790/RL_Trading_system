"""
Example script for backtesting the trained agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import EnhancedTradingEnvironmentV2
from src.agent import AttentionTradingAgent
from src.utils import evaluate_enhanced_agent, plot_results, calculate_portfolio_metrics
import torch
import yaml
import pandas as pd
import numpy as np
from datetime import datetime


def backtest_agent(checkpoint_path, n_episodes=30):
    """
    Backtest a trained agent
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        n_episodes: Number of backtest episodes
    """
    
    print(f"\n{'='*60}")
    print(f"BACKTESTING ENHANCED RL TRADING AGENT")
    print(f"{'='*60}")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    print("\nInitializing environment...")
    env = EnhancedTradingEnvironmentV2(config)
    
    # Create and load agent
    print("Loading trained agent...")
    agent = AttentionTradingAgent(
        state_dim=env.state_dim,
        n_assets=env.n_assets,
        hidden_dim=config['agent']['hidden_dim']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Training episodes completed: {checkpoint.get('episode', 'Unknown')}")
    
    # Run backtest
    print(f"\nRunning backtest for {n_episodes} episodes...")
    eval_results = evaluate_enhanced_agent(
        env=env,
        agent=agent,
        n_episodes=n_episodes,
        epsilon=0.01  # Very low exploration for backtest
    )
    
    # Calculate aggregate metrics
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    # Performance metrics
    all_returns = eval_results['net_return'].values / 100  # Convert to decimal
    metrics = calculate_portfolio_metrics(all_returns)
    
    print("\nPERFORMANCE METRICS:")
    print(f"  Average Return: {eval_results['net_return'].mean():.2f}% ± {eval_results['net_return'].std():.2f}%")
    print(f"  Median Return: {eval_results['net_return'].median():.2f}%")
    print(f"  Best Episode: {eval_results['net_return'].max():.2f}%")
    print(f"  Worst Episode: {eval_results['net_return'].min():.2f}%")
    print(f"  Success Rate: {(eval_results['net_return'] > 0).mean():.1%}")
    
    print("\nRISK METRICS:")
    print(f"  Sharpe Ratio: {eval_results['sharpe_ratio'].mean():.3f}")
    print(f"  Sortino Ratio: {eval_results['sortino_ratio'].mean():.3f}")
    print(f"  Maximum Drawdown: {eval_results['max_drawdown'].mean():.2f}%")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    print(f"  95% VaR: {metrics['var_95']*100:.2f}%")
    print(f"  95% CVaR: {metrics['cvar_95']*100:.2f}%")
    
    print("\nTRADING STATISTICS:")
    print(f"  Win Rate: {eval_results['win_rate'].mean():.1%} ± {eval_results['win_rate'].std():.1%}")
    print(f"  Average Trades per Episode: {eval_results['trades'].mean():.1f}")
    print(f"  Total Trades: {eval_results['trades'].sum()}")
    print(f"  Average Confidence: {eval_results['avg_confidence'].mean():.3f}")
    
    # Statistical tests
    print("\nSTATISTICAL ANALYSIS:")
    
    # T-test against zero return
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(eval_results['net_return'], 0)
    print(f"  T-test vs zero return: t={t_stat:.3f}, p={p_value:.4f}")
    
    # Test for normality
    _, norm_p_value = stats.normaltest(eval_results['net_return'])
    print(f"  Normality test p-value: {norm_p_value:.4f}")
    
    # Information ratio (assuming 0% benchmark)
    information_ratio = eval_results['net_return'].mean() / eval_results['net_return'].std()
    print(f"  Information Ratio: {information_ratio:.3f}")
    
    # Save detailed results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"{results_dir}/backtest_results_{timestamp}.csv"
    eval_results.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Generate plots
    plot_file = f"{results_dir}/backtest_plots_{timestamp}.png"
    plot_results(eval_results, save_path=plot_file)
    
    # Generate trade log
    print("\nGenerating sample trade log...")
    generate_trade_log(env, agent, save_path=f"{results_dir}/trade_log_{timestamp}.csv")
    
    return eval_results


def generate_trade_log(env, agent, save_path=None):
    """Generate detailed trade log for one episode"""
    
    state = env.reset()
    done = False
    
    trade_log = []
    step = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        actions, confidences, value = agent.get_action(state_tensor, epsilon=0)
        
        # Log pre-trade state
        for i, symbol in enumerate(env.symbols):
            trade_log.append({
                'step': step,
                'symbol': symbol,
                'action': ['SELL', 'HOLD', 'BUY'][actions[i]],
                'confidence': confidences[i],
                'position': env.positions[symbol],
                'price': env.data[symbol]['close'][env.current_step],
                'cash': env.cash,
                'portfolio_value': env._calculate_portfolio_value(),
                'value_estimate': value.item()
            })
        
        state, reward, done, info = env.step(actions, confidences)
        step += 1
    
    # Save trade log
    trade_df = pd.DataFrame(trade_log)
    if save_path:
        trade_df.to_csv(save_path, index=False)
        print(f"Trade log saved to: {save_path}")
    
    # Print summary
    print("\nTRADE LOG SUMMARY:")
    print(f"  Total steps: {step}")
    print(f"  Final return: {info['net_return']:.2f}%")
    print(f"  Total trades executed: {info['trades']}")
    
    # Action distribution
    action_counts = trade_df['action'].value_counts()
    print(f"\n  Action distribution:")
    for action, count in action_counts.items():
        print(f"    {action}: {count} ({count/len(trade_df)*100:.1f}%)")
    
    return trade_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest trained RL trading agent')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=30,
        help='Number of backtest episodes'
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train a model first using train_agent.py")
        sys.exit(1)
    
    # Run backtest
    results = backtest_agent(args.checkpoint, args.episodes)