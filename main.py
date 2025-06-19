"""
Enhanced RL Trading System - Main Entry Point
"""

import argparse
import yaml
import torch
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

from src.environment import EnhancedTradingEnvironmentV2
from src.agent import AttentionTradingAgent
from src.trainer import ImprovedPPOTrainer
from src.utils import set_seeds, setup_logger, evaluate_enhanced_agent, plot_results


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training and evaluation pipeline"""
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger(
        name='rl_trading',
        log_file=f"{config['logging']['log_dir']}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger.info("Enhanced RL Trading System V2 - Starting")
    
    # Set random seeds
    set_seeds(config['system']['seed'])
    
    # Device configuration
    if config['system']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['system']['device'])
    logger.info(f"Using device: {device}")
    
    # Create environment
    logger.info("Initializing trading environment...")
    env = EnhancedTradingEnvironmentV2(config=config)
    logger.info(f"Environment initialized: {env.n_assets} assets, state_dim={env.state_dim}")
    
    # Create agent
    logger.info("Creating neural network agent...")
    agent = AttentionTradingAgent(
        state_dim=env.state_dim,
        n_assets=env.n_assets,
        hidden_dim=config['agent']['hidden_dim']
    ).to(device)
    
    num_params = sum(p.numel() for p in agent.parameters())
    logger.info(f"Agent created with {num_params:,} parameters")
    
    # Load checkpoint if specified
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        start_episode = checkpoint.get('episode', 0)
    else:
        start_episode = 0
    
    if args.mode == 'train':
        # Create trainer
        logger.info("Initializing PPO trainer...")
        trainer = ImprovedPPOTrainer(
            env=env,
            agent=agent,
            config=config['training'],
            device=device,
            logger=logger
        )
        
        # Resume training if checkpoint loaded
        if args.checkpoint and 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.episode_rewards = checkpoint.get('episode_rewards', [])
            trainer.episode_infos = checkpoint.get('episode_infos', [])
        
        # Train the agent
        logger.info(f"Starting training for {config['training']['n_episodes']} episodes...")
        trainer.train(
            n_episodes=config['training']['n_episodes'] - start_episode,
            start_episode=start_episode
        )
        
        # Save final model
        final_path = Path(config['logging']['checkpoint_dir']) / 'final_model.pt'
        trainer.save_checkpoint(str(final_path))
        logger.info(f"Final model saved to {final_path}")
        
    elif args.mode == 'evaluate':
        # Evaluation mode
        logger.info("Starting evaluation...")
        eval_results = evaluate_enhanced_agent(
            env=env,
            agent=agent,
            n_episodes=config['evaluation']['n_test_episodes'],
            device=device,
            epsilon=config['evaluation']['test_epsilon']
        )
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\nReturns:")
        print(f"  Average: {eval_results['net_return'].mean():.2f}% ± {eval_results['net_return'].std():.2f}%")
        print(f"  Best: {eval_results['net_return'].max():.2f}%")
        print(f"  Worst: {eval_results['net_return'].min():.2f}%")
        print(f"  Success Rate: {(eval_results['net_return'] > 0).mean():.1%}")
        
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio: {eval_results['sharpe_ratio'].mean():.2f}")
        print(f"  Sortino Ratio: {eval_results['sortino_ratio'].mean():.2f}")
        print(f"  Max Drawdown: {eval_results['max_drawdown'].mean():.2f}%")
        
        print(f"\nTrading Statistics:")
        print(f"  Win Rate: {eval_results['win_rate'].mean():.1%}")
        print(f"  Avg Trades: {eval_results['trades'].mean():.1f}")
        
        # Save results
        results_path = Path(config['logging']['results_dir']) / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        eval_results.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Generate plots
        if config['visualization']['save_plots']:
            plot_results(
                eval_results=eval_results,
                trainer=None,
                save_path=Path(config['logging']['results_dir']) / 'evaluation_plots.png'
            )
    
    elif args.mode == 'backtest':
        # Full backtesting mode
        logger.info("Running backtest...")
        # TODO: Implement backtesting with real market data
        raise NotImplementedError("Backtesting mode not yet implemented")
    
    logger.info("Process completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced RL Trading System')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'backtest'],
        default='train',
        help='Operation mode'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to resume from'
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    main(args)