"""
Example script for training the RL trading agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import EnhancedTradingEnvironmentV2
from src.agent import AttentionTradingAgent
from src.trainer import ImprovedPPOTrainer
from src.utils import set_seeds, setup_logger
import yaml
import torch


def main():
    """Train the RL trading agent with default configuration"""
    
    # Set random seeds for reproducibility
    set_seeds(42)
    
    # Setup logging
    logger = setup_logger('training_example', 'logs/training_example.log')
    logger.info("Starting training example...")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    logger.info("Creating trading environment...")
    env = EnhancedTradingEnvironmentV2(config)
    
    # Create agent
    logger.info("Creating neural network agent...")
    agent = AttentionTradingAgent(
        state_dim=env.state_dim,
        n_assets=env.n_assets,
        hidden_dim=config['agent']['hidden_dim']
    )
    
    # Log model information
    num_params = sum(p.numel() for p in agent.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create trainer
    logger.info("Initializing PPO trainer...")
    trainer = ImprovedPPOTrainer(
        env=env,
        agent=agent,
        config=config['training'],
        logger=logger
    )
    
    # Train for a shorter period as example
    logger.info("Starting training for 100 episodes (example)...")
    trainer.train(n_episodes=100)
    
    # Save the trained model
    trainer.save_checkpoint('checkpoints/example_model.pt')
    logger.info("Model saved to checkpoints/example_model.pt")
    
    # Quick evaluation
    logger.info("Running quick evaluation...")
    state = env.reset()
    done = False
    total_return = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        actions, confidences, _ = agent.get_action(state_tensor, epsilon=0.05)
        state, reward, done, info = env.step(actions, confidences)
        total_return += reward
    
    logger.info(f"Example episode completed:")
    logger.info(f"  Total reward: {total_return:.2f}")
    logger.info(f"  Net return: {info['net_return']:.2f}%")
    logger.info(f"  Sharpe ratio: {info['sharpe_ratio']:.2f}")
    logger.info(f"  Win rate: {info['win_rate']:.1%}")
    
    print("\nTraining example completed successfully!")
    print(f"Check logs/training_example.log for detailed information")
    print(f"Model saved to checkpoints/example_model.pt")


if __name__ == "__main__":
    main()