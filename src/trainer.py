"""
Improved PPO Trainer with Adaptive Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import logging
from pathlib import Path


class ImprovedPPOTrainer:
    """
    Enhanced PPO with adaptive learning and better stability
    """
    
    def __init__(self, env, agent, config=None, device=None, logger=None):
        self.env = env
        self.agent = agent
        
        # Default configuration
        if config is None:
            config = {
                'lr': 5e-5,
                'lr_min': 1e-6,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'clip_decay': 0.999,
                'value_coef': 0.5,
                'entropy_coef': 0.05,
                'entropy_min': 0.001,
                'max_grad_norm': 0.5,
                'batch_size': 128,
                'n_epochs': 10,
                'n_steps': 1024,
                'target_kl': 0.02
            }
        self.config = config
        
        # Device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent.to(self.device)
        
        # Logger
        self.logger = logger if logger else logging.getLogger(__name__)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            agent.parameters(), 
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.0001)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2
        )
        
        # Tracking
        self.episode_rewards = []
        self.episode_infos = []
        self.training_metrics = []
        
    def train(self, n_episodes=1500, start_episode=0):
        """
        Train with improved stability
        
        Args:
            n_episodes: Number of episodes to train
            start_episode: Starting episode number (for resuming)
        """
        self.logger.info(f"Training Enhanced Agent V2 for {n_episodes} episodes...")
        
        best_sharpe = -np.inf
        
        for episode in range(start_episode, start_episode + n_episodes):
            # Collect experience
            trajectory = self.collect_trajectory()
            
            # Update policy
            if len(trajectory['states']) >= self.config['batch_size']:
                metrics = self.update_policy(trajectory)
                self.training_metrics.append(metrics)
            
            # Adaptive parameters
            self._adapt_parameters(episode)
            
            # Logging and checkpointing
            if episode % 50 == 0:
                avg_sharpe = self.log_progress(episode, start_episode + n_episodes)
                
                # Save best model
                if avg_sharpe > best_sharpe:
                    best_sharpe = avg_sharpe
                    self.save_checkpoint(f'checkpoints/best_model_ep{episode}.pt')
            
            # Regular checkpointing
            if episode % 100 == 0 and episode > 0:
                self.save_checkpoint(f'checkpoints/checkpoint_ep{episode}.pt')
    
    def collect_trajectory(self):
        """
        Collect trajectory with enhanced features
        
        Returns:
            trajectory: Dictionary containing collected experience
        """
        states, actions, rewards = [], [], []
        values, log_probs, confidences = [], [], []
        
        state = self.env.reset()
        episode_reward = 0
        
        for step in range(self.config['n_steps']):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action with confidence
            with torch.no_grad():
                action_list, confidence_list, value = self.agent.get_action(
                    state_tensor, 
                    epsilon=0.1 * (0.99 ** len(self.episode_rewards))
                )
                
                # Calculate log probabilities
                action_logits, _, _ = self.agent(state_tensor)
                
                log_prob_list = []
                for i, a in enumerate(action_list):
                    logits = action_logits[i]
                    probs = F.softmax(logits, dim=-1)
                    dist = Categorical(probs)
                    log_prob = dist.log_prob(torch.tensor(a))
                    log_prob_list.append(log_prob.item())
            
            # Environment step
            next_state, reward, done, info = self.env.step(action_list, confidence_list)
            
            # Store experience
            states.append(state)
            actions.append(action_list)
            rewards.append(reward)
            values.append(value.cpu().item())
            log_probs.append(log_prob_list)
            confidences.append(confidence_list)
            
            episode_reward += reward
            state = next_state
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_infos.append(info)
                break
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'values': np.array(values),
            'log_probs': np.array(log_probs),
            'confidences': np.array(confidences)
        }
    
    def update_policy(self, trajectory):
        """
        Update with improved PPO
        
        Args:
            trajectory: Collected experience
            
        Returns:
            metrics: Training metrics
        """
        # Convert to tensors
        states = torch.FloatTensor(trajectory['states']).to(self.device)
        actions = torch.LongTensor(trajectory['actions']).to(self.device)
        rewards = torch.FloatTensor(trajectory['rewards']).to(self.device)
        old_values = torch.FloatTensor(trajectory['values']).to(self.device)
        old_log_probs = torch.FloatTensor(trajectory['log_probs']).to(self.device)
        confidences = torch.FloatTensor(trajectory['confidences']).to(self.device)
        
        # Calculate advantages with GAE
        advantages = self.calculate_gae(rewards, old_values)
        returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for epoch in range(self.config['n_epochs']):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.config['batch_size']):
                end = min(start + self.config['batch_size'], len(states))
                batch_idx = indices[start:end]
                
                # Get batch
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_confidences = confidences[batch_idx]
                
                # Forward pass
                action_logits, new_confidences, values = self.agent(batch_states)
                
                # Calculate new log probs
                log_probs = []
                entropy = 0
                
                for i in range(self.agent.n_assets):
                    logits = action_logits[i]
                    probs = F.softmax(logits, dim=-1)
                    dist = Categorical(probs)
                    
                    log_prob = dist.log_prob(batch_actions[:, i])
                    log_probs.append(log_prob)
                    
                    # Weighted entropy by confidence
                    asset_entropy = dist.entropy()
                    weighted_entropy = asset_entropy * (1 - batch_confidences[:, i])
                    entropy += weighted_entropy.mean()
                
                log_probs = torch.stack(log_probs, dim=1)
                
                # PPO loss with confidence weighting
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Weight advantages by confidence
                weighted_advantages = batch_advantages.unsqueeze(1) * batch_confidences
                
                surr1 = ratio * weighted_advantages
                surr2 = torch.clamp(ratio, 
                                   1 - self.config['clip_epsilon'],
                                   1 + self.config['clip_epsilon']) * weighted_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.smooth_l1_loss(values.squeeze(), batch_returns)
                
                # Confidence loss (encourage high confidence when correct)
                correct_actions = (batch_advantages > 0).float()
                confidence_targets = correct_actions.unsqueeze(1).expand_as(batch_confidences)
                confidence_loss = F.binary_cross_entropy(
                    torch.cat(new_confidences, dim=1).squeeze(),
                    confidence_targets.mean(dim=1)
                )
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.config['value_coef'] * value_loss + 
                    0.1 * confidence_loss -
                    self.config['entropy_coef'] * entropy
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), 
                    self.config['max_grad_norm']
                )
                
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
                
                # Early stopping if KL divergence too high
                with torch.no_grad():
                    kl = (batch_old_log_probs - log_probs).mean()
                    if kl > self.config['target_kl']:
                        return {
                            'policy_loss': total_policy_loss / n_updates,
                            'value_loss': total_value_loss / n_updates,
                            'entropy': total_entropy / n_updates,
                            'kl': kl.item()
                        }
        
        # Step scheduler
        self.scheduler.step()
        
        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'kl': 0
        }
    
    def calculate_gae(self, rewards, values):
        """
        Generalized Advantage Estimation
        
        Args:
            rewards: Episode rewards
            values: Value estimates
            
        Returns:
            advantages: Calculated advantages
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config['gamma'] * next_value - values[t]
            advantages[t] = delta + self.config['gamma'] * self.config['gae_lambda'] * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def _adapt_parameters(self, episode):
        """
        Adaptive parameter adjustment
        
        Args:
            episode: Current episode number
        """
        # Decay exploration
        self.config['clip_epsilon'] = max(0.1, self.config['clip_epsilon'] * self.config['clip_decay'])
        
        # Decay entropy coefficient
        self.config['entropy_coef'] = max(
            self.config['entropy_min'],
            self.config['entropy_coef'] * 0.999
        )
        
        # Adjust learning rate based on recent performance
        if len(self.episode_infos) > 50:
            recent_sharpes = [info['sharpe_ratio'] for info in self.episode_infos[-50:]]
            avg_sharpe = np.mean(recent_sharpes)
            
            # If performance plateaus, reduce learning rate
            if episode > 500 and avg_sharpe < 1.0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(
                        self.config['lr_min'],
                        param_group['lr'] * 0.95
                    )
    
    def log_progress(self, episode, total_episodes):
        """
        Enhanced progress logging
        
        Args:
            episode: Current episode
            total_episodes: Total episodes to train
            
        Returns:
            avg_sharpe: Average Sharpe ratio
        """
        if len(self.episode_rewards) < 10:
            return 0
        
        recent_rewards = self.episode_rewards[-50:] if len(self.episode_rewards) >= 50 else self.episode_rewards
        recent_infos = self.episode_infos[-50:] if len(self.episode_infos) >= 50 else self.episode_infos
        
        # Calculate metrics
        metrics = {
            'reward': np.mean(recent_rewards),
            'return': np.mean([info['net_return'] for info in recent_infos]),
            'sharpe': np.mean([info['sharpe_ratio'] for info in recent_infos]),
            'sortino': np.mean([info['sortino_ratio'] for info in recent_infos]),
            'max_dd': np.mean([info['max_drawdown'] for info in recent_infos]),
            'win_rate': np.mean([info['win_rate'] for info in recent_infos]),
            'trades': np.mean([info['trades'] for info in recent_infos])
        }
        
        self.logger.info(f"\nEpisode {episode}/{total_episodes}")
        self.logger.info(f"  Avg Reward: {metrics['reward']:.2f}")
        self.logger.info(f"  Avg Return: {metrics['return']:.2f}%")
        self.logger.info(f"  Avg Sharpe: {metrics['sharpe']:.2f}")
        self.logger.info(f"  Avg Sortino: {metrics['sortino']:.2f}")
        self.logger.info(f"  Avg Max DD: {metrics['max_dd']:.2f}%")
        self.logger.info(f"  Win Rate: {metrics['win_rate']:.1%}")
        self.logger.info(f"  Avg Trades: {metrics['trades']:.1f}")
        self.logger.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        if len(self.training_metrics) > 0:
            recent_metrics = self.training_metrics[-10:]
            self.logger.info(f"  Policy Loss: {np.mean([m['policy_loss'] for m in recent_metrics]):.4f}")
            self.logger.info(f"  Value Loss: {np.mean([m['value_loss'] for m in recent_metrics]):.4f}")
            self.logger.info(f"  Entropy: {np.mean([m['entropy'] for m in recent_metrics]):.4f}")
        
        return metrics['sharpe']
    
    def save_checkpoint(self, filepath):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'episode': len(self.episode_rewards),
            'episode_rewards': self.episode_rewards,
            'episode_infos': self.episode_infos,
            'training_metrics': self.training_metrics
        }, filepath)
        
        self.logger.info(f"Checkpoint saved to {filepath}")