"""
Attention-based Trading Agent with Confidence Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class AttentionTradingAgent(nn.Module):
    """
    Advanced agent with attention mechanism and confidence estimation
    """
    
    def __init__(self, state_dim, n_assets=5, hidden_dim=512):
        super().__init__()
        self.n_assets = n_assets
        self.hidden_dim = hidden_dim
        
        # Input processing
        self.input_norm = nn.LayerNorm(state_dim)
        
        # Shared encoder with self-attention
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Asset-specific processing
        self.asset_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU()
            ) for _ in range(n_assets)
        ])
        
        # Action heads (one per asset)
        self.action_heads = nn.ModuleList([
            nn.Linear(128, 3) for _ in range(n_assets)
        ])
        
        # Confidence heads (one per asset)
        self.confidence_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(n_assets)
        ])
        
        # Value estimation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state: Market state tensor [batch_size, state_dim]
            
        Returns:
            action_logits: List of action logits for each asset
            confidences: List of confidence scores for each asset
            value: Estimated state value
        """
        # Normalize and encode
        x = self.input_norm(state)
        encoded = self.encoder(x)
        
        # Self-attention
        if encoded.dim() == 1:
            encoded = encoded.unsqueeze(0).unsqueeze(0)
        elif encoded.dim() == 2:
            encoded = encoded.unsqueeze(1)
        
        attended, _ = self.attention(encoded, encoded, encoded)
        attended = attended.squeeze(1)
        
        # Process each asset
        action_logits = []
        confidences = []
        
        for i in range(self.n_assets):
            asset_features = self.asset_processors[i](attended)
            
            # Action logits
            logits = self.action_heads[i](asset_features)
            action_logits.append(logits)
            
            # Confidence (sigmoid for 0-1 range)
            conf = torch.sigmoid(self.confidence_heads[i](asset_features))
            confidences.append(conf)
        
        # Value estimation
        value = self.value_head(attended.squeeze(0) if attended.dim() > 2 else attended)
        
        return action_logits, confidences, value
    
    def get_action(self, state, epsilon=0.05):
        """
        Get actions with confidence-weighted exploration
        
        Args:
            state: Current market state
            epsilon: Base exploration rate
            
        Returns:
            actions: List of actions for each asset
            confidence_values: List of confidence scores
            value: Estimated state value
        """
        with torch.no_grad():
            action_logits, confidences, value = self.forward(state)
            
            actions = []
            confidence_values = []
            
            for i in range(self.n_assets):
                conf = confidences[i].item()
                confidence_values.append(conf)
                
                # Confidence-weighted exploration
                if np.random.random() < epsilon * (1 - conf):
                    action = np.random.randint(0, 3)
                else:
                    # Temperature based on confidence
                    temperature = 1.0 - conf * 0.5  # Higher confidence = lower temperature
                    
                    logits = action_logits[i]
                    probs = F.softmax(logits / temperature, dim=-1)
                    
                    if probs.dim() > 1:
                        probs = probs.squeeze(0)
                    
                    dist = Categorical(probs)
                    action = dist.sample().item()
                
                actions.append(action)
            
            return actions, confidence_values, value
    
    def get_log_probs(self, states, actions):
        """
        Calculate log probabilities for given state-action pairs
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            log_probs: Log probabilities for each action
            values: State values
            entropy: Policy entropy
        """
        action_logits, confidences, values = self.forward(states)
        
        log_probs = []
        entropy = 0
        
        for i in range(self.n_assets):
            logits = action_logits[i]
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            # Get log probability for the taken action
            if actions.dim() == 1:
                action_i = actions[i]
            else:
                action_i = actions[:, i]
            
            log_prob = dist.log_prob(action_i)
            log_probs.append(log_prob)
            
            # Calculate entropy
            entropy += dist.entropy().mean()
        
        # Stack log probs
        log_probs = torch.stack(log_probs, dim=1) if len(log_probs[0].shape) > 0 else torch.stack(log_probs)
        
        return log_probs, values, entropy / self.n_assets, confidences
    
    def evaluate_actions(self, states, actions):
        """
        Evaluate actions for PPO update
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            values: State values
            log_probs: Action log probabilities
            entropy: Policy entropy
            confidences: Confidence scores
        """
        log_probs, values, entropy, confidences = self.get_log_probs(states, actions)
        return values, log_probs, entropy, confidences