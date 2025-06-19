"""
Tests for the trading agent
"""

import pytest
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import AttentionTradingAgent


class TestAttentionTradingAgent:
    """Test suite for the trading agent"""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent"""
        state_dim = 136
        n_assets = 5
        hidden_dim = 128  # Smaller for testing
        return AttentionTradingAgent(state_dim, n_assets, hidden_dim)
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample state"""
        state_dim = 136
        return torch.randn(1, state_dim)
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent.n_assets == 5
        assert agent.hidden_dim == 128
        assert hasattr(agent, 'encoder')
        assert hasattr(agent, 'attention')
        assert hasattr(agent, 'value_head')
        assert len(agent.action_heads) == 5
        assert len(agent.confidence_heads) == 5
    
    def test_forward_pass(self, agent, sample_state):
        """Test forward pass through the network"""
        action_logits, confidences, value = agent(sample_state)
        
        assert len(action_logits) == agent.n_assets
        assert len(confidences) == agent.n_assets
        
        # Check shapes
        for i in range(agent.n_assets):
            assert action_logits[i].shape == (1, 3)  # 3 actions
            assert confidences[i].shape == (1, 1)
            assert 0 <= confidences[i].item() <= 1  # Sigmoid output
        
        assert value.shape == (1, 1)
    
    def test_get_action(self, agent, sample_state):
        """Test action selection"""
        actions, confidence_values, value = agent.get_action(sample_state, epsilon=0.1)
        
        assert len(actions) == agent.n_assets
        assert len(confidence_values) == agent.n_assets
        
        # Check action values
        for action in actions:
            assert action in [0, 1, 2]
        
        # Check confidence values
        for conf in confidence_values:
            assert 0 <= conf <= 1
    
    def test_deterministic_action(self, agent, sample_state):
        """Test deterministic action selection"""
        # Set epsilon to 0 for deterministic behavior
        actions1, _, _ = agent.get_action(sample_state, epsilon=0)
        actions2, _, _ = agent.get_action(sample_state, epsilon=0)
        
        # Should get same actions for same state
        assert actions1 == actions2
    
    def test_exploration(self, agent, sample_state):
        """Test that exploration works"""
        # Run multiple times with high epsilon
        action_sets = []
        for _ in range(10):
            actions, _, _ = agent.get_action(sample_state, epsilon=0.9)
            action_sets.append(tuple(actions))
        
        # Should see some variation in actions
        unique_action_sets = set(action_sets)
        assert len(unique_action_sets) > 1
    
    def test_batch_processing(self, agent):
        """Test processing batch of states"""
        batch_size = 16
        state_dim = 136
        batch_states = torch.randn(batch_size, state_dim)
        
        action_logits, confidences, values = agent(batch_states)
        
        # Check batch dimensions
        for i in range(agent.n_assets):
            assert action_logits[i].shape == (batch_size, 3)
            assert confidences[i].shape == (batch_size, 1)
        
        assert values.shape == (batch_size, 1)
    
    def test_gradient_flow(self, agent, sample_state):
        """Test that gradients flow through the network"""
        # Enable gradients
        agent.train()
        
        # Forward pass
        action_logits, confidences, value = agent(sample_state)
        
        # Create dummy loss
        loss = value.mean()
        for logits in action_logits:
            loss = loss + logits.mean()
        for conf in confidences:
            loss = loss + conf.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in agent.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_confidence_weighting(self, agent, sample_state):
        """Test confidence affects action selection"""
        # Get actions multiple times
        all_confidences = []
        action_variations = []
        
        for _ in range(20):
            actions, confidences, _ = agent.get_action(sample_state, epsilon=0.1)
            all_confidences.extend(confidences)
            action_variations.append(actions)
        
        # Check confidence values are diverse
        conf_array = np.array(all_confidences)
        assert conf_array.std() > 0.01  # Some variation in confidence
    
    def test_evaluate_actions(self, agent):
        """Test action evaluation for PPO"""
        batch_size = 8
        state_dim = 136
        
        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, 3, (batch_size, agent.n_assets))
        
        values, log_probs, entropy, confidences = agent.evaluate_actions(states, actions)
        
        assert values.shape == (batch_size, 1)
        assert log_probs.shape == (batch_size, agent.n_assets)
        assert isinstance(entropy, torch.Tensor)
        assert entropy.item() >= 0  # Entropy should be non-negative
    
    def test_model_save_load(self, agent, tmp_path):
        """Test model can be saved and loaded"""
        # Save model
        save_path = tmp_path / "test_model.pt"
        torch.save(agent.state_dict(), save_path)
        
        # Create new agent and load weights
        new_agent = AttentionTradingAgent(136, 5, 128)
        new_agent.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        state = torch.randn(1, 136)
        
        agent.eval()
        new_agent.eval()
        
        with torch.no_grad():
            actions1, conf1, value1 = agent.get_action(state, epsilon=0)
            actions2, conf2, value2 = new_agent.get_action(state, epsilon=0)
        
        assert actions1 == actions2
        assert np.allclose(conf1, conf2, atol=1e-6)
        assert torch.allclose(value1, value2, atol=1e-6)