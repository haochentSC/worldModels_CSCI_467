"""
Controller (C) - The decision-making component

A simple linear controller that maps [z, h] to actions.
Deliberately minimal (867 parameters) to force the world model to learn useful representations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class Controller(nn.Module):
    """
    Linear controller: maps concatenated [z, h] to actions.
    
    Input: z (32-dim latent from VAE) + h (256-dim hidden from RNN) = 288-dim
    Output: 3-dim action (steering, gas, brake)
    
    Only 867 parameters! (288 * 3 + 3 bias)
    """
    
    def __init__(self, 
                 latent_dim: int = 32,
                 hidden_dim: int = 256,
                 action_dim: int = 3):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        input_dim = latent_dim + hidden_dim
        self.fc = nn.Linear(input_dim, action_dim)
        
    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Latent vector from VAE, shape (batch, latent_dim)
            h: Hidden state from RNN, shape (batch, hidden_dim)
            
        Returns:
            action: Raw action logits, shape (batch, action_dim)
        """
        x = torch.cat([z, h], dim=-1)
        return self.fc(x)
    
    def get_action(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Get action with proper activation functions for CarRacing.
        
        Returns:
            action: [steering, gas, brake] with appropriate ranges
                   steering: tanh -> [-1, 1]
                   gas: sigmoid -> [0, 1]
                   brake: sigmoid -> [0, 1]
        """
        raw = self.forward(z, h)
        
        steering = torch.tanh(raw[:, 0:1])
        gas = torch.sigmoid(raw[:, 1:2])
        brake = torch.sigmoid(raw[:, 2:3])
        
        return torch.cat([steering, gas, brake], dim=-1)
    
    def get_params(self) -> np.ndarray:
        """Get flattened parameters as numpy array (for CMA-ES)."""
        return torch.cat([p.data.view(-1) for p in self.parameters()]).cpu().numpy()
    
    def set_params(self, params: np.ndarray):
        """Set parameters from flattened numpy array (for CMA-ES)."""
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(torch.from_numpy(params[idx:idx+size]).view(p.shape))
            idx += size
    
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


class ControllerWithoutRNN(nn.Module):
    """
    Simplified controller that only uses VAE latent (no RNN hidden state).
    Useful for quick prototypes or when testing if RNN is necessary.
    
    Input: z (32-dim)
    Output: 3-dim action
    """
    
    def __init__(self, latent_dim: int = 32, action_dim: int = 3):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Slightly larger network to compensate for no RNN
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
    
    def get_action(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.forward(z)
        
        steering = torch.tanh(raw[:, 0:1])
        gas = torch.sigmoid(raw[:, 1:2])
        brake = torch.sigmoid(raw[:, 2:3])
        
        return torch.cat([steering, gas, brake], dim=-1)
    
    def get_params(self) -> np.ndarray:
        return torch.cat([p.data.view(-1) for p in self.parameters()]).cpu().numpy()
    
    def set_params(self, params: np.ndarray):
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(torch.from_numpy(params[idx:idx+size]).view(p.shape))
            idx += size
    
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test full controller
    controller = Controller(latent_dim=32, hidden_dim=256, action_dim=3).to(device)
    print(f"\nFull Controller:")
    print(f"  Parameters: {controller.num_params()}")
    
    z = torch.randn(4, 32).to(device)
    h = torch.randn(4, 256).to(device)
    
    action = controller.get_action(z, h)
    print(f"  Input z shape: {z.shape}")
    print(f"  Input h shape: {h.shape}")
    print(f"  Output action shape: {action.shape}")
    print(f"  Action sample: {action[0].detach().cpu().numpy()}")
    
    # Test controller without RNN
    controller_simple = ControllerWithoutRNN(latent_dim=32, action_dim=3).to(device)
    print(f"\nController without RNN:")
    print(f"  Parameters: {controller_simple.num_params()}")
    
    action = controller_simple.get_action(z)
    print(f"  Output action shape: {action.shape}")
    
    # Test param get/set (for CMA-ES)
    params = controller.get_params()
    print(f"\nParam vector length: {len(params)}")
    controller.set_params(params)
    print("Param get/set test passed!")
