"""
Mixture Density Network RNN (MDN-RNN) - Memory Model (M)

Predicts future latent states as a mixture of Gaussians.
Architecture: LSTM with MDN output layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MDRNN(nn.Module):
    """
    MDN-RNN: LSTM that predicts next latent state as mixture of Gaussians.
    
    Input: z_t (latent from VAE) + a_t (action)
    Output: Parameters of GMM for p(z_{t+1} | z_t, a_t, h_t)
    
    The MDN outputs:
        - pi: mixture weights (n_gaussians,)
        - mu: means for each Gaussian (n_gaussians, latent_dim)
        - sigma: std devs for each Gaussian (n_gaussians, latent_dim)
    """
    
    def __init__(self, 
                 latent_dim: int = 32,
                 action_dim: int = 3,
                 hidden_dim: int = 256,
                 n_gaussians: int = 5):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians
        
        # Input: z + a
        input_dim = latent_dim + action_dim
        
        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # MDN output heads
        # For each Gaussian: 1 weight + latent_dim means + latent_dim sigmas
        self.fc_pi = nn.Linear(hidden_dim, n_gaussians)
        self.fc_mu = nn.Linear(hidden_dim, n_gaussians * latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, n_gaussians * latent_dim)
        
    def forward(self, 
                z: torch.Tensor, 
                action: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            z: Latent vectors, shape (batch, seq_len, latent_dim)
            action: Actions, shape (batch, seq_len, action_dim)
            hidden: Optional initial hidden state (h, c)
            
        Returns:
            pi: Mixture weights, shape (batch, seq_len, n_gaussians)
            mu: Means, shape (batch, seq_len, n_gaussians, latent_dim)
            sigma: Std devs, shape (batch, seq_len, n_gaussians, latent_dim)
            hidden: Final hidden state (h, c)
        """
        batch_size, seq_len, _ = z.shape
        
        # Concatenate z and action
        x = torch.cat([z, action], dim=-1)
        
        # LSTM forward
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # MDN outputs
        pi = self.fc_pi(lstm_out)
        pi = F.softmax(pi, dim=-1)  # Normalize mixture weights
        
        mu = self.fc_mu(lstm_out)
        mu = mu.view(batch_size, seq_len, self.n_gaussians, self.latent_dim)
        
        sigma = self.fc_sigma(lstm_out)
        sigma = torch.exp(sigma)  # Ensure positive
        sigma = sigma.view(batch_size, seq_len, self.n_gaussians, self.latent_dim)
        
        return pi, mu, sigma, hidden
    
    def forward_single(self,
                       z: torch.Tensor,
                       action: torch.Tensor,
                       hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass for a single timestep (used during rollouts).
        
        Args:
            z: Latent vector, shape (batch, latent_dim)
            action: Action, shape (batch, action_dim)
            hidden: Hidden state
            
        Returns:
            pi, mu, sigma, hidden (all for single timestep)
        """
        # Add sequence dimension
        z = z.unsqueeze(1)
        action = action.unsqueeze(1)
        
        pi, mu, sigma, hidden = self.forward(z, action, hidden)
        
        # Remove sequence dimension
        pi = pi.squeeze(1)
        mu = mu.squeeze(1)
        sigma = sigma.squeeze(1)
        
        return pi, mu, sigma, hidden
    
    def get_initial_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get zero-initialized hidden state."""
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)
    
    def sample(self,
               pi: torch.Tensor,
               mu: torch.Tensor,
               sigma: torch.Tensor,
               temperature: float = 1.0) -> torch.Tensor:
        """
        Sample from the GMM.
        
        Args:
            pi: Mixture weights, shape (batch, n_gaussians)
            mu: Means, shape (batch, n_gaussians, latent_dim)
            sigma: Std devs, shape (batch, n_gaussians, latent_dim)
            temperature: Controls randomness (lower = more deterministic)
            
        Returns:
            Sampled z, shape (batch, latent_dim)
        """
        batch_size = pi.shape[0]
        
        # Apply temperature to mixture weights
        pi = pi / temperature
        pi = F.softmax(pi, dim=-1)
        
        # Sample which Gaussian to use
        mixture_idx = torch.multinomial(pi, 1).squeeze(-1)  # (batch,)
        
        # Get the corresponding mu and sigma
        batch_idx = torch.arange(batch_size, device=pi.device)
        selected_mu = mu[batch_idx, mixture_idx]  # (batch, latent_dim)
        selected_sigma = sigma[batch_idx, mixture_idx] * temperature
        
        # Sample from the selected Gaussian
        eps = torch.randn_like(selected_mu)
        z_sample = selected_mu + selected_sigma * eps
        
        return z_sample


def gmm_loss(z_target: torch.Tensor,
             pi: torch.Tensor,
             mu: torch.Tensor,
             sigma: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood of target under the GMM.
    
    Args:
        z_target: Target latent vectors, shape (batch, seq_len, latent_dim)
        pi: Mixture weights, shape (batch, seq_len, n_gaussians)
        mu: Means, shape (batch, seq_len, n_gaussians, latent_dim)
        sigma: Std devs, shape (batch, seq_len, n_gaussians, latent_dim)
        
    Returns:
        Negative log-likelihood loss
    """
    # Expand z_target for broadcasting: (batch, seq, 1, latent_dim)
    z_target = z_target.unsqueeze(2)
    
    # Compute log probability for each Gaussian
    # log N(z; mu, sigma) = -0.5 * (log(2*pi) + 2*log(sigma) + ((z-mu)/sigma)^2)
    log_prob = -0.5 * (
        math.log(2 * math.pi) + 
        2 * torch.log(sigma) + 
        ((z_target - mu) / sigma) ** 2
    )
    
    # Sum over latent dimensions
    log_prob = log_prob.sum(dim=-1)  # (batch, seq, n_gaussians)
    
    # Add log mixture weights
    log_pi = torch.log(pi + 1e-8)
    log_prob = log_prob + log_pi
    
    # Log-sum-exp to combine Gaussians
    log_prob = torch.logsumexp(log_prob, dim=-1)  # (batch, seq)
    
    # Negative log-likelihood
    nll = -log_prob.mean()
    
    return nll


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MDRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test sequence forward pass
    batch_size, seq_len = 4, 32
    z = torch.randn(batch_size, seq_len, 32).to(device)
    action = torch.randn(batch_size, seq_len, 3).to(device)
    
    pi, mu, sigma, hidden = model(z, action)
    
    print(f"\nSequence forward pass:")
    print(f"  Input z shape: {z.shape}")
    print(f"  Input action shape: {action.shape}")
    print(f"  Pi shape: {pi.shape}")
    print(f"  Mu shape: {mu.shape}")
    print(f"  Sigma shape: {sigma.shape}")
    
    # Test loss
    z_target = torch.randn(batch_size, seq_len, 32).to(device)
    loss = gmm_loss(z_target, pi, mu, sigma)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test single-step forward (for rollouts)
    z_single = torch.randn(batch_size, 32).to(device)
    action_single = torch.randn(batch_size, 3).to(device)
    hidden = model.get_initial_hidden(batch_size, device)
    
    pi, mu, sigma, hidden = model.forward_single(z_single, action_single, hidden)
    
    print(f"\nSingle-step forward pass:")
    print(f"  Pi shape: {pi.shape}")
    print(f"  Mu shape: {mu.shape}")
    
    # Test sampling
    z_sample = model.sample(pi, mu, sigma, temperature=1.0)
    print(f"  Sampled z shape: {z_sample.shape}")
