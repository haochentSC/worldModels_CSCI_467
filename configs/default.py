"""
Configuration for World Models project.
All hyperparameters in one place for easy tuning.
"""

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Paths
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    
    # Environment
    env_name: str = "CarRacing-v3"
    img_size: int = 64  # Resize observations to 64x64
    action_dim: int = 3  # steering, gas, brake
    
    # VAE
    latent_dim: int = 32
    vae_lr: float = 1e-3
    vae_batch_size: int = 64
    vae_epochs: int = 10
    kl_weight: float = 1.0  # Beta for KL divergence term
    
    # MDN-RNN
    hidden_dim: int = 256
    n_gaussians: int = 5
    mdrnn_lr: float = 1e-3
    mdrnn_batch_size: int = 32
    mdrnn_epochs: int = 20
    seq_len: int = 32
    
    # Controller
    controller_input_dim: int = 288  # latent_dim + hidden_dim = 32 + 256
    
    # CMA-ES
    cma_pop_size: int = 64
    cma_n_samples: int = 16  # rollouts per individual
    cma_target_return: float = 950
    cma_generations: int = 500
    
    # PPO (alternative to CMA-ES)
    ppo_timesteps: int = 1000000
    ppo_lr: float = 3e-4
    
    # Data collection
    n_episodes: int = 2000  # Full training
    n_episodes_prototype: int = 500  # Quick prototype
    max_steps: int = 1000
    n_threads: int = 8
    
    # Hardware
    device: str = "cuda"  # or "cpu"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)


# Global config instance
config = Config()
