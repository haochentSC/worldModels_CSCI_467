"""
Miscellaneous utility functions.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_checkpoint(model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer],
                   epoch: int,
                   loss: float,
                   path: str,
                   extra: Optional[Dict[str, Any]] = None):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        epoch: Current epoch
        loss: Current loss
        path: Path to save
        extra: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if extra is not None:
        checkpoint.update(extra)
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model: torch.nn.Module,
                   path: str,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: str = "cuda") -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        path: Path to checkpoint
        optimizer: Optimizer (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    
    return checkpoint


class EpisodeBuffer:
    """
    Buffer to store episode data for training.
    
    Stores observations, actions, and rewards from collected episodes.
    """
    
    def __init__(self, max_episodes: int = 10000):
        self.max_episodes = max_episodes
        self.episodes = []
        
    def add_episode(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """
        Add an episode to the buffer.
        
        Args:
            observations: Array of observations, shape (T, 3, 64, 64)
            actions: Array of actions, shape (T, 3)
            rewards: Array of rewards, shape (T,)
        """
        episode = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards
        }
        
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)
        
        self.episodes.append(episode)
        
    def save(self, path: str):
        """Save buffer to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, episodes=self.episodes)
        print(f"Saved {len(self.episodes)} episodes to {path}")
        
    def load(self, path: str):
        """Load buffer from disk."""
        data = np.load(path, allow_pickle=True)
        self.episodes = list(data['episodes'])
        print(f"Loaded {len(self.episodes)} episodes from {path}")
        
    def __len__(self):
        return len(self.episodes)
    
    def get_all_observations(self) -> np.ndarray:
        """Get all observations concatenated."""
        return np.concatenate([ep['observations'] for ep in self.episodes], axis=0)
    
    def sample_sequences(self, 
                        batch_size: int, 
                        seq_len: int) -> tuple:
        """
        Sample random sequences for RNN training.
        
        Args:
            batch_size: Number of sequences to sample
            seq_len: Length of each sequence
            
        Returns:
            observations: (batch_size, seq_len, 3, 64, 64)
            actions: (batch_size, seq_len, 3)
        """
        observations = []
        actions = []
        
        for _ in range(batch_size):
            # Sample random episode
            ep_idx = np.random.randint(len(self.episodes))
            ep = self.episodes[ep_idx]
            
            # Sample random starting point
            max_start = len(ep['observations']) - seq_len
            if max_start <= 0:
                start = 0
                actual_len = len(ep['observations'])
            else:
                start = np.random.randint(max_start)
                actual_len = seq_len
            
            obs_seq = ep['observations'][start:start+actual_len]
            act_seq = ep['actions'][start:start+actual_len]
            
            # Pad if necessary
            if actual_len < seq_len:
                pad_len = seq_len - actual_len
                obs_seq = np.pad(obs_seq, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
                act_seq = np.pad(act_seq, ((0, pad_len), (0, 0)))
            
            observations.append(obs_seq)
            actions.append(act_seq)
        
        return np.array(observations), np.array(actions)


class Logger:
    """Simple logger for training metrics."""
    
    def __init__(self, log_dir: str, name: str = "training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.metrics = []
        
    def log(self, step: int, **kwargs):
        """Log metrics for a step."""
        entry = {'step': step, **kwargs}
        self.metrics.append(entry)
        
        # Save periodically
        if len(self.metrics) % 100 == 0:
            self.save()
    
    def save(self):
        """Save metrics to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_metrics(self) -> list:
        """Get all logged metrics."""
        return self.metrics


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, name: str = "Model"):
    """Print a summary of the model architecture."""
    print(f"\n{name} Summary:")
    print("-" * 50)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            params = param.numel()
            total_params += params
            print(f"  {name}: {list(param.shape)} = {params:,}")
    print("-" * 50)
    print(f"Total trainable parameters: {total_params:,}")
    print()


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test seed setting
    set_seed(42)
    print("Seed set to 42")
    
    # Test device detection
    device = get_device()
    
    # Test episode buffer
    print("\nTesting EpisodeBuffer...")
    buffer = EpisodeBuffer()
    
    # Add dummy episodes
    for i in range(5):
        obs = np.random.randn(100, 3, 64, 64).astype(np.float32)
        actions = np.random.randn(100, 3).astype(np.float32)
        rewards = np.random.randn(100).astype(np.float32)
        buffer.add_episode(obs, actions, rewards)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Test sequence sampling
    obs_batch, act_batch = buffer.sample_sequences(4, 32)
    print(f"Sampled observations shape: {obs_batch.shape}")
    print(f"Sampled actions shape: {act_batch.shape}")
    
    print("\nAll utility tests passed!")
