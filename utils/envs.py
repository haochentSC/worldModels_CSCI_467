"""
Environment wrappers and utilities for CarRacing.
"""

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional
import sys


def make_env(render_mode: Optional[str] = None) -> gym.Env:
    """
    Create CarRacing environment with proper settings.
    
    Args:
        render_mode: 'human' for visualization, 'rgb_array' for recording, None for training
        
    Returns:
        Gymnasium environment
    """
    env = gym.make('CarRacing-v3', render_mode=render_mode)
    return env


def preprocess_obs(obs: np.ndarray, img_size: int = 64) -> np.ndarray:
    """
    Preprocess observation for VAE.
    
    Args:
        obs: Raw observation from env, shape (96, 96, 3), values [0, 255]
        img_size: Target size (default 64x64)
        
    Returns:
        Preprocessed image, shape (3, img_size, img_size), values [0, 1]
    """
    # Resize using PIL (higher quality than cv2)
    img = Image.fromarray(obs)
    img = img.resize((img_size, img_size), Image.BILINEAR)
    
    # Convert to numpy and normalize
    img = np.array(img, dtype=np.float32) / 255.0
    
    # CHW format for PyTorch
    img = img.transpose(2, 0, 1)
    
    return img


def preprocess_obs_batch(obs_batch: np.ndarray, img_size: int = 64) -> np.ndarray:
    """
    Preprocess a batch of observations.
    
    Args:
        obs_batch: Batch of observations, shape (batch, 96, 96, 3)
        img_size: Target size
        
    Returns:
        Preprocessed batch, shape (batch, 3, img_size, img_size)
    """
    batch_size = obs_batch.shape[0]
    result = np.zeros((batch_size, 3, img_size, img_size), dtype=np.float32)
    
    for i in range(batch_size):
        result[i] = preprocess_obs(obs_batch[i], img_size)
    
    return result


class BrownianPolicy:
    """
    Brownian motion policy for data collection.
    
    Generates smoother, more realistic trajectories than random sampling.
    Actions change gradually rather than jumping randomly.
    """
    
    def __init__(self, action_dim: int = 3, noise_scale: float = 0.1):
        self.action_dim = action_dim
        self.noise_scale = noise_scale
        self.current_action = np.zeros(action_dim)
        
    def reset(self):
        """Reset action to zeros."""
        self.current_action = np.zeros(self.action_dim)
        
    def sample(self) -> np.ndarray:
        """
        Sample next action using Brownian motion.
        
        Returns:
            Action array [steering, gas, brake]
        """
        # Add noise to current action
        noise = np.random.randn(self.action_dim) * self.noise_scale
        self.current_action = self.current_action + noise
        
        # Clip to valid ranges
        action = self.current_action.copy()
        action[0] = np.clip(action[0], -1, 1)  # steering
        action[1] = np.clip(action[1], 0, 1)   # gas
        action[2] = np.clip(action[2], 0, 1)   # brake
        
        return action


class WorldModelEnv(gym.Env):
    """
    Wrapper that exposes [z, h] as observations for controller training.
    
    This allows using standard RL algorithms (like PPO) to train the controller
    on the world model's latent representations.
    """
    
    def __init__(self, 
                 vae: torch.nn.Module,
                 mdrnn: Optional[torch.nn.Module] = None,
                 device: str = "cuda",
                 img_size: int = 64,
                 use_rnn: bool = True):
        super().__init__()
        
        self.vae = vae
        self.mdrnn = mdrnn
        self.device = torch.device(device)
        self.img_size = img_size
        self.use_rnn = use_rnn and mdrnn is not None
        
        # Create underlying environment
        self.env = make_env(render_mode=None)
        
        # Observation space: z (32) + h (256) = 288, or just z (32)
        if self.use_rnn:
            obs_dim = 32 + 256  # z + h
        else:
            obs_dim = 32  # just z
            
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space (same as CarRacing)
        self.action_space = self.env.action_space
        
        # State
        self.hidden = None
        self.current_z = None
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Encode observation
        self.current_z = self._encode(obs)
        
        # Reset RNN hidden state
        if self.use_rnn:
            self.hidden = self.mdrnn.get_initial_hidden(1, self.device)
            h = self.hidden[0].squeeze(0).squeeze(0)  # (hidden_dim,)
            latent_obs = torch.cat([self.current_z.squeeze(0), h], dim=-1)
        else:
            latent_obs = self.current_z.squeeze(0)
            
        return latent_obs.cpu().numpy(), info
    
    def step(self, action):
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update RNN hidden state
        if self.use_rnn:
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, _, _, self.hidden = self.mdrnn.forward_single(
                self.current_z, action_tensor, self.hidden
            )
        
        # Encode new observation
        self.current_z = self._encode(obs)
        
        # Construct latent observation
        if self.use_rnn:
            h = self.hidden[0].squeeze(0).squeeze(0)
            latent_obs = torch.cat([self.current_z.squeeze(0), h], dim=-1)
        else:
            latent_obs = self.current_z.squeeze(0)
            
        return latent_obs.cpu().numpy(), reward, terminated, truncated, info
    
    def _encode(self, obs: np.ndarray) -> torch.Tensor:
        """Encode observation to latent vector."""
        # Preprocess
        obs_processed = preprocess_obs(obs, self.img_size)
        obs_tensor = torch.tensor(obs_processed, dtype=torch.float32, device=self.device)
        obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dim
        
        # Encode
        with torch.no_grad():
            z = self.vae.encode(obs_tensor)
            
        return z
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()


def evaluate_policy(env: gym.Env,
                   vae: torch.nn.Module,
                   mdrnn: Optional[torch.nn.Module],
                   controller: torch.nn.Module,
                   device: str = "cuda",
                   n_episodes: int = 10,
                   max_steps: int = 1000,
                   render: bool = False) -> Tuple[float, float]:
    """
    Evaluate a trained controller.
    
    Args:
        env: Gymnasium environment
        vae: Trained VAE
        mdrnn: Trained MDN-RNN (or None)
        controller: Trained controller
        device: Device to use
        n_episodes: Number of episodes to evaluate
        max_steps: Max steps per episode
        render: Whether to render
        
    Returns:
        mean_reward: Average episode reward
        std_reward: Standard deviation of rewards
    """
    device = torch.device(device)
    vae.eval()
    controller.eval()
    if mdrnn is not None:
        mdrnn.eval()
    
    rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        # Initialize hidden state
        if mdrnn is not None:
            hidden = mdrnn.get_initial_hidden(1, device)
        
        for step in range(max_steps):
            if render:
                env.render()
            
            # Preprocess and encode
            obs_processed = preprocess_obs(obs)
            obs_tensor = torch.tensor(obs_processed, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                z = vae.encode(obs_tensor)
                
                # Get hidden state
                if mdrnn is not None:
                    h = hidden[0].squeeze(0)  # (1, hidden_dim)
                else:
                    h = torch.zeros(1, 256, device=device)
                
                # Get action
                action = controller.get_action(z, h)
                action_np = action.cpu().numpy()[0]
                
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            episode_reward += reward
            
            # Update RNN
            if mdrnn is not None:
                action_tensor = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
                _, _, _, hidden = mdrnn.forward_single(z, action_tensor, hidden)
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        print(f"Episode {ep+1}/{n_episodes}: {episode_reward:.1f}")
    
    return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":
    # Test environment creation
    print("Testing environment creation...")
    env = make_env(render_mode=None)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test preprocessing
    print("\nTesting preprocessing...")
    processed = preprocess_obs(obs)
    print(f"Preprocessed shape: {processed.shape}")
    print(f"Value range: [{processed.min():.2f}, {processed.max():.2f}]")
    
    # Test Brownian policy
    print("\nTesting Brownian policy...")
    policy = BrownianPolicy()
    actions = [policy.sample() for _ in range(5)]
    for i, a in enumerate(actions):
        print(f"  Action {i}: {a}")
    
    # Quick rollout test
    print("\nTesting rollout...")
    policy.reset()
    total_reward = 0
    for _ in range(100):
        action = policy.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"100-step reward: {total_reward:.1f}")
    
    env.close()
    print("\nAll tests passed!")
