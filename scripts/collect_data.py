"""
Data Collection Script

Collects random rollouts using Brownian motion policy for training VAE and MDN-RNN.
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.envs import make_env, preprocess_obs, BrownianPolicy


def collect_episode(args):
    """
    Collect a single episode.
    
    Args:
        args: Tuple of (episode_idx, max_steps, img_size, seed)
        
    Returns:
        Dictionary with observations, actions, rewards
    """
    episode_idx, max_steps, img_size, seed = args
    
    # Set seed for this episode
    np.random.seed(seed + episode_idx)
    
    # Create environment and policy
    env = make_env(render_mode=None)
    policy = BrownianPolicy(action_dim=3, noise_scale=0.1)
    
    observations = []
    actions = []
    rewards = []
    
    obs, _ = env.reset(seed=seed + episode_idx)
    policy.reset()
    
    for step in range(max_steps):
        # Preprocess observation
        obs_processed = preprocess_obs(obs, img_size)
        observations.append(obs_processed)
        
        # Sample action
        action = policy.sample()
        actions.append(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    env.close()
    
    return {
        'observations': np.array(observations, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'rewards': np.array(rewards, dtype=np.float32),
        'length': len(observations)
    }


def main():
    parser = argparse.ArgumentParser(description="Collect training data for World Models")
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of episodes to collect (default: 500)')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Max steps per episode (default: 1000)')
    parser.add_argument('--img-size', type=int, default=64,
                       help='Image size to resize to (default: 64)')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--output', type=str, default='data/episodes.npz',
                       help='Output file path (default: data/episodes.npz)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("World Models - Data Collection")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Workers: {args.threads}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Prepare arguments for parallel collection
    collection_args = [
        (i, args.max_steps, args.img_size, args.seed)
        for i in range(args.episodes)
    ]
    
    # Collect episodes in parallel
    start_time = time.time()
    
    print(f"\nCollecting {args.episodes} episodes using {args.threads} workers...")
    
    if args.threads > 1:
        with Pool(args.threads) as pool:
            episodes = list(tqdm(
                pool.imap(collect_episode, collection_args),
                total=args.episodes,
                desc="Collecting"
            ))
    else:
        episodes = []
        for arg in tqdm(collection_args, desc="Collecting"):
            episodes.append(collect_episode(arg))
    
    elapsed = time.time() - start_time
    
    # Compute statistics
    total_frames = sum(ep['length'] for ep in episodes)
    total_rewards = [np.sum(ep['rewards']) for ep in episodes]
    
    print(f"\nCollection complete in {elapsed:.1f} seconds!")
    print(f"Total frames: {total_frames:,}")
    print(f"Average episode length: {total_frames / args.episodes:.1f}")
    print(f"Average episode reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    
    # Save data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Concatenate all observations and actions
    all_observations = np.concatenate([ep['observations'] for ep in episodes], axis=0)
    all_actions = np.concatenate([ep['actions'] for ep in episodes], axis=0)
    all_rewards = np.concatenate([ep['rewards'] for ep in episodes], axis=0)
    episode_lengths = np.array([ep['length'] for ep in episodes])
    
    # Save
    np.savez_compressed(
        output_path,
        observations=all_observations,
        actions=all_actions,
        rewards=all_rewards,
        episode_lengths=episode_lengths
    )
    
    # Report file size
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {output_path}")
    print(f"File size: {file_size:.1f} MB")
    print(f"Observations shape: {all_observations.shape}")
    print(f"Actions shape: {all_actions.shape}")
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("Next step: python -m scripts.train_vae --epochs 5")
    print("=" * 60)


if __name__ == "__main__":
    main()
