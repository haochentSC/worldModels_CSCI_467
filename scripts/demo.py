"""
Demo Script - Watch the trained agent play CarRacing

Run after training to visualize the results.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import VAE
from models.mdrnn import MDRNN
from models.controller import Controller
from utils.envs import make_env, preprocess_obs
from utils.misc import get_device, load_checkpoint


def run_demo(vae, mdrnn, controller, device, n_episodes=5, render=True, use_rnn=True):
    """Run demo episodes with visualization."""
    
    render_mode = 'human' if render else None
    env = make_env(render_mode=render_mode)
    
    rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        # Initialize hidden state
        if use_rnn and mdrnn is not None:
            hidden = mdrnn.get_initial_hidden(1, device)
        else:
            hidden = None
        
        print(f"\nEpisode {ep + 1}/{n_episodes}")
        
        with torch.no_grad():
            for step in range(1000):
                # Encode observation
                obs_processed = preprocess_obs(obs)
                obs_tensor = torch.tensor(
                    obs_processed, dtype=torch.float32, device=device
                ).unsqueeze(0)
                
                z = vae.encode(obs_tensor)
                
                # Get hidden state
                if use_rnn and mdrnn is not None:
                    h = hidden[0].squeeze(0)
                else:
                    h = torch.zeros(1, 256, device=device)
                
                # Get action
                action = controller.get_action(z, h)
                action_np = action.cpu().numpy()[0]
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action_np)
                episode_reward += reward
                
                # Update hidden state
                if use_rnn and mdrnn is not None:
                    action_tensor = torch.tensor(
                        action_np, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    _, _, _, hidden = mdrnn.forward_single(z, action_tensor, hidden)
                
                if render:
                    time.sleep(0.01)  # Slow down for visualization
                
                if terminated or truncated:
                    break
        
        rewards.append(episode_reward)
        print(f"  Reward: {episode_reward:.1f}")
    
    env.close()
    
    return np.mean(rewards), np.std(rewards)


def main():
    parser = argparse.ArgumentParser(description="Demo trained World Models agent")
    parser.add_argument('--vae-checkpoint', type=str, default='checkpoints/vae_best.pt',
                       help='Path to trained VAE')
    parser.add_argument('--mdrnn-checkpoint', type=str, default=None,
                       help='Path to trained MDN-RNN (optional)')
    parser.add_argument('--controller-checkpoint', type=str, default='checkpoints/controller_cma_best.pt',
                       help='Path to trained controller')
    parser.add_argument('--ppo-model', type=str, default=None,
                       help='Path to PPO model (alternative to controller checkpoint)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run (default: 5)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='Latent dimension (default: 32)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='RNN hidden dimension (default: 256)')
    
    args = parser.parse_args()
    
    device = get_device()
    
    print("=" * 60)
    print("World Models - Demo")
    print("=" * 60)
    
    # Load VAE
    print("\nLoading VAE...")
    vae = VAE(latent_dim=args.latent_dim).to(device)
    load_checkpoint(vae, args.vae_checkpoint, device=device)
    vae.eval()
    
    # Load MDN-RNN (optional)
    mdrnn = None
    use_rnn = False
    
    if args.mdrnn_checkpoint and Path(args.mdrnn_checkpoint).exists():
        print("Loading MDN-RNN...")
        mdrnn = MDRNN(
            latent_dim=args.latent_dim,
            action_dim=3,
            hidden_dim=args.hidden_dim,
            n_gaussians=5
        ).to(device)
        load_checkpoint(mdrnn, args.mdrnn_checkpoint, device=device)
        mdrnn.eval()
        use_rnn = True
    
    # Load controller or PPO model
    if args.ppo_model and Path(args.ppo_model).exists():
        print("Loading PPO model...")
        from stable_baselines3 import PPO
        ppo_model = PPO.load(args.ppo_model)
        
        # Run PPO demo
        env = make_env(render_mode='human' if not args.no_render else None)
        rewards = []
        
        for ep in range(args.episodes):
            obs, _ = env.reset()
            episode_reward = 0
            
            if use_rnn:
                hidden = mdrnn.get_initial_hidden(1, device)
            
            print(f"\nEpisode {ep + 1}/{args.episodes}")
            
            with torch.no_grad():
                for step in range(1000):
                    # Encode observation
                    obs_processed = preprocess_obs(obs)
                    obs_tensor = torch.tensor(
                        obs_processed, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    
                    z = vae.encode(obs_tensor)
                    
                    if use_rnn:
                        h = hidden[0].squeeze(0).squeeze(0)
                        latent_obs = torch.cat([z.squeeze(0), h], dim=-1).cpu().numpy()
                    else:
                        latent_obs = z.squeeze(0).cpu().numpy()
                    
                    action, _ = ppo_model.predict(latent_obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    
                    if use_rnn:
                        action_tensor = torch.tensor(
                            action, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        _, _, _, hidden = mdrnn.forward_single(z, action_tensor, hidden)
                    
                    if not args.no_render:
                        time.sleep(0.01)
                    
                    if terminated or truncated:
                        break
            
            rewards.append(episode_reward)
            print(f"  Reward: {episode_reward:.1f}")
        
        env.close()
        mean_reward, std_reward = np.mean(rewards), np.std(rewards)
    
    else:
        # Load traditional controller
        print("Loading controller...")
        controller = Controller(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            action_dim=3
        ).to(device)
        load_checkpoint(controller, args.controller_checkpoint, device=device)
        controller.eval()
        
        # Run demo
        mean_reward, std_reward = run_demo(
            vae, mdrnn, controller, device,
            n_episodes=args.episodes,
            render=not args.no_render,
            use_rnn=use_rnn
        )
    
    print("\n" + "=" * 60)
    print(f"Average Score: {mean_reward:.1f} ± {std_reward:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
