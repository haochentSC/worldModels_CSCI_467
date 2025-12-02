"""
Controller Training with PPO

Trains the controller using Proximal Policy Optimization.
This is faster than CMA-ES and works well for prototypes.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from models.vae import VAE
from models.mdrnn import MDRNN
from utils.envs import WorldModelEnv, make_env, evaluate_policy
from utils.misc import set_seed, get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train Controller with PPO")
    parser.add_argument('--vae-checkpoint', type=str, default='checkpoints/vae_best.pt',
                       help='Path to trained VAE (default: checkpoints/vae_best.pt)')
    parser.add_argument('--mdrnn-checkpoint', type=str, default=None,
                       help='Path to trained MDN-RNN (optional, skip for faster prototype)')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps (default: 500000)')
    parser.add_argument('--eval-freq', type=int, default=10000,
                       help='Evaluation frequency (default: 10000)')
    parser.add_argument('--n-eval-episodes', type=int, default=5,
                       help='Number of evaluation episodes (default: 5)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='Latent dimension (default: 32)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='RNN hidden dimension (default: 256)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for checkpoints (default: checkpoints)')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for logs (default: logs)')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    
    print("=" * 60)
    print("World Models - Controller Training (PPO)")
    print("=" * 60)
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"MDN-RNN checkpoint: {args.mdrnn_checkpoint or 'None (using random RNN)'}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Learning rate: {args.lr}")
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
        print("\nLoading MDN-RNN...")
        mdrnn = MDRNN(
            latent_dim=args.latent_dim,
            action_dim=3,
            hidden_dim=args.hidden_dim,
            n_gaussians=5
        ).to(device)
        load_checkpoint(mdrnn, args.mdrnn_checkpoint, device=device)
        mdrnn.eval()
        use_rnn = True
        print("Using trained MDN-RNN")
    else:
        print("\nNo MDN-RNN checkpoint provided - training with VAE only")
        print("(This is fine for prototypes - untrained RNN achieves similar scores)")
    
    # Create wrapped environment
    print("\nCreating World Model environment...")
    
    def make_world_model_env():
        return WorldModelEnv(
            vae=vae,
            mdrnn=mdrnn,
            device=str(device),
            img_size=64,
            use_rnn=use_rnn
        )
    
    # Training environment
    train_env = DummyVecEnv([make_world_model_env])
    train_env = VecMonitor(train_env)
    
    # Evaluation environment
    eval_env = DummyVecEnv([make_world_model_env])
    eval_env = VecMonitor(eval_env)
    
    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / 'ppo_controller'),
        log_path=str(log_dir / f'ppo_eval_{timestamp}'),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq * 2,
        save_path=str(checkpoint_dir / 'ppo_checkpoints'),
        name_prefix='ppo_controller'
    )
    
    # Create PPO agent
    print("\nCreating PPO agent...")
    
    # Policy network architecture
    policy_kwargs = dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]
    )
    
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir / f'ppo_{timestamp}'),
        seed=args.seed
    )
    
    # Train
    print("\nStarting training...")
    print("This will take several hours. Check TensorBoard for progress:")
    print(f"  tensorboard --logdir {log_dir}")
    print()
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_path = checkpoint_dir / 'ppo_controller_final'
    model.save(final_path)
    print(f"\nSaved final model to {final_path}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    print("\nEvaluating on real environment...")
    real_env = make_env(render_mode=None)
    
    # Load the best model
    best_model_path = checkpoint_dir / 'ppo_controller' / 'best_model.zip'
    if best_model_path.exists():
        model = PPO.load(best_model_path)
        print(f"Loaded best model from {best_model_path}")
    
    # Evaluate
    rewards = []
    for ep in range(10):
        obs, _ = real_env.reset()
        episode_reward = 0
        
        # Get initial latent state
        from utils.envs import preprocess_obs
        obs_processed = preprocess_obs(obs)
        obs_tensor = torch.tensor(obs_processed, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            z = vae.encode(obs_tensor)
        
        if use_rnn:
            hidden = mdrnn.get_initial_hidden(1, device)
            h = hidden[0].squeeze(0).squeeze(0)
            latent_obs = torch.cat([z.squeeze(0), h], dim=-1).cpu().numpy()
        else:
            latent_obs = z.squeeze(0).cpu().numpy()
        
        for step in range(1000):
            action, _ = model.predict(latent_obs, deterministic=True)
            obs, reward, terminated, truncated, info = real_env.step(action)
            episode_reward += reward
            
            # Update latent state
            obs_processed = preprocess_obs(obs)
            obs_tensor = torch.tensor(obs_processed, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                z = vae.encode(obs_tensor)
            
            if use_rnn:
                action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                _, _, _, hidden = mdrnn.forward_single(z, action_tensor, hidden)
                h = hidden[0].squeeze(0).squeeze(0)
                latent_obs = torch.cat([z.squeeze(0), h], dim=-1).cpu().numpy()
            else:
                latent_obs = z.squeeze(0).cpu().numpy()
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        print(f"  Episode {ep+1}: {episode_reward:.1f}")
    
    real_env.close()
    train_env.close()
    eval_env.close()
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final score: {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"Best model: {checkpoint_dir / 'ppo_controller' / 'best_model.zip'}")
    print("=" * 60)
    
    # Save results
    results = {
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'rewards': [float(r) for r in rewards],
        'timesteps': args.timesteps,
        'use_rnn': use_rnn
    }
    
    import json
    with open(checkpoint_dir / 'ppo_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
