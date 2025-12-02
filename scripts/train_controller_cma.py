"""
Controller Training with CMA-ES

Trains the controller using Covariance Matrix Adaptation Evolution Strategy.
This is the original approach from the paper but takes longer than PPO.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from multiprocessing import Pool
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cma

from models.vae import VAE
from models.mdrnn import MDRNN
from models.controller import Controller
from utils.envs import make_env, preprocess_obs
from utils.misc import set_seed, get_device, load_checkpoint, save_checkpoint


# Global variables for multiprocessing (avoids pickling large models)
_vae = None
_mdrnn = None
_device = None
_use_rnn = False
_latent_dim = 32
_hidden_dim = 256


def init_worker(vae_state, mdrnn_state, device_str, use_rnn, latent_dim, hidden_dim):
    """Initialize worker process with models."""
    global _vae, _mdrnn, _device, _use_rnn, _latent_dim, _hidden_dim
    
    _device = torch.device(device_str)
    _use_rnn = use_rnn
    _latent_dim = latent_dim
    _hidden_dim = hidden_dim
    
    # Load VAE
    _vae = VAE(latent_dim=latent_dim).to(_device)
    _vae.load_state_dict(vae_state)
    _vae.eval()
    
    # Load MDN-RNN if provided
    if mdrnn_state is not None:
        _mdrnn = MDRNN(
            latent_dim=latent_dim,
            action_dim=3,
            hidden_dim=hidden_dim,
            n_gaussians=5
        ).to(_device)
        _mdrnn.load_state_dict(mdrnn_state)
        _mdrnn.eval()
    else:
        _mdrnn = None


def evaluate_controller(params):
    """
    Evaluate a controller with given parameters.
    Returns negative reward (for minimization).
    """
    global _vae, _mdrnn, _device, _use_rnn, _latent_dim, _hidden_dim
    
    # Create controller and set parameters
    controller = Controller(
        latent_dim=_latent_dim,
        hidden_dim=_hidden_dim,
        action_dim=3
    ).to(_device)
    controller.set_params(params)
    controller.eval()
    
    # Run episode
    env = make_env(render_mode=None)
    obs, _ = env.reset()
    
    total_reward = 0
    
    # Initialize hidden state
    if _use_rnn and _mdrnn is not None:
        hidden = _mdrnn.get_initial_hidden(1, _device)
    else:
        hidden = None
    
    with torch.no_grad():
        for step in range(1000):
            # Encode observation
            obs_processed = preprocess_obs(obs)
            obs_tensor = torch.tensor(
                obs_processed, dtype=torch.float32, device=_device
            ).unsqueeze(0)
            
            z = _vae.encode(obs_tensor)
            
            # Get hidden state
            if _use_rnn and _mdrnn is not None:
                h = hidden[0].squeeze(0)
            else:
                h = torch.zeros(1, _hidden_dim, device=_device)
            
            # Get action
            action = controller.get_action(z, h)
            action_np = action.cpu().numpy()[0]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            
            # Update hidden state
            if _use_rnn and _mdrnn is not None:
                action_tensor = torch.tensor(
                    action_np, dtype=torch.float32, device=_device
                ).unsqueeze(0)
                _, _, _, hidden = _mdrnn.forward_single(z, action_tensor, hidden)
            
            if terminated or truncated:
                break
    
    env.close()
    
    # Return negative reward (CMA-ES minimizes)
    return -total_reward


def evaluate_batch(params_batch):
    """Evaluate a batch of controllers."""
    return [evaluate_controller(params) for params in params_batch]


def main():
    parser = argparse.ArgumentParser(description="Train Controller with CMA-ES")
    parser.add_argument('--vae-checkpoint', type=str, default='checkpoints/vae_best.pt',
                       help='Path to trained VAE')
    parser.add_argument('--mdrnn-checkpoint', type=str, default=None,
                       help='Path to trained MDN-RNN (optional)')
    parser.add_argument('--generations', type=int, default=500,
                       help='Number of CMA-ES generations (default: 500)')
    parser.add_argument('--pop-size', type=int, default=64,
                       help='Population size (default: 64)')
    parser.add_argument('--n-samples', type=int, default=16,
                       help='Rollouts per individual for averaging (default: 16)')
    parser.add_argument('--target-return', type=float, default=900,
                       help='Target return to stop early (default: 900)')
    parser.add_argument('--sigma', type=float, default=0.1,
                       help='Initial standard deviation (default: 0.1)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='Latent dimension (default: 32)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='RNN hidden dimension (default: 256)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for logs')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    
    print("=" * 60)
    print("World Models - Controller Training (CMA-ES)")
    print("=" * 60)
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"MDN-RNN checkpoint: {args.mdrnn_checkpoint or 'None'}")
    print(f"Generations: {args.generations}")
    print(f"Population size: {args.pop_size}")
    print(f"Workers: {args.workers}")
    print(f"Target return: {args.target_return}")
    print("=" * 60)
    
    # Load VAE
    print("\nLoading VAE...")
    vae = VAE(latent_dim=args.latent_dim).to(device)
    load_checkpoint(vae, args.vae_checkpoint, device=device)
    vae.eval()
    vae_state = vae.state_dict()
    
    # Load MDN-RNN (optional)
    mdrnn_state = None
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
        mdrnn_state = mdrnn.state_dict()
        use_rnn = True
    else:
        print("\nNo MDN-RNN - using VAE only")
    
    # Create controller to get parameter count
    controller = Controller(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        action_dim=3
    )
    n_params = controller.num_params()
    print(f"\nController has {n_params} parameters")
    
    # Initialize parameters
    initial_params = controller.get_params()
    
    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Initialize CMA-ES
    print("\nInitializing CMA-ES...")
    es = cma.CMAEvolutionStrategy(
        initial_params,
        args.sigma,
        {
            'popsize': args.pop_size,
            'seed': args.seed
        }
    )
    
    # Training log
    training_log = []
    best_reward = float('-inf')
    best_params = None
    
    # Initialize worker pool
    print(f"\nStarting training with {args.workers} workers...")
    print("This may take several hours to days.\n")
    
    # Use single process for debugging, multi-process for speed
    if args.workers > 1:
        pool = Pool(
            args.workers,
            initializer=init_worker,
            initargs=(vae_state, mdrnn_state, str(device), use_rnn, 
                     args.latent_dim, args.hidden_dim)
        )
    else:
        init_worker(vae_state, mdrnn_state, str(device), use_rnn,
                   args.latent_dim, args.hidden_dim)
        pool = None
    
    try:
        for gen in range(args.generations):
            # Ask for new solutions
            solutions = es.ask()
            
            # Evaluate solutions
            if pool is not None:
                # Parallel evaluation
                rewards = []
                for sol in solutions:
                    # Average over multiple rollouts
                    sol_rewards = pool.map(evaluate_controller, [sol] * args.n_samples)
                    rewards.append(np.mean(sol_rewards))
            else:
                # Sequential evaluation
                rewards = []
                for sol in solutions:
                    sol_rewards = [evaluate_controller(sol) for _ in range(args.n_samples)]
                    rewards.append(np.mean(sol_rewards))
            
            # Tell CMA-ES the fitness values (negative rewards for minimization)
            es.tell(solutions, rewards)
            
            # Log results (convert back to positive rewards)
            gen_best = -min(rewards)
            gen_mean = -np.mean(rewards)
            gen_std = np.std([-r for r in rewards])
            
            print(f"Gen {gen+1}/{args.generations}: "
                  f"Best={gen_best:.1f}, Mean={gen_mean:.1f}±{gen_std:.1f}")
            
            training_log.append({
                'generation': gen + 1,
                'best_reward': float(gen_best),
                'mean_reward': float(gen_mean),
                'std_reward': float(gen_std)
            })
            
            # Update best
            if gen_best > best_reward:
                best_reward = gen_best
                best_idx = np.argmin(rewards)
                best_params = solutions[best_idx].copy()
                
                # Save best controller
                controller.set_params(best_params)
                save_checkpoint(
                    controller, None, gen + 1, best_reward,
                    checkpoint_dir / 'controller_cma_best.pt',
                    extra={
                        'latent_dim': args.latent_dim,
                        'hidden_dim': args.hidden_dim,
                        'params': best_params.tolist()
                    }
                )
            
            # Save periodic checkpoint
            if (gen + 1) % 50 == 0:
                controller.set_params(es.result.xbest)
                save_checkpoint(
                    controller, None, gen + 1, -es.result.fbest,
                    checkpoint_dir / f'controller_cma_gen_{gen+1}.pt',
                    extra={
                        'latent_dim': args.latent_dim,
                        'hidden_dim': args.hidden_dim,
                        'params': es.result.xbest.tolist()
                    }
                )
                
                # Save training log
                with open(log_dir / f'cma_log_{timestamp}.json', 'w') as f:
                    json.dump(training_log, f, indent=2)
            
            # Early stopping
            if best_reward >= args.target_return:
                print(f"\nTarget return {args.target_return} achieved!")
                break
    
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    
    # Save final results
    controller.set_params(best_params)
    save_checkpoint(
        controller, None, args.generations, best_reward,
        checkpoint_dir / 'controller_cma_final.pt',
        extra={
            'latent_dim': args.latent_dim,
            'hidden_dim': args.hidden_dim,
            'params': best_params.tolist()
        }
    )
    
    with open(log_dir / f'cma_log_{timestamp}.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("\n" + "=" * 60)
    print("CMA-ES Training Complete!")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Checkpoint: {checkpoint_dir / 'controller_cma_best.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
