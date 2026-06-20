"""
Record Demo Script - Run the trained agent and save GIF recordings.

Unlike demo.py (which opens a live 'human' window), this renders to
'rgb_array' and writes animated GIFs + a results summary, suitable for
embedding in a README, LinkedIn post, or portfolio page.

Examples
--------
# Record the PPO controller (VAE-only path):
python -m scripts.record_demo --ppo-model checkpoints/ppo_controller/best_model.zip \
    --episodes 5 --out-dir docs

# Record a CMA/linear controller:
python -m scripts.record_demo --controller-checkpoint checkpoints/controller_cma_best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import VAE
from models.mdrnn import MDRNN
from models.controller import Controller
from utils.envs import make_env, preprocess_obs
from utils.misc import get_device, load_checkpoint


def _resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    img = Image.fromarray(frame).resize(
        (int(w * scale), int(h * scale)), Image.BILINEAR
    )
    return np.array(img)


def _encode(vae, obs, device):
    obs_processed = preprocess_obs(obs)
    obs_tensor = torch.tensor(
        obs_processed, dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        return vae.encode(obs_tensor)


def run_episode(env, vae, mdrnn, policy_fn, device, use_rnn, max_steps,
                capture=True, frame_scale=0.5, frame_skip=2):
    """Run one episode, returning (reward, frames)."""
    obs, _ = env.reset()
    episode_reward = 0.0
    frames = []

    hidden = mdrnn.get_initial_hidden(1, device) if use_rnn else None

    for step in range(max_steps):
        z = _encode(vae, obs, device)

        if use_rnn:
            h = hidden[0].squeeze(0).squeeze(0)
            latent_obs = torch.cat([z.squeeze(0), h], dim=-1)
        else:
            latent_obs = z.squeeze(0)

        action = policy_fn(latent_obs, z, hidden)

        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        if use_rnn:
            action_tensor = torch.tensor(
                action, dtype=torch.float32, device=device
            ).unsqueeze(0)
            _, _, _, hidden = mdrnn.forward_single(z, action_tensor, hidden)

        if capture and (step % frame_skip == 0):
            frame = env.render()
            if frame is not None:
                frames.append(_resize_frame(frame, frame_scale))

        if terminated or truncated:
            break

    return episode_reward, frames


def main():
    parser = argparse.ArgumentParser(description="Record GIFs of the trained agent")
    parser.add_argument('--vae-checkpoint', type=str, default='checkpoints/vae_best.pt')
    parser.add_argument('--mdrnn-checkpoint', type=str, default=None)
    parser.add_argument('--ppo-model', type=str, default=None,
                        help='Path to PPO .zip model (preferred for prototype)')
    parser.add_argument('--controller-checkpoint', type=str, default=None,
                        help='Path to linear/CMA controller checkpoint')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--out-dir', type=str, default='docs')
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--frame-scale', type=float, default=0.5)
    parser.add_argument('--frame-skip', type=int, default=2)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    out_dir = Path(args.out_dir)
    gif_dir = out_dir / 'gifs'
    gif_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("World Models - Recording Demo")
    print("=" * 60)

    # Load VAE
    print("\nLoading VAE...")
    vae = VAE(latent_dim=args.latent_dim).to(device)
    load_checkpoint(vae, args.vae_checkpoint, device=device)
    vae.eval()

    # Optional MDN-RNN
    mdrnn = None
    use_rnn = False
    if args.mdrnn_checkpoint and Path(args.mdrnn_checkpoint).exists():
        print("Loading MDN-RNN...")
        mdrnn = MDRNN(latent_dim=args.latent_dim, action_dim=3,
                      hidden_dim=args.hidden_dim, n_gaussians=5).to(device)
        load_checkpoint(mdrnn, args.mdrnn_checkpoint, device=device)
        mdrnn.eval()
        use_rnn = True

    # Build the policy function
    if args.ppo_model and Path(args.ppo_model).exists():
        print(f"Loading PPO model from {args.ppo_model}...")
        from stable_baselines3 import PPO
        ppo_model = PPO.load(args.ppo_model, device=device)

        def policy_fn(latent_obs, z, hidden):
            action, _ = ppo_model.predict(
                latent_obs.cpu().numpy(), deterministic=True
            )
            return action
        method = "PPO"
    elif args.controller_checkpoint and Path(args.controller_checkpoint).exists():
        print(f"Loading controller from {args.controller_checkpoint}...")
        controller = Controller(latent_dim=args.latent_dim,
                                hidden_dim=args.hidden_dim, action_dim=3).to(device)
        load_checkpoint(controller, args.controller_checkpoint, device=device)
        controller.eval()

        def policy_fn(latent_obs, z, hidden):
            if use_rnn:
                h = hidden[0].squeeze(0)
            else:
                h = torch.zeros(1, args.hidden_dim, device=device)
            with torch.no_grad():
                action = controller.get_action(z, h)
            return action.cpu().numpy()[0]
        method = "Controller"
    else:
        print("ERROR: provide --ppo-model or --controller-checkpoint")
        sys.exit(1)

    env = make_env(render_mode='rgb_array')

    rewards = []
    episodes = []
    for ep in range(args.episodes):
        reward, frames = run_episode(
            env, vae, mdrnn, policy_fn, device, use_rnn, args.max_steps,
            capture=True, frame_scale=args.frame_scale, frame_skip=args.frame_skip
        )
        rewards.append(reward)
        gif_path = gif_dir / f"episode_{ep+1}_score_{int(reward)}.gif"
        if frames:
            imageio.mimsave(gif_path, frames, fps=args.fps, loop=0)
        episodes.append({'episode': ep + 1, 'reward': float(reward),
                         'frames': len(frames), 'gif': str(gif_path)})
        print(f"  Episode {ep+1}/{args.episodes}: score {reward:.1f} "
              f"-> {gif_path.name} ({len(frames)} frames)")

    env.close()

    mean_r, std_r = float(np.mean(rewards)), float(np.std(rewards))
    best = max(episodes, key=lambda e: e['reward'])

    # Copy the best episode to a stable filename for README embedding
    best_gif = out_dir / 'gifs' / 'best_episode.gif'
    if Path(best['gif']).exists():
        import shutil
        shutil.copyfile(best['gif'], best_gif)

    results = {
        'method': method,
        'use_rnn': use_rnn,
        'episodes': episodes,
        'mean_reward': mean_r,
        'std_reward': std_r,
        'best_episode': best['episode'],
        'best_reward': best['reward'],
        'best_gif': str(best_gif),
    }
    results_path = out_dir / 'demo_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Average Score: {mean_r:.1f} +/- {std_r:.1f}  (best: {best['reward']:.1f})")
    print(f"GIFs saved to: {gif_dir}")
    print(f"Best episode  -> {best_gif}")
    print(f"Results JSON  -> {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
