"""
Generate result plots for the docs:
  - training curve (eval reward vs timesteps)
  - final per-episode scores (real environment)
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-glob', type=str, default='logs/ppo_eval_*/evaluations.npz')
    parser.add_argument('--results', type=str, default='checkpoints/ppo_results.json')
    parser.add_argument('--out-dir', type=str, default='docs/images')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Training curve ---
    eval_files = sorted(glob.glob(args.eval_glob))
    if eval_files:
        eval_path = eval_files[-1]  # most recent run
        data = np.load(eval_path)
        ts = data['timesteps']
        results = data['results']  # (n_evals, n_episodes)
        mean = results.mean(axis=1)
        std = results.std(axis=1)

        plt.figure(figsize=(8, 5))
        plt.plot(ts, mean, '-o', color='#2563eb', label='Eval mean reward')
        plt.fill_between(ts, mean - std, mean + std, alpha=0.2, color='#2563eb',
                         label='±1 std')
        plt.xlabel('Training timesteps')
        plt.ylabel('Episode reward (latent env)')
        plt.title('PPO Controller Training Curve (CarRacing-v3)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        curve_path = out_dir / 'training_curve.png'
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"Saved {curve_path}")
    else:
        print(f"No eval files matched {args.eval_glob}")

    # --- Final per-episode scores ---
    if Path(args.results).exists():
        with open(args.results) as f:
            res = json.load(f)
        rewards = res['rewards']
        mean_r = res['mean_reward']

        plt.figure(figsize=(8, 5))
        colors = ['#16a34a' if r >= mean_r else '#dc2626' for r in rewards]
        plt.bar(range(1, len(rewards) + 1), rewards, color=colors)
        plt.axhline(mean_r, color='black', linestyle='--',
                    label=f"Mean = {mean_r:.0f}")
        plt.xlabel('Evaluation episode')
        plt.ylabel('Total reward (real env)')
        plt.title('Final Evaluation - Per-Episode Scores')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        plt.tight_layout()
        bar_path = out_dir / 'final_scores.png'
        plt.savefig(bar_path, dpi=150)
        plt.close()
        print(f"Saved {bar_path}")
    else:
        print(f"No results file at {args.results}")


if __name__ == "__main__":
    main()
