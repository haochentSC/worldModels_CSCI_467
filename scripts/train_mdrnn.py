"""
MDN-RNN Training Script

Trains the Memory model (MDN-RNN) on latent sequences.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss
from utils.misc import set_seed, get_device, save_checkpoint, load_checkpoint, print_model_summary


class LatentSequenceDataset(Dataset):
    """
    Dataset that provides sequences of (z, action, next_z) for RNN training.
    """
    
    def __init__(self, observations, actions, episode_lengths, vae, device, seq_len=32):
        """
        Args:
            observations: All observations, shape (N, 3, 64, 64)
            actions: All actions, shape (N, 3)
            episode_lengths: Length of each episode
            vae: Trained VAE model
            device: Device for encoding
            seq_len: Sequence length for training
        """
        self.seq_len = seq_len
        self.device = device
        
        # Encode all observations to latent vectors
        print("Encoding observations to latent space...")
        vae.eval()
        
        latents = []
        batch_size = 256
        
        with torch.no_grad():
            for i in tqdm(range(0, len(observations), batch_size)):
                batch = torch.tensor(
                    observations[i:i+batch_size], 
                    dtype=torch.float32, 
                    device=device
                )
                z = vae.encode(batch)
                latents.append(z.cpu().numpy())
        
        self.latents = np.concatenate(latents, axis=0)
        self.actions = actions
        
        print(f"Encoded {len(self.latents)} observations to latent space")
        
        # Build index of valid sequence starting points
        self.valid_starts = []
        current_idx = 0
        
        for ep_len in episode_lengths:
            # Each position in the episode except the last seq_len can be a start
            for i in range(max(1, ep_len - seq_len)):
                self.valid_starts.append(current_idx + i)
            current_idx += ep_len
        
        print(f"Valid sequence starts: {len(self.valid_starts)}")
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.seq_len + 1  # +1 for next_z target
        
        # Get sequence
        z_seq = self.latents[start:end]
        action_seq = self.actions[start:end-1]  # Actions that led to each z
        
        # Pad if necessary (shouldn't happen often with valid_starts logic)
        if len(z_seq) < self.seq_len + 1:
            pad_len = self.seq_len + 1 - len(z_seq)
            z_seq = np.pad(z_seq, ((0, pad_len), (0, 0)), mode='edge')
            action_seq = np.pad(action_seq, ((0, pad_len), (0, 0)), mode='edge')
        
        # Split into input and target
        z_input = z_seq[:-1]  # z_0 to z_{T-1}
        z_target = z_seq[1:]  # z_1 to z_T
        
        return (
            torch.tensor(z_input, dtype=torch.float32),
            torch.tensor(action_seq, dtype=torch.float32),
            torch.tensor(z_target, dtype=torch.float32)
        )


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for z_input, actions, z_target in dataloader:
        z_input = z_input.to(device)
        actions = actions.to(device)
        z_target = z_target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pi, mu, sigma, _ = model(z_input, actions)
        
        # Compute loss
        loss = gmm_loss(z_target, pi, mu, sigma)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for z_input, actions, z_target in dataloader:
            z_input = z_input.to(device)
            actions = actions.to(device)
            z_target = z_target.to(device)
            
            pi, mu, sigma, _ = model(z_input, actions)
            loss = gmm_loss(z_target, pi, mu, sigma)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train MDN-RNN for World Models")
    parser.add_argument('--data', type=str, default='data/episodes.npz',
                       help='Path to collected data (default: data/episodes.npz)')
    parser.add_argument('--vae-checkpoint', type=str, default='checkpoints/vae_best.pt',
                       help='Path to trained VAE (default: checkpoints/vae_best.pt)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--seq-len', type=int, default=32,
                       help='Sequence length (default: 32)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='RNN hidden dimension (default: 256)')
    parser.add_argument('--n-gaussians', type=int, default=5,
                       help='Number of Gaussians in MDN (default: 5)')
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='Latent dimension from VAE (default: 32)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1)')
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
    print("World Models - MDN-RNN Training")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Number of Gaussians: {args.n_gaussians}")
    print("=" * 60)
    
    # Load VAE
    print("\nLoading VAE...")
    vae = VAE(latent_dim=args.latent_dim).to(device)
    load_checkpoint(vae, args.vae_checkpoint, device=device)
    vae.eval()
    
    # Load data
    print("\nLoading data...")
    data = np.load(args.data)
    observations = data['observations']
    actions = data['actions']
    episode_lengths = data['episode_lengths']
    
    print(f"Loaded {len(observations):,} frames from {len(episode_lengths)} episodes")
    
    # Create dataset
    full_dataset = LatentSequenceDataset(
        observations, actions, episode_lengths,
        vae, device, seq_len=args.seq_len
    )
    
    # Train/val split
    n_samples = len(full_dataset)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )
    
    print(f"Training sequences: {n_train:,}")
    print(f"Validation sequences: {n_val:,}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create MDN-RNN model
    model = MDRNN(
        latent_dim=args.latent_dim,
        action_dim=3,
        hidden_dim=args.hidden_dim,
        n_gaussians=args.n_gaussians
    ).to(device)
    print_model_summary(model, "MDN-RNN")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"{args.log_dir}/mdrnn_{timestamp}")
    
    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    
    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Log
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                checkpoint_dir / 'mdrnn_best.pt',
                extra={
                    'hidden_dim': args.hidden_dim,
                    'n_gaussians': args.n_gaussians,
                    'latent_dim': args.latent_dim
                }
            )
        
        # Save periodic checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                checkpoint_dir / f'mdrnn_epoch_{epoch}.pt',
                extra={
                    'hidden_dim': args.hidden_dim,
                    'n_gaussians': args.n_gaussians,
                    'latent_dim': args.latent_dim
                }
            )
    
    writer.close()
    
    print("\n" + "=" * 60)
    print("MDN-RNN Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'mdrnn_best.pt'}")
    print("\nNext step: python -m scripts.train_controller_ppo")
    print("Or use CMA-ES: python -m scripts.train_controller_cma")
    print("=" * 60)


if __name__ == "__main__":
    main()
