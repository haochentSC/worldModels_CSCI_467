"""
VAE Training Script

Trains the Vision model (VAE) on collected data.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import VAE, vae_loss
from utils.misc import set_seed, get_device, save_checkpoint, print_model_summary


def visualize_reconstructions(model, data, device, num_samples=8, save_path=None):
    """Visualize original and reconstructed images."""
    model.eval()
    
    # Get random samples
    indices = np.random.choice(len(data), num_samples, replace=False)
    samples = torch.tensor(data[indices], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        recons, _, _, _ = model(samples)
    
    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    for i in range(num_samples):
        # Original
        img = samples[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstruction
        recon = recons[i].cpu().numpy().transpose(1, 2, 0)
        axes[1, i].imshow(recon)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reconstruction visualization to {save_path}")
    
    plt.close()
    return fig


def train_epoch(model, dataloader, optimizer, device, kl_weight=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for batch in dataloader:
        x = batch[0].to(device)
        
        optimizer.zero_grad()
        
        recon, mu, logvar, z = model(x)
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, kl_weight)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def validate(model, dataloader, device, kl_weight=1.0):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            
            recon, mu, logvar, z = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, kl_weight)
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train VAE for World Models")
    parser.add_argument('--data', type=str, default='data/episodes.npz',
                       help='Path to collected data (default: data/episodes.npz)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='Latent dimension (default: 32)')
    parser.add_argument('--kl-weight', type=float, default=1.0,
                       help='KL divergence weight (default: 1.0)')
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
    print("World Models - VAE Training")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"KL weight: {args.kl_weight}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data = np.load(args.data)
    observations = data['observations']
    print(f"Loaded {len(observations):,} frames")
    print(f"Shape: {observations.shape}")
    
    # Train/val split
    n_samples = len(observations)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_data = observations[train_indices]
    val_data = observations[val_indices]
    
    print(f"Training samples: {n_train:,}")
    print(f"Validation samples: {n_val:,}")
    
    # Create datasets
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = VAE(latent_dim=args.latent_dim).to(device)
    print_model_summary(model, "VAE")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"{args.log_dir}/vae_{timestamp}")
    
    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, args.kl_weight
        )
        
        # Validate
        val_loss, val_recon, val_kl = validate(
            model, val_loader, device, args.kl_weight
        )
        
        # Log
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Recon/train', train_recon, epoch)
        writer.add_scalar('Recon/val', val_recon, epoch)
        writer.add_scalar('KL/train', train_kl, epoch)
        writer.add_scalar('KL/val', val_kl, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                checkpoint_dir / 'vae_best.pt',
                extra={'latent_dim': args.latent_dim}
            )
        
        # Save periodic checkpoint
        if epoch % 5 == 0 or epoch == args.epochs:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                checkpoint_dir / f'vae_epoch_{epoch}.pt',
                extra={'latent_dim': args.latent_dim}
            )
            
            # Visualize reconstructions
            visualize_reconstructions(
                model, val_data, device,
                save_path=checkpoint_dir / f'reconstructions_epoch_{epoch}.png'
            )
    
    # Final visualization
    visualize_reconstructions(
        model, val_data, device,
        save_path=checkpoint_dir / 'reconstructions_final.png'
    )
    
    writer.close()
    
    print("\n" + "=" * 60)
    print("VAE Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'vae_best.pt'}")
    print("\nNext step: python -m scripts.train_mdrnn --epochs 20")
    print("Or skip RNN: python -m scripts.train_controller_ppo")
    print("=" * 60)


if __name__ == "__main__":
    main()
