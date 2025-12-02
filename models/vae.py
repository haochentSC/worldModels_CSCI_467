"""
Variational Autoencoder (VAE) - Vision Model (V)

Compresses 64x64x3 RGB frames into 32-dimensional latent vectors.
Architecture: 4 conv layers (encoder) + 4 transposed conv layers (decoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """
    Encodes 64x64x3 images to latent distribution parameters (mu, logvar).
    
    Architecture:
        Conv2d(3, 32, 4, 2) -> 32x32
        Conv2d(32, 64, 4, 2) -> 16x16
        Conv2d(64, 128, 4, 2) -> 8x8
        Conv2d(128, 256, 4, 2) -> 4x4
        Flatten -> 4096
        Linear -> mu (32), logvar (32)
    """
    
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # After 4 convolutions with stride 2: 64 -> 32 -> 16 -> 8 -> 4
        # So output is 256 * 4 * 4 = 4096
        self.fc_mu = nn.Linear(4096, latent_dim)
        self.fc_logvar = nn.Linear(4096, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images, shape (batch, 3, 64, 64)
            
        Returns:
            mu: Mean of latent distribution, shape (batch, latent_dim)
            logvar: Log variance of latent distribution, shape (batch, latent_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    Decodes latent vectors back to 64x64x3 images.
    
    Architecture:
        Linear(32, 4096)
        Reshape -> 256x4x4
        ConvTranspose2d(256, 128, 4, 2) -> 8x8
        ConvTranspose2d(128, 64, 4, 2) -> 16x16
        ConvTranspose2d(64, 32, 4, 2) -> 32x32
        ConvTranspose2d(32, 3, 4, 2) -> 64x64
    """
    
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, 4096)
        
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector, shape (batch, latent_dim)
            
        Returns:
            Reconstructed image, shape (batch, 3, 64, 64)
        """
        x = F.relu(self.fc(z))
        x = x.view(-1, 256, 4, 4)
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))  # Output in [0, 1]
        
        return x


class VAE(nn.Module):
    """
    Complete Variational Autoencoder.
    
    The VAE learns to compress observations into a compact latent space
    that captures the essential features of the environment.
    """
    
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        This allows gradients to flow through the sampling operation.
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            
        Returns:
            Sampled latent vector z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input images, shape (batch, 3, 64, 64)
            
        Returns:
            recon: Reconstructed images, shape (batch, 3, 64, 64)
            mu: Latent mean
            logvar: Latent log variance
            z: Sampled latent vector
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent vectors (using mean, no sampling).
        Use this for inference/evaluation.
        
        Args:
            x: Input images, shape (batch, 3, 64, 64)
            
        Returns:
            z: Latent vectors (mu), shape (batch, latent_dim)
        """
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.
        
        Args:
            z: Latent vectors, shape (batch, latent_dim)
            
        Returns:
            Reconstructed images, shape (batch, 3, 64, 64)
        """
        return self.decoder(z)


def vae_loss(recon: torch.Tensor, 
             target: torch.Tensor, 
             mu: torch.Tensor, 
             logvar: torch.Tensor,
             kl_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Args:
        recon: Reconstructed images
        target: Original images
        mu: Latent mean
        logvar: Latent log variance
        kl_weight: Weight for KL term (beta-VAE)
        
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, target, reduction='mean')
    
    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = VAE(latent_dim=32).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 64, 64).to(device)
    recon, mu, logvar, z = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    
    # Test loss
    loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)
    print(f"Loss: {loss.item():.4f} (recon: {recon_loss.item():.4f}, kl: {kl_loss.item():.4f})")
