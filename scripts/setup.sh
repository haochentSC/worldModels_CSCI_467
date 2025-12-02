#!/bin/bash

# World Models - Environment Setup Script
# Run this first to set up your environment

echo "=============================================="
echo "World Models CSCI 467 - Environment Setup"
echo "=============================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install SWIG first (required for Box2D)
echo ""
echo "Installing SWIG..."
pip install swig

# Install PyTorch (adjust for your CUDA version if needed)
echo ""
echo "Installing PyTorch..."
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1, use: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CPU only: pip install torch torchvision

# Install other requirements
echo ""
echo "Installing other requirements..."
pip install gymnasium[box2d] stable-baselines3 cma matplotlib tensorboard tqdm pillow numpy

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data checkpoints logs

# Test installation
echo ""
echo "Testing installation..."
python3 -c "
import torch
import gymnasium
import stable_baselines3

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('Gymnasium version:', gymnasium.__version__)
print('Stable-Baselines3 version:', stable_baselines3.__version__)

# Test CarRacing environment
env = gymnasium.make('CarRacing-v3')
obs, info = env.reset()
print('CarRacing environment created successfully!')
print('Observation shape:', obs.shape)
env.close()
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Collect data: python -m scripts.collect_data --episodes 500"
echo "  2. Train VAE: python -m scripts.train_vae --epochs 5"
echo "  3. Train controller: python -m scripts.train_controller_ppo"
echo "=============================================="
