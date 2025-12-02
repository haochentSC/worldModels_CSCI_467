"""
Quick Test Script

Run this to verify everything is installed correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        import gymnasium
        print(f"  ✓ Gymnasium {gymnasium.__version__}")
    except ImportError as e:
        print(f"  ✗ Gymnasium: {e}")
        return False
    
    try:
        import stable_baselines3
        print(f"  ✓ Stable-Baselines3 {stable_baselines3.__version__}")
    except ImportError as e:
        print(f"  ✗ Stable-Baselines3: {e}")
        return False
    
    try:
        import cma
        print(f"  ✓ CMA-ES")
    except ImportError as e:
        print(f"  ✗ CMA-ES: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False
    
    return True


def test_environment():
    """Test CarRacing environment."""
    print("\nTesting CarRacing environment...")
    
    try:
        import gymnasium as gym
        env = gym.make('CarRacing-v3')
        obs, info = env.reset()
        print(f"  ✓ Environment created")
        print(f"    Observation shape: {obs.shape}")
        print(f"    Action space: {env.action_space}")
        
        # Quick step test
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Environment step works")
        
        env.close()
        return True
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        return False


def test_models():
    """Test model creation."""
    print("\nTesting models...")
    
    try:
        import torch
        from models.vae import VAE
        from models.mdrnn import MDRNN
        from models.controller import Controller
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test VAE
        vae = VAE(latent_dim=32).to(device)
        x = torch.randn(2, 3, 64, 64).to(device)
        recon, mu, logvar, z = vae(x)
        print(f"  ✓ VAE: input {x.shape} -> latent {z.shape} -> recon {recon.shape}")
        
        # Test MDN-RNN
        mdrnn = MDRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5).to(device)
        z_seq = torch.randn(2, 10, 32).to(device)
        a_seq = torch.randn(2, 10, 3).to(device)
        pi, mu, sigma, hidden = mdrnn(z_seq, a_seq)
        print(f"  ✓ MDN-RNN: input z{z_seq.shape} + a{a_seq.shape} -> pi{pi.shape}")
        
        # Test Controller
        controller = Controller(latent_dim=32, hidden_dim=256, action_dim=3).to(device)
        z = torch.randn(2, 32).to(device)
        h = torch.randn(2, 256).to(device)
        action = controller.get_action(z, h)
        print(f"  ✓ Controller: z{z.shape} + h{h.shape} -> action{action.shape}")
        print(f"    Controller params: {controller.num_params()}")
        
        return True
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions."""
    print("\nTesting utilities...")
    
    try:
        from utils.envs import preprocess_obs, BrownianPolicy
        from utils.misc import set_seed, get_device
        import numpy as np
        
        # Test preprocessing
        obs = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        processed = preprocess_obs(obs)
        print(f"  ✓ Preprocessing: {obs.shape} -> {processed.shape}, range [{processed.min():.2f}, {processed.max():.2f}]")
        
        # Test Brownian policy
        policy = BrownianPolicy()
        actions = [policy.sample() for _ in range(5)]
        print(f"  ✓ Brownian policy: generates smooth actions")
        
        # Test device detection
        device = get_device()
        print(f"  ✓ Device detection: {device}")
        
        return True
    except Exception as e:
        print(f"  ✗ Utility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("World Models - Installation Test")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_environment()
    all_passed &= test_models()
    all_passed &= test_utils()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
        print("\nYou're ready to start training:")
        print("  1. python -m scripts.collect_data --episodes 500")
        print("  2. python -m scripts.train_vae --epochs 5")
        print("  3. python -m scripts.train_controller_ppo --timesteps 500000")
    else:
        print("Some tests failed. Please fix the issues above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
