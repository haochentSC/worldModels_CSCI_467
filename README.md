# World Models - CSCI 467 Project

A PyTorch implementation of [World Models](https://worldmodels.github.io/) (Ha & Schmidhuber, 2018) for the CarRacing-v3 environment.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        WORLD MODELS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐      ┌─────────┐      ┌────────────┐             │
│   │  VAE    │      │ MDN-RNN │      │ Controller │             │
│   │ (V)     │      │ (M)     │      │ (C)        │             │
│   └────┬────┘      └────┬────┘      └─────┬──────┘             │
│        │                │                  │                    │
│   64x64x3 → z(32)   z+a → h(256)      z+h → action            │
│   4.35M params      422K params        867 params              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start (Prototype - Day 1)

```bash
# 1. Setup environment
./scripts/setup.sh

# 2. Collect data (500 episodes, ~20 min)
python -m scripts.collect_data --episodes 500 --threads 4

# 3. Train VAE (~30-45 min)
python -m scripts.train_vae --epochs 5

# 4. Train Controller with PPO (~4-6 hours)
python -m scripts.train_controller_ppo --timesteps 500000
```

## Full Training Pipeline

```bash
# 1. Collect more data (2000 episodes)
python -m scripts.collect_data --episodes 2000 --threads 8

# 2. Train VAE (10 epochs)
python -m scripts.train_vae --epochs 10

# 3. Train MDN-RNN (20 epochs)
python -m scripts.train_mdrnn --epochs 20

# 4. Train Controller
# Option A: CMA-ES (original paper, 2-4 days)
python -m scripts.train_controller_cma --generations 500

# Option B: PPO (faster, ~6-12 hours)
python -m scripts.train_controller_ppo --timesteps 1000000
```

## Project Structure

```
worldModels_CSCI_467/
├── models/
│   ├── vae.py          # Vision model (V)
│   ├── mdrnn.py        # Memory model (M) 
│   └── controller.py   # Controller (C)
├── data/
│   └── (collected episodes stored here)
├── utils/
│   ├── misc.py         # Helper functions
│   └── envs.py         # Environment wrappers
├── configs/
│   └── default.py      # Hyperparameters
├── scripts/
│   ├── setup.sh        # Environment setup
│   ├── collect_data.py # Data collection
│   ├── train_vae.py    # VAE training
│   ├── train_mdrnn.py  # MDN-RNN training
│   ├── train_controller_ppo.py
│   └── train_controller_cma.py
├── checkpoints/        # Saved models
├── logs/               # Training logs
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU with 8GB+ VRAM (RTX 3080 recommended)
- 16GB+ RAM

## Results

| Method | Score (avg ± std) | Training Time |
|--------|-------------------|---------------|
| Prototype (VAE + random RNN + PPO) | ~700 | 6-8 hours |
| Full (VAE + MDN-RNN + CMA-ES) | ~850-900 | 3-5 days |
| Paper reported | 906 ± 21 | - |

## References

- [World Models Paper](https://arxiv.org/abs/1803.10122)
- [Interactive Article](https://worldmodels.github.io/)
- [ctallec PyTorch Implementation](https://github.com/ctallec/world-models)

## Authors

CSCI 467 - Machine Learning Project
