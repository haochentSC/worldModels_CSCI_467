# Portfolio handoff prompt

Paste the block below into a Claude Code session **running inside your portfolio
repo**. It tells that session everything it needs to add a "World Models" project
entry with the GIF and graphs — no need for it to re-derive anything.

---

```
I want to add a new project to this portfolio: a "World Models" deep-RL agent that
learns to drive in CarRacing-v3. Read the source repo for full context, then add a
project entry to this portfolio that matches the existing style/structure of the
other projects here.

SOURCE REPO
- Local path:  C:\Users\tongh\worldModels_CSCI_467
- GitHub:      https://github.com/haochentSC/worldModels_CSCI_467
- Read its README.md and docs/README.md first to understand the project and tone.

WHAT THE PROJECT IS (use for the write-up)
- A PyTorch implementation of "World Models" (Ha & Schmidhuber, 2018) for the
  CarRacing-v3 environment.
- Pipeline: a Variational Autoencoder (VAE) compresses each 64x64 game frame into a
  32-dimensional latent vector; a controller then learns to drive purely from those
  latents using PPO (Stable-Baselines3).
- Trained end-to-end on a single NVIDIA RTX 3080 Laptop GPU in ~2 hours.
- Notable engineering detail worth highlighting: diagnosed and fixed VAE *posterior
  collapse* (the default KL weight made the latents uninformative and reconstructions
  blurry/identical); dropping the KL weight to 1e-4 restored sharp, informative
  latents.

RESULTS (use these exact numbers)
- Mean score: 285 +/- 195 over 10 evaluation episodes (real environment)
- Best episode: 600
- Training: 500,000 PPO timesteps, ~2 hours
- Eval reward improved from about -32 to a ~440 peak during training.

TECH STACK / SKILLS TO TAG
Python, PyTorch, Reinforcement Learning, PPO, Variational Autoencoders,
Stable-Baselines3, Gymnasium, CUDA/GPU training, NumPy.

ASSETS TO EMBED (use GitHub raw URLs so they work on a deployed site; or copy the
files locally into the portfolio's assets folder if that's the convention here)
- Gameplay GIF (hero/thumbnail):
  https://raw.githubusercontent.com/haochentSC/worldModels_CSCI_467/main/docs/gifs/best_episode.gif
- VAE reconstructions (what the agent "sees"):
  https://raw.githubusercontent.com/haochentSC/worldModels_CSCI_467/main/docs/images/vae_reconstructions.png
- PPO training curve (reward vs timesteps):
  https://raw.githubusercontent.com/haochentSC/worldModels_CSCI_467/main/docs/images/training_curve.png
- Final per-episode scores (bar chart):
  https://raw.githubusercontent.com/haochentSC/worldModels_CSCI_467/main/docs/images/final_scores.png
  (Local copies of all of the above live under C:\Users\tongh\worldModels_CSCI_467\docs\)

WHAT TO DO
1. Detect this portfolio's structure and conventions (framework, where projects are
   defined — e.g. a projects data file, MDX/Markdown files, or a component — and how
   existing projects embed images/GIFs). Follow that exact pattern; do not invent a
   new one.
2. Add the World Models project: title, 1-2 sentence summary, a longer description
   using the facts above, the tech tags, a link to the GitHub repo, and the embedded
   GIF + the three graphs (with alt text / captions).
3. If the portfolio copies assets locally, download/copy the four files into the
   correct assets directory and reference them with the site's normal asset paths.
4. Keep the writing concise and recruiter-friendly. Lead with the GIF.
5. Show me a diff/preview before finalizing, and tell me how to run the dev server to
   verify it renders.

Suggested one-liner summary you may use or improve:
"Self-driving RL agent for CarRacing-v3 built on the World Models architecture
(VAE + PPO) — trained end-to-end on a single GPU, reaching a best score of 600."
```
