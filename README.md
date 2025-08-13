# GPU-Accelerated Sandpile Model Simulation (Colab Ready)

## Overview
This repository contains a **Google Colab / Jupyter Notebook** for simulating the **Bak–Tang–Wiesenfeld (BTW) sandpile model** with GPU acceleration using PyTorch.  
It supports:
- Large grid sizes (e.g., 256×256)
- **Wind bias** via a tunable parameter σ
- **Mass conservation** with fractional–integer grain accumulation
- Avalanche statistics and **power-law** analysis

The notebook can be opened directly in Google Colab, allowing you to run experiments without local setup.

---

## Run in Colab
Click the badge below to launch the notebook in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

---

## Features
- **Self-organized criticality**: The simulation evolves naturally to a critical state without fine-tuning.
- **Wind-biased kernel**: Models directional driving forces via σ.
- **Fully GPU-accelerated**: Fast relaxation steps via `torch.nn.functional.conv2d`.
- **Interactive parameters**: Change grid size, number of drops, burn-in period, σ, and see immediate results.
- **Built-in visualization**:
  - Heatmap of final stable configuration
  - Power-law distribution plots for avalanche size/duration
  - K–S test statistics

---

## How to Use
1. **Open in Colab** (no installation required)  
   or  
   **Clone and run locally**:
   ```bash
   git clone https://github.com/yourusername/sandpile_colab.git
   cd sandpile_colab
   pip install -r requirements.txt
   jupyter notebook
