# Minimax-Optimal and Locally-Adaptive Online Nonparametric Regression

This repository provides an implementation to illustrate the results of the paper:  
📄 **"Minimax-Optimal and Locally-Adaptive Online Nonparametric Regression"**  
by **Paul Liautaud, Pierre Gaillard, and Olivier Wintenberger**.  
🚀 The paper is available at: [arXiv:2410.03363](https://arxiv.org/pdf/2410.03363).

## 📌 Overview

This project implements two online learning algorithms:
- **Chaining Tree (CT)**: A decision tree-based online regression method achieving minimax-optimal regret against Hölder continuous functions without prior knowledge.
- **Locally Adaptive Online Regression (Core Tree)**: A method that dynamically adapts to local smoothness variations.

The repository includes:
- A **separate `models.py` file**, which defines both algorithms.
- A **notebook (`minimax_locally_adaptive_reg.ipynb`)**, which trains the models and visualizes experimental results.
