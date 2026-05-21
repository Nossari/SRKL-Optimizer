# SRKL Optimizer: Second-Order Autograd with Kinetic Braking

A high-performance, memory-efficient second-order optimization framework for PyTorch. This repository implements dynamic kinetic braking and structural curvature damping directly inside a custom autograd loop, bypassing framework-level memory constraints.

---

## 🔬 Core Architecture & Mathematical Foundation

Standard second-order optimizers suffer from severe memory bottlenecks because calculating the full Hessian matrix $\mathbf{H}$ requires $O(N^2)$ space complexity. 

The **SRKL Optimizer** addresses this by avoiding the full matrix construction entirely. Instead, it computes a localized, empirical curvature approximation during the manual Vector-Jacobian Product (VJP) step.

### 1. Vector-Jacobian Product (VJP) Formulation
During the custom backward pass, we compute the first-order parameter gradients cleanly:
$$\mathbf{g}_w = \frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^T \cdot \nabla_{\mathbf{O}}\mathcal{L}$$

### 2. Kinetic Braking & Structural Damping Protocol
To suppress optimization oscillations and prevent gradient explosions without allocating heavy tensor graphs, we apply an empirical local curvature damping factor $\gamma$:
$$\gamma = \frac{1}{1 + \|\mathbf{g}_w\|_2^2}$$

The updated, stabilized gradient passed back to the computational subsystem is formulated as:
$$\mathbf{g}_{stabilized} = \mathbf{g}_w \odot \gamma$$

This mathematical constraints ensures structural stability during aggressive learning rate phases while operating strictly within a friendly memory footprint.

---

## 🚀 Features
* **Zero Core Overhead:** Operates entirely in user space utilizing PyTorch standard Eager Mode via `torch.autograd.Function`.
* **Memory Protection:** Avoids full Hessian memory accumulation blocks ($O(N)$ memory scaling).
* **Native Integration:** Fully inherits from `torch.optim.Optimizer` for a plug-and-play experience.

---

## 📦 Project Structure
* `optimizer.py`: Contains the core `SRKLOptimizationFunction` and the production-ready `SRKLOptimizer` engine.
* `benchmark.py`: A comprehensive simulation environment verifying optimizer convergence against basic baselines.

---

## 🛠️ Quick Start

```python
import torch
from optimizer import SRKLOptimizer, SRKLOptimizationFunction

# Model & Data initialization
inputs = torch.randn(10, 5)
weights = torch.nn.Parameter(torch.randn(5, 3))

# Instantiate Optimizer
optimizer = SRKLOptimizer([weights], lr=0.01)

# Training loop step
optimizer.zero_grad()
outputs = SRKLOptimizationFunction.apply(inputs, weights)
loss = outputs.sum()
loss.backward()
optimizer.step()
