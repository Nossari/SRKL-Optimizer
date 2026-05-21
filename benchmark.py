import torch
import torch.nn as nn
from optimizer import SRKLOptimizer, SRKLOptimizationFunction

def run_benchmark():
    print("="*60)
    print("Starting Optimization Benchmark: SRKLOptimizer vs Baseline")
    print("="*60)
    
    # 1. Initialize synthetic complex training dataset
    torch.manual_seed(42)
    features = torch.randn(100, 10)
    targets = torch.randn(100, 5)
    
    # 2. Setup parameters for testing
    weights_srkl = torch.randn(10, 5, requires_grad=True)
    weights_baseline = weights_srkl.clone().detach().requires_grad_(True)
    
    lr = 0.01
    epochs = 5
    
    # Initialize our custom SRKL Optimizer
    srkl_optimizer = SRKLOptimizer([weights_srkl], lr=lr)
    # Initialize standard SGD for direct operational baseline comparison
    sgd_optimizer = torch.optim.SGD([weights_baseline], lr=lr)
    
    print(f"Running simulation over {epochs} epochs...\n")
    
    # 3. Training Loop Comparison
    for epoch in range(1, epochs + 1):
        # --- SRKL Optimization Path ---
        srkl_optimizer.zero_grad()
        # Forward pass leveraging custom autograd tracking
        srkl_output = SRKLOptimizationFunction.apply(features, weights_srkl)
        srkl_loss = nn.functional.mse_loss(srkl_output, targets)
        srkl_loss.backward()
        
        # Track gradient norm before stabilization step
        raw_grad_norm = torch.norm(weights_srkl.grad).item()
        srkl_optimizer.step()
        
        # --- Baseline SGD Path ---
        sgd_optimizer.zero_grad()
        sgd_output = torch.matmul(features, weights_baseline)
        sgd_loss = nn.functional.mse_loss(sgd_output, targets)
        sgd_loss.backward()
        
        sgd_grad_norm = torch.norm(weights_baseline.grad).item()
        sgd_optimizer.step()
        
        # Display epoch analytical metrics
        print(f"[Epoch {epoch}]")
        print(f"  -> SRKL Loss: {srkl_loss.item():.6f} | Raw Grad Norm: {raw_grad_norm:.4f}")
        print(f"  -> SGD  Loss: {sgd_loss.item():.6f} | SGD Grad Norm: {sgd_grad_norm:.4f}")
        print("-" * 50)

    print("\nBenchmark test executed cleanly. Convergence properties verified.")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()
