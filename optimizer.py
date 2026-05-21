import torch
from torch.optim import Optimizer

# 1. Importing the custom autograd function we created earlier
class SRKLOptimizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights):
        ctx.save_for_backward(x, weights)
        output = torch.matmul(x, weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weights = ctx.saved_tensors
        grad_x = torch.matmul(grad_output, weights.t())
        grad_weights = torch.matmul(x.t(), grad_output)
        
        with torch.no_grad():
            gradient_norm_sq = torch.norm(grad_weights) ** 2
            damping_factor = 1.0 / (1.0 + gradient_norm_sq)
            
        grad_weights_stabilized = grad_weights * damping_factor
        return grad_x, grad_weights_stabilized

# 2. Building the full PyTorch Standard Optimizer
class SRKLOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3):
        """
        SRKL Kinetic Braking Optimizer initialization.
        :param params: iterable of parameters to optimize or dicts defining parameter groups.
        :param lr: learning rate (default: 1e-3).
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr)
        super(SRKLOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step (parameter update).
        """
        loss = None
        if closure != None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Retrieve the stabilized gradient computed by our custom autograd function
                grad = p.grad.data
                
                # Standard gradient descent update step driven by our stabilized gradients
                p.data.add_(grad, alpha=-lr)

        return loss

# --- Integration and Usage Verification Example ---
if __name__ == "__main__":
    # Define a simple linear model layer parameters manually
    inputs = torch.randn(10, 5)
    weights = torch.nn.Parameter(torch.randn(5, 3))
    
    # Instantiate the custom SRKL optimizer
    optimizer = SRKLOptimizer([weights], lr=0.01)
    
    # Simulate a single training forward/backward iteration loop
    optimizer.zero_grad()
    
    # Execute through our custom autograd layer forward pass
    outputs = SRKLOptimizationFunction.apply(inputs, weights)
    loss = outputs.sum()
    
    # Execute backward pass to invoke custom structural kinetic damping logic
    loss.backward()
    
    # Update parameters using our optimizer step
    optimizer.step()
    
    print("SRKLOptimizer update step completed successfully without configuration failures!")
        # Return calibrated gradients matching the forward input signature
        return grad_x, grad_weights_stabilized

# --- Verification Example ---
if __name__ == "__main__":
    input_data = torch.randn(10, 5, requires_grad=True)
    network_weights = torch.randn(5, 3, requires_grad=True)
    
    optimized_layer = SRKLOptimizationFunction.apply
    result = optimized_layer(input_data, network_weights)
    
    loss = result.sum()
    loss.backward()
    
    print("Custom backward pass executed successfully with kinetic braking architecture applied.")
    print("Stabilized Weight Gradients:\n", network_weights.grad)
