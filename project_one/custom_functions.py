# Custom functions for FCNN, ReLU, softmax, regularization, optimization, and CNN
import torch
import torch.nn as nn
from torch.autograd import Function

import math

# Fully connected layer
# https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html
class CustomLinearFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # Forward pass: y = xW + b
        out = x.mm(weight) + bias
        
        # Cache the z, w, b for backward pass
        ctx.save_for_backward(x, weight, bias)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        # Backward pass
        # Get the saved tensor
        x, weight, bias = ctx.saved_tensors
        
        # Compute gradients
        gradient_x = grad_out.mm(weight.t())
        gradient_weight = x.t().mm(grad_out)
        gradient_bias = grad_out.sum(dim=0)

        return gradient_x, gradient_weight, gradient_bias

# Create a layer that inherits from nn.Module so I can use it in a model
class CustomFCLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        nn.init.xavier_uniform_(self.weight) # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
        
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return CustomLinearFunction.apply(x, self.weight, self.bias)

# ReLU activation function
def CustomReLU(x: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.zeros_like(x), x)

# Softmax activation function
def CustomSoftmax(x: torch.Tensor) -> torch.Tensor:
    # return torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)
    
    # Subtract the maximum value of x to prevent overflow
    shifted_x = x - x.max(dim=1, keepdim=True)[0]
    exp_x = torch.exp(shifted_x)
    
    return exp_x / exp_x.sum(dim=1, keepdim=True)



# Cross-Entropy loss function
class CustomCrossEntropyLoss():
    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return -torch.log(outputs[range(outputs.shape[0]), labels]).mean()

# L2 regularization
class CustomL2Regularization():
    def penalty(self, weight: torch.Tensor, l2_lambda: float) -> torch.Tensor:
        # lambda/2 * sum(weight^2)
        return (l2_lambda / 2) * (weight ** 2).sum() 
    
    def gradient(self, weight: torch.Tensor, l2_lambda: float) -> torch.Tensor:
        # lambda * weight
        return l2_lambda * weight

# L1 regularization
class CustomL1Regularization():
    def penalty(self, weight: torch.Tensor, l1_lambda: float) -> torch.Tensor:
        # lambda * sum(|w|)
        return l1_lambda * weight.abs().sum()
    
    def gradient(self, weight: torch.Tensor, l1_lambda: float) -> torch.Tensor:
        # lambda * sign(weight)
        return l1_lambda * torch.sign(weight)

# Dropout
class CustomDropout():
    def __call__(self, x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
        # Dont apply dropout during testing
        if not training:
            return x
        
        # Create a mask with the same shape as x
        # The mask is a boolean tensor with True values with probability p
        mask = torch.rand(x.shape) > p
        
        mask = mask.to(x.device)
        
        # Apply the mask to x
        # This will zero out some values of x
        return x * mask

# Cosine rate decay
def CustomCosineRateDecay(alpha_0: int, alpha_t: int, total_epoch: int) -> float:
    return (alpha_0 / 2) * (1 + math.cos((alpha_t * math.pi) / total_epoch))

# Exponential weighted average
def CustomExponentialWeightedAverage(beta: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return beta * x + (1 - beta) * y

# Adam optimizer
class CustomAdamOptimizer():
    def __init__(self, params, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.99, epsilon: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [torch.zeros_like(param) for param in params] # first moment
        self.v = [torch.zeros_like(param) for param in params] # second moment
        self.t = 0
        
    def zero_grad(self):
        for param in self.params:
            param.grad = torch.zeros_like(param)
            
    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad**2
            
            corrected_m = self.m[i] / (1 - self.beta1**self.t)
            corrected_v = self.v[i] / (1 - self.beta2**self.t)
            
            with torch.no_grad():
                param -= self.lr * corrected_m / (torch.sqrt(corrected_v) + self.epsilon)

# CNN layer

# Create a layer that inherits from nn.Module so I can use it in a model

