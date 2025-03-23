import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import math
from typing import Union

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

def CustomStepRateDecay(alpha_t: int, decay_rate: float) -> float:
    return alpha_t * decay_rate

# Exponential weighted average
def CustomExponentialWeightedAverage(beta: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return beta * x + (1 - beta) * y

# Gradient Descent with Momentum
class CustomSGDMomentumOptimizer():
    def __init__(self, params, lr: float = 0.001, momentum: float = 0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(param) for param in params]
        
    def zero_grad(self):
        for param in self.params:
            param.grad = torch.zeros_like(param)
            
    def step(self):
        for i, param in enumerate(self.params):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * param.grad
            with torch.no_grad():
                param += self.velocity[i]

# Adam optimizer
class CustomAdamOptimizer():
    def __init__(self, params, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
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

class CustomConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = True, device: Union[any, None] = None, dtype: Union[any, None] = None):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.device = device
        self.dtype = dtype

        # Initialize the kernel
        # Using P.Parameter to make the kernel a learnable parameter
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device, dtype=dtype))
        nn.init.xavier_uniform_(self.kernel)
        self.bias_param = nn.Parameter(torch.randn(out_channels))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Flip the kernel along the spatial dimensions (for convolution, not cross-correlation)
        kernel = torch.flip(self.kernel, [2, 3])
        kernel_height, kernel_width = kernel.shape[-2:]

        # # Unfold the input along the height and width dimensions.
        # # For input of shape [batch, channels, height, width], patches shape becomes:
        # # [batch, channels, out_height, out_width, kernel_height, kernel_width]
        # patches = input.unfold(2, kernel_height, self.stride).unfold(3, kernel_width, self.stride)
        
        # # Unsqueeze to add an output channel dimension:
        # # New shape: [batch, 1, channels, out_height, out_width, kernel_height, kernel_width]
        # patches = patches.unsqueeze(1)
        
        # # Reshape kernel:
        # # Starting shape: [out_channels, in_channels, kernel_height, kernel_width]
        # # Unsqueeze to: [1, out_channels, in_channels, 1, 1, kernel_height, kernel_width]
        # kernel = kernel.unsqueeze(0).unsqueeze(3).unsqueeze(3)
        
        # # Perform element-wise multiplication and sum over in_channels and kernel dimensions.
        # # This results in output shape: [batch, out_channels, out_height, out_width]
        # conv_out = (patches * kernel).sum(dim=(2, 5, 6))
        
        # patches shape: [batch, in_channels * kernel_height * kernel_width, L]
        patches = F.unfold(input, kernel_size=(kernel_height, kernel_width), padding=self.padding, stride=self.stride)

        kernel_flat = self.kernel.view(self.out_channels, -1)  # [out_channels, in_channels * kernel_height * kernel_width]
        conv_out = kernel_flat @ patches  # [batch, out_channels, L]

        # Reshape conv_out into [batch, out_channels, out_height, out_width]
        conv_out = conv_out.view(input.shape[0], self.out_channels, 
                                    int((input.shape[2] + 2 * self.padding - kernel_height) / self.stride + 1),
                                    int((input.shape[3] + 2 * self.padding - kernel_width) / self.stride + 1))

        # Optionally add bias
        if self.bias_param is not None:
            # Bias is added per output channel, so we need to reshape it for broadcasting
            conv_out = conv_out + self.bias_param.view(1, -1, 1, 1)
            
        return conv_out        

    def backward():
        pass
