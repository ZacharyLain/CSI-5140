import torch
import time
import matplotlib.pyplot as plt


# Set the random seed
torch.manual_seed(52)


# Example data for comparison (5 features, 100 examples)
X_pt = torch.rand(5, 100, dtype=torch.float32)  # 5 features, 100 examples
y_pt = torch.randint(0, 3, (100,), dtype=torch.long)  # Random labels (0, 1, or 2)


n_x, m = X_pt.shape  # Number of input features and number of examples
n_h1 = 5  # Number of neurons in the first hidden layer
n_h2 = 4  # Number of neurons in the second hidden layer
n_output = 3  # 3 output classes
learning_rate = 0.01
iterations = 1000


# Softmax function
def softmax(Z):
    """
    Use torch.exp() and XXX.sum(dim=0, keepdim=True)
    """
    theta = torch.exp(Z) / torch.sum(torch.exp(Z), dim=0, keepdim=True)
    return theta


# Cross-entropy loss
def cross_entropy_loss(A, y):
    """
    Use torch.log() and torch.sum()
    """
    Your Code Here
    return Your Code Here


### PyTorch 3-layer neural network for 3-class classification ###
def pytorch_3_layer_network(X, y, n_x, n_h1, n_h2, n_output, m, learning_rate, iterations):
    # Initialize weights and biases using torch.nn.Parameter(), so Pytorch knows they are parameters
    W1 = Your Code Here # Use torch.nn.Parameter() and torch.randn(dimesion1, dimesion2, dtype=torch.float32) * 0.01
    b1 = Your Code Here # Use torch.nn.Parameter() and torch.zeros((dimesion1, dimesion2), dtype=torch.float32)
    
    W2 = Your Code Here # Use torch.nn.Parameter() and torch.randn(dimesion1, dimesion2, dtype=torch.float32) * 0.01
    b2 = Your Code Here # Use torch.nn.Parameter() and torch.zeros((dimesion1, dimesion2), dtype=torch.float32)
    
    W3 = Your Code Here # Use torch.nn.Parameter() and torch.randn(dimesion1, dimesion2, dtype=torch.float32) * 0.01
    b3 = Your Code Here # Use torch.nn.Parameter() and torch.zeros((dimesion1, dimesion2), dtype=torch.float32)

    cost_values = []

    for iter in range(iterations):
        # Forward propagation
        Z1 = Your Code Here  # First hidden layer pre-activation, use torch.mm()
        A1 = Your Code Here  # First hidden layer activation, use torch.sigmoid()
        
        Z2 = Your Code Here  # Second hidden layer pre-activation, use torch.mm()
        A2 = Your Code Here  # Second hidden layer activation, use torch.sigmoid()
        
        Z3 = Your Code Here  # Output layer pre-activation (logits), use torch.mm()
        
        # Softmax activation for multi-class classification
        A3 = softmax(Z3)  # Apply softmax function to get class probabilities
                
        # Compute the cross-entropy loss
        J = cross_entropy_loss(A3, y)

        cost_values.append(J.item())


        # Backward propagation (calculate gradients)
        Your Code Here # Only one line


        # Update weights and biases using gradient descent
        with torch.no_grad():
            Your Code Here # W1 
            Your Code Here # b1 
            Your Code Here # W2 
            Your Code Here # b2 
            Your Code Here # W3 
            Your Code Here # b3

            # Zero the gradients after updating, use XXX.grad.zero_()
            Your Code Here # Zero the gradients of W1
            Your Code Here # Zero the gradients of b1
            Your Code Here # Zero the gradients of W2
            Your Code Here # Zero the gradients of b2
            Your Code Here # Zero the gradients of W3
            Your Code Here # Zero the gradients of b3

    return cost_values


### Timing and execution ###
start_time = time.time()
cost_values = pytorch_3_layer_network(X_pt, y_pt, n_x, n_h1, n_h2, n_output, m, learning_rate, iterations)
total_time = time.time() - start_time


# Plotting the cost value over iterations
plt.plot(range(iterations), cost_values)
plt.xlabel('Iterations')
plt.ylabel('Cost Value')
plt.title('Cost Value over Iterations (3-Class Classification)')
plt.tight_layout()
plt.savefig("cost_3layerNN_randomData.png")


print(f"Time: {total_time:.6f} seconds")
print(f"Final cost value: {cost_values[-1]:.6f}")

