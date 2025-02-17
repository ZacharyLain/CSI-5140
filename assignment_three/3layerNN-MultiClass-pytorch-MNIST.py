import torch
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt


# Set the random seed
torch.manual_seed(100)


# Load full MNIST dataset and nomalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)


# Create a smaller dataset by selecting a subset (e.g., 1000 samples) since training the full dataset is slow
small_dataset_size = 1000  # Specify the size of the smaller dataset
small_dataset = Subset(full_dataset, list(range(small_dataset_size)))


# Create a DataLoader with smaller batch size for the subset
train_loader = DataLoader(small_dataset, batch_size=32, shuffle=True)


# Define model parameters for MNIST
n_x = 28 * 28  # Input size: 28x28 grayscale images flattened to a 1D array
n_h1 = 128  # Number of neurons in the first hidden layer
n_h2 = 64  # Number of neurons in the second hidden layer
n_output = 10  # 10 output classes (for MNIST)
learning_rate = 0.8
epochs = 100  # Reduce the number of iterations


# Softmax function
def softmax(Z):
    theta = torch.exp(Z) / torch.sum(torch.exp(Z), dim=0, keepdim=True)
    return theta


# Manual cross-entropy loss
def cross_entropy_loss(A, y):
    loss = -torch.log(A[y, torch.arange(y.shape[0])]).mean()
    
    return loss


# Neural network for MNIST classification
def pytorch_3_layer_network(X, y, n_x, n_h1, n_h2, n_output, learning_rate, epochs):
    # Initialize weights and biases using torch.nn.Parameter(), so Pytorch knows they are parameters
    W1 = torch.nn.Parameter(torch.randn(n_h1, n_x, dtype=torch.float32) * 0.01) # Use torch.nn.Parameter() and torch.randn(dimesion1, dimesion2, dtype=torch.float32) * 0.01
    b1 = torch.nn.Parameter(torch.zeros((n_h1, 1), dtype=torch.float32)) # Use torch.nn.Parameter() and torch.zeros((dimesion1, dimesion2), dtype=torch.float32)
    
    W2 = torch.nn.Parameter(torch.randn(n_h2, n_h1, dtype=torch.float32) * 0.01) # Use torch.nn.Parameter() and torch.randn(dimesion1, dimesion2, dtype=torch.float32) * 0.01
    b2 = torch.nn.Parameter(torch.zeros((n_h2, 1), dtype=torch.float32)) # Use torch.nn.Parameter() and torch.zeros((dimesion1, dimesion2), dtype=torch.float32)
    
    W3 = torch.nn.Parameter(torch.randn(n_output, n_h2, dtype=torch.float32) * 0.01) # Use torch.nn.Parameter() and torch.randn(dimesion1, dimesion2, dtype=torch.float32) * 0.01
    b3 = torch.nn.Parameter(torch.zeros((n_output, 1), dtype=torch.float32)) # Use torch.nn.Parameter() and torch.zeros((dimesion1, dimesion2), dtype=torch.float32)

    cost_values = []

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for data in train_loader:
            inputs, labels = data
            X = inputs.view(-1, n_x).T # Flatten the images and transpose, you can use inputs.view(-1, dimension).T
            y = labels

            # Forward propagation
            Z1 = torch.mm(W1, X) + b1  # First hidden layer pre-activation, use torch.mm()
            A1 = torch.sigmoid(Z1)  # First hidden layer activation, use torch.sigmoid()
            
            Z2 = torch.mm(W2, A1) + b2  # Second hidden layer pre-activation, use torch.mm()
            A2 = torch.sigmoid(Z2) # Second hidden layer activation, use torch.sigmoid()
            
            Z3 = torch.mm(W3, A2) + b3 # Output layer pre-activation (logits), use torch.mm()
            
            # Softmax activation for multi-class classification
            A3 = softmax(Z3)  # Apply softmax to get class probabilities
            
            # Compute the cross-entropy loss
            J = cross_entropy_loss(A3, y)

            # Backward propagation (calculate gradients)
            J.backward() # Only one line

            # Update weights and biases using gradient descent
            with torch.no_grad():
                W1 -= learning_rate * W1.grad # W1 
                b1 -= learning_rate * b1.grad # b1 
                W2 -= learning_rate * W2.grad # W2 
                b2 -= learning_rate * b2.grad # b2 
                W3 -= learning_rate * W3.grad # W3 
                b3 -= learning_rate * b3.grad # b3

                # Zero the gradients after updating, use XXX.grad.zero_()
                W1.grad.zero_() # Zero the gradients of W1
                b1.grad.zero_() # Zero the gradients of b1
                W2.grad.zero_() # Zero the gradients of W2
                b2.grad.zero_() # Zero the gradients of b2
                W3.grad.zero_() # Zero the gradients of W3
                b3.grad.zero_() # Zero the gradients of b3

            cost_values.append(J.item())
            
            print(f'Loss: {J.item()}')

    return cost_values

### Timing and execution ###
start_time = time.time()
cost_values = pytorch_3_layer_network(None, None, n_x, n_h1, n_h2, n_output, learning_rate, epochs)
total_time = time.time() - start_time

# Plotting the cost value over iterations
plt.plot(range(len(cost_values)), cost_values)
plt.xlabel('Iterations')
plt.ylabel('Cost Value')
plt.title('Cost Value over Iterations (MNIST Classification with Small Dataset)')
plt.tight_layout()
plt.savefig("cost_MNIST_small_dataset.png")

print(f"Time: {total_time:.6f} seconds")
print(f"Final cost value: {cost_values[-1]:.6f}")
