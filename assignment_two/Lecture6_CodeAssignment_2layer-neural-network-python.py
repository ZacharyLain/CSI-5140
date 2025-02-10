import numpy as np
import matplotlib.pyplot as plt
import time

# Set the random seed
np.random.seed(52)

# Example data for comparison (5 features, 100 examples)
X = np.random.rand(5, 100)  # 5 features, 100 examples
y = np.random.randint(0, 2, (1, 100))  # Random binary labels (0 or 1)

n_x, m = X.shape  # Number of input features and number of examples
n_h = 4  # Number of neurons in the hidden layer
learning_rate = 0.01
iterations = 1000

# Activation function (sigmoid) and its derivative
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    return A * (1 - A)

### Non-vectorized 2-layer neural network ###
def non_vectorized_2_layer_network(X, y, n_x, n_h, m, learning_rate, iterations):
    # Initialize weights and biases, the small value is set to 0.01
    W1 = np.random.randn(n_h, n_x) * 0.01  # Initialize weights for the first layer: np.random.randn(Shape) * 0.01
    b1 = np.zeros(n_h)  # Bias for the first layer: np.zeros((Shape))
    W2 = np.random.randn(1, n_h)  # Initialize weights for the second layer: np.random.randn(Shape) * 0.01  
    b2 = np.zeros(1) # Bias for the second layer: np.zeros((Shape))
    
    costs = []  # To store cost values
    
    for iter in range(iterations): 
        J = 0  # Cost initialization

        # Gradients initialization
        dW1 = np.zeros(W1.shape)
        db1 = np.zeros(b1.shape)
        dW2 = np.zeros(W2.shape)
        db2 = np.zeros(b2.shape)

        for i in range(m):            
            # Forward propagation for each example
            Z1 = np.dot(W1, X[:, i]) + b1  # First layer Z1
            A1 = sigmoid(Z1) # First layer A1
            Z2 = np.dot(W2, A1)  # Second layer Z2
            A2 = sigmoid(Z2)  # Second layer A2 (output)

            # Compute the cost for this example
            J += -(y[:, i] * np.log(A2) + (1 - y[:, i]) * np.log(1 - A2))

            # Backward propagation
            dZ2 = A2 - y[:, i] # Gradient for output layer
            dW2 += dZ2 * A1 # Gradient for W2
            db2 += dZ2  # Gradient for b2

            dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(Z1) # Gradient for hidden layer
            dW1 += np.outer(dZ1, X[:, i]) # Gradient for W1
            db1 += dZ1  # Gradient for b1

        # Average the cost and gradients
        J = J / m # J 
        dW1 = dW1 / m # dW1 
        db1 = db1 / m # db1 
        dW2 = dW2 / m # dW2 
        db2 = db2 / m # db2 

        # Update weights and biases
        W1 = W1 - learning_rate * dW1 # W1 
        b1 = b1 - learning_rate * db1 # b1 
        W2 = W2 - learning_rate * dW2 # W2 
        b2 = b2 - learning_rate * db2 # b2 

        # Store the cost for this iteration
        costs.append(J)
    
    return W1, b1, W2, b2, costs

### Vectorized 2-layer neural network ###
def vectorized_2_layer_network(X, y, n_x, n_h, m, learning_rate, iterations):

    # Initialize weights and biases
    W1 = np.random.randn(n_h, n_x) * 0.01 # Initialize weights for the first layer: np.random.randn(Shape) * 0.01
    b1 = np.zeros((n_h, 1)) # Bias for the first layer: np.zeros((Shape))
    W2 = np.random.randn(1, n_h) * 0.01 # Initialize weights for the second layer: np.random.randn(Shape) * 0.01  
    b2 = np.zeros((1, 1)) # Bias for the second layer: np.zeros((Shape))

    costs = []  # To store cost values

    for iter in range(iterations):
        # Forward propagation
        Z1 = np.dot(W1, X) + b1 # First layer Z1 (vectorized)
        A1 = sigmoid(Z1) # First layer activation (vectorized)
        Z2 = np.dot(W2, A1) + b2 # Second layer Z2 (vectorized)
        A2 = sigmoid(Z2) # Second layer activation (output, vectorized)
        
        # Compute cost
        J = -np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2)) / m
        
        # Backward propagation
        dZ2 = A2 - y # Gradient for output layer (vectorized)
        dW2 = 1/m * np.dot(dZ2, A1.T)  # Gradient for W2 (vectorized)
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True) # Gradient for b2 (vectorized)

        dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(Z1) # Gradient for hidden layer (vectorized)
        dW1 = 1/m * np.dot(dZ1, X.T) # Gradient for W1 (vectorized)
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True) # Gradient for b1 (vectorized)

        # Update weights and biases
        W1 = W1 - learning_rate * dW1 # W1 
        b1 = b1 - learning_rate * db1 # b1 
        W2 = W2 - learning_rate * dW2 # W2 
        b2 = b2 - learning_rate * db2 # b2 

        # Store the cost for this iteration
        costs.append(J)
        
    return W1, b1, W2, b2, costs

### Timing and execution ###

# Non-vectorized execution and cost tracking
start_time = time.time()
W1_non_vec, b1_non_vec, W2_non_vec, b2_non_vec, costs_non_vec = non_vectorized_2_layer_network(X, y, n_x, n_h, m, learning_rate, iterations)
non_vec_time = time.time() - start_time
print(f"Non-vectorized 2-layer Network Time: {non_vec_time:.6f} seconds")

# Vectorized execution and cost tracking
start_time = time.time()
W1_vec, b1_vec, W2_vec, b2_vec, costs_vec = vectorized_2_layer_network(X, y, n_x, n_h, m, learning_rate, iterations)
vec_time = time.time() - start_time
print(f"Vectorized 2-layer Network Time: {vec_time:.6f} seconds")



## Plot the cost for both approaches

# Flatten costs lists (costs are stored as 2D arrays, so we need to flatten them for plotting)
costs_non_vec = np.array(costs_non_vec).flatten()
costs_vec = np.array(costs_vec).flatten()

# Plot
plt.figure(figsize=(10,6))
plt.plot(range(iterations), costs_non_vec, label='Non-vectorized', color='blue')
plt.plot(range(iterations), costs_vec, label='Vectorized', color='green', linestyle='--')
plt.title('Cost of a 2-layer neuron network over iterations for Non-vectorized and Vectorized Implementations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)
plt.savefig("cost.png")


# Compare the final cost values
# In the non-vectorized version, each calculation is done step-by-step for individual examples, whereas in the vectorized version, operations are done in bulk. 
# Due to how floating-point arithmetic is handled by the underlying hardware, the order of operations can introduce very small differences.
J_non_vec = costs_non_vec[-1]
J_vec = costs_vec[-1]
print(f"Cost from non-vectorized: {J_non_vec:.6f}")
print(f"Cost from vectorized: {J_vec:.6f}")
tolerance = 1e-3 # Define a tolerance for comparison
if abs(J_non_vec - J_vec) < tolerance:
    print("The costs are effectively the same.")
else:
    print(f"The costs are different. Difference: {abs(J_non_vec - J_vec):.10f}")