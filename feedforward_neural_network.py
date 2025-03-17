import numpy as np
from tensorflow.keras.datasets import fashion_mnist

class FeedforwardNeuralNetwork:
    def __init__(self, n, h, k):  
        self.layers = len(h) + 1  # Number of hidden layers + output layer
        self.w = []  
        self.b = []

        prev_size = n
        for h_i in h:
            self.w.append(np.random.randn(h_i, prev_size))
            self.b.append(np.zeros((h_i, 1)))
            prev_size = h_i
        
        # Output layer
        self.w.append(np.random.randn(k, prev_size))
        self.b.append(np.zeros((k, 1)))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # For numerical stability
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward_pass(self, X):
        A = X.T  # Transpose to match (features, samples) shape
        
        for i in range(self.layers - 1):  
            Z = np.dot(self.w[i], A) + self.b[i]
            A = self.sigmoid(Z)
        
        # Output layer 
        Z_out = np.dot(self.w[-1], A) + self.b[-1]
        Y_hat = self.softmax(Z_out)
        
        return Y_hat

#Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#preprocess data
x_train = x_train.reshape(-1, 28 * 28) / 255.0  # Flatten images
x_test = x_test.reshape(-1, 28 * 28) / 255.0  

#initialize model
input_size = 784  
hidden_sizes = [128, 64]  
output_size = 10 
model = FeedforwardNeuralNetwork(input_size, hidden_sizes, output_size)

X_sample = x_train[0:1]  #select the first image
Y_pred = model.forward_pass(X_sample)

print("Predicted Probabilities:", Y_pred)

