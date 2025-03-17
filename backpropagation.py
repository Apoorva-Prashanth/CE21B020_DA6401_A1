import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Neural Network Class
class FeedforwardNN:
    def __init__(self, layer_sizes, activation="relu"):
        self.layer_sizes = layer_sizes
        self.activation_func = relu if activation == "relu" else sigmoid
        self.activation_derivative = relu_derivative if activation == "relu" else sigmoid_derivative
        self.initialize_weights()

    def initialize_weights(self):
        self.w = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(2 / self.layer_sizes[i]) 
                  for i in range(len(self.layer_sizes) - 1)]
        self.b = [np.zeros((1, self.layer_sizes[i + 1])) for i in range(len(self.layer_sizes) - 1)]

    def forward_pass(self, X):
        activations = [X]
        Z_values = []
        for i in range(len(self.w)):
            Z = np.dot(activations[-1], self.w[i]) + self.b[i]
            Z_values.append(Z)
            if i < len(self.w) - 1:
                A = self.activation_func(Z)  # Hidden layers use activation
            else:
                A = self.softmax(Z)  # Final layer uses softmax
            activations.append(A)
        return activations, Z_values
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Optimizer Class
class Optimizer:
    def __init__(self, model, method="adam", lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.method = method.lower()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.initialize_optimizers()

    def initialize_optimizers(self):
        self.v_dW = [np.zeros_like(w) for w in self.model.w]
        self.v_db = [np.zeros_like(b) for b in self.model.b]
        self.s_dW = [np.zeros_like(w) for w in self.model.w]
        self.s_db = [np.zeros_like(b) for b in self.model.b]
    
    def backward_pass(self, X, Y):
        m = X.shape[0]
        activations, Z_values = self.model.forward_pass(X)
        dA = activations[-1] - Y  # Softmax + Cross-Entropy gradient
        dW, db = [], []
        
        for i in reversed(range(len(self.model.w))):
            dZ = dA if i == len(self.model.w) - 1 else dA * self.model.activation_derivative(activations[i + 1])
            dW.insert(0, np.dot(activations[i].T, dZ) / m)
            db.insert(0, np.sum(dZ, axis=0, keepdims=True) / m)
            dA = np.dot(dZ, self.model.w[i].T)
        return dW, db
    
    def update_parameters(self, dW, db):
        self.t += 1
        for i in range(len(self.model.w)):
            if self.method == "sgd":
                self.model.w[i] -= self.lr * dW[i]
                self.model.b[i] -= self.lr * db[i]
            
            elif self.method == "momentum":
                self.v_dW[i] = self.beta1 * self.v_dW[i] + (1 - self.beta1) * dW[i]
                self.v_db[i] = self.beta1 * self.v_db[i] + (1 - self.beta1) * db[i]
                self.model.w[i] -= self.lr * self.v_dW[i]
                self.model.b[i] -= self.lr * self.v_db[i]
                
            elif self.method == "nestrov":
                # Nesterov Accelerated Gradient (NAG)
                prev_v_dW = self.v_dW[i].copy()
                prev_v_db = self.v_db[i].copy()
                
                self.v_dW[i] = self.beta1 * self.v_dW[i] - self.lr * dW[i]
                self.v_db[i] = self.beta1 * self.v_db[i] - self.lr * db[i]
                
                self.model.w[i] += -self.beta1 * prev_v_dW + (1 + self.beta1) * self.v_dW[i]
                self.model.b[i] += -self.beta1 * prev_v_db + (1 + self.beta1) * self.v_db[i]
            
            elif self.method == "rmsprop":
                self.s_dW[i] = self.beta2 * self.s_dW[i] + (1 - self.beta2) * (dW[i] ** 2)
                self.s_db[i] = self.beta2 * self.s_db[i] + (1 - self.beta2) * (db[i] ** 2)
                self.model.w[i] -= self.lr * dW[i] / (np.sqrt(self.s_dW[i]) + self.epsilon)
                self.model.b[i] -= self.lr * db[i] / (np.sqrt(self.s_db[i]) + self.epsilon)
            
            elif self.method in ["adam", "nadam"]:
                self.v_dW[i] = self.beta1 * self.v_dW[i] + (1 - self.beta1) * dW[i]
                self.v_db[i] = self.beta1 * self.v_db[i] + (1 - self.beta1) * db[i]
                
                self.s_dW[i] = self.beta2 * self.s_dW[i] + (1 - self.beta2) * (dW[i] ** 2)
                self.s_db[i] = self.beta2 * self.s_db[i] + (1 - self.beta2) * (db[i] ** 2)
                
                v_dW_corr = self.v_dW[i] / (1 - self.beta1 ** self.t)
                v_db_corr = self.v_db[i] / (1 - self.beta1 ** self.t)
                s_dW_corr = self.s_dW[i] / (1 - self.beta2 ** self.t)
                s_db_corr = self.s_db[i] / (1 - self.beta2 ** self.t)
                
                self.model.w[i] -= self.lr * v_dW_corr / (np.sqrt(s_dW_corr) + self.epsilon)
                self.model.b[i] -= self.lr * v_db_corr / (np.sqrt(s_db_corr) + self.epsilon)

# Load Fashion MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train.reshape(-1, 28 * 28) / 255.0, x_test.reshape(-1, 28 * 28) / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Define model and training with variable hidden layers
layer_sizes = [784, 512, 256, 64, 10]  
model = FeedforwardNN(layer_sizes, activation="relu")
trainer = Optimizer(model, method="nestrov", lr=0.001)

# Training loop
epochs = 10
batch_size = 64

for epoch in range(epochs):
    indices = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[indices], y_train[indices]
    
    for i in range(0, x_train.shape[0], batch_size):
        X_batch, Y_batch = x_train[i:i + batch_size], y_train[i:i + batch_size]
        dW, db = trainer.backward_pass(X_batch, Y_batch)
        trainer.update_parameters(dW, db)
    
    predictions = np.argmax(model.forward_pass(x_test)[0][-1], axis=1)
    labels = np.argmax(y_test, axis=1)
    acc = np.mean(predictions == labels)
    print(f"Epoch {epoch + 1}/{epochs} - Test Accuracy: {acc:.4f}")
