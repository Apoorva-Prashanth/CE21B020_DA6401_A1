import numpy as np
import wandb
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Xavier Initialization
def xavier_init(size_in, size_out):
    return np.random.randn(size_in, size_out) * np.sqrt(1 / size_in)

# Loss Function: Categorical Cross-Entropy
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]


# Neural Network Class
class FeedforwardNN:
    def __init__(self, layer_sizes, activation="relu", weight_init="random", weight_decay=0):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weight_decay = weight_decay  # L2 Regularization
        self.initialize_weights(weight_init)

    def initialize_weights(self, weight_init):
        if weight_init == "xavier":
            self.w = [xavier_init(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)]
        else:  # Default to random
            self.w = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01 for i in range(len(self.layer_sizes) - 1)]
        
        self.b = [np.zeros((1, self.layer_sizes[i + 1])) for i in range(len(self.layer_sizes) - 1)]

    def activation_func(self, x):
        if self.activation == "relu":
            return relu(x)
        elif self.activation == "sigmoid":
            return sigmoid(x)
        elif self.activation == "tanh":
            return tanh(x)

    def activation_derivative(self, x):
        if self.activation == "relu":
            return relu_derivative(x)
        elif self.activation == "sigmoid":
            return sigmoid_derivative(x)
        elif self.activation == "tanh":
            return tanh_derivative(x)

    def forward_pass(self, X):
        activations = [X]
        Z_values = []
        for i in range(len(self.w)):
            Z = np.dot(activations[-1], self.w[i]) + self.b[i]
            Z_values.append(Z)
            if i < len(self.w) - 1:
                A = self.activation_func(Z)
            else:
                A = self.softmax(Z)
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
            dW.insert(0, np.dot(activations[i].T, dZ) / m + self.model.weight_decay * self.model.w[i])
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

# Load Fashion MNIST Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train.reshape(-1, 28 * 28) / 255.0, x_test.reshape(-1, 28 * 28) / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Create validation split (10% of train)
val_split = int(0.1 * x_train.shape[0])
x_val, y_val = x_train[:val_split], y_train[:val_split]
x_train, y_train = x_train[val_split:], y_train[val_split:]



# Training Function
def train(args):
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    config = wandb.config
    config.epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.optimizer = args.optimizer
    config.num_hidden_layers = args.num_layers
    config.hidden_layer_size = args.hidden_size
    config.activation = args.activation
    config.weight_init = args.weight_init
    config.weight_decay = args.weight_decay

    run_name = f"hl_{config.num_hidden_layers}_bs_{args.batch_size}_ac_{config.activation}"
    
    wandb.run.name = run_name
    wandb.run.save()

    layer_sizes = [784] + [config.hidden_layer_size] * config.num_hidden_layers + [10]
    model = FeedforwardNN(layer_sizes, activation=config.activation, weight_init=config.weight_init, weight_decay=config.weight_decay)

    trainer = Optimizer(model, method=config.optimizer, lr=config.learning_rate)

    for epoch in range(config.epochs):
        indices = np.random.permutation(x_train.shape[0])
        x_train_shuffled, y_train_shuffled = x_train[indices], y_train[indices]

        for i in range(0, x_train.shape[0], args.batch_size):
            X_batch, Y_batch = x_train_shuffled[i:i + args.batch_size], y_train_shuffled[i:i + args.batch_size]
            dW, db = trainer.backward_pass(X_batch, Y_batch)
            trainer.update_parameters(dW, db)

        # Evaluate on train, validation, and test sets
        train_preds = model.forward_pass(x_train)[0][-1]
        val_preds = model.forward_pass(x_val)[0][-1]
        test_preds = model.forward_pass(x_test)[0][-1]

        train_loss = cross_entropy_loss(y_train, train_preds)
        val_loss = cross_entropy_loss(y_val, val_preds)

        train_acc = np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train, axis=1))
        val_acc = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1))
        test_acc = np.mean(np.argmax(test_preds, axis=1) == np.argmax(y_test, axis=1))


        wandb.log({"training_loss": train_loss, "validation_loss": val_loss, "training_accuracy": train_acc, "validation_accuracy": val_acc, "test_accuracy": test_acc, "epoch": epoch + 1})
        print(f"Epoch {epoch+1}/{config.epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, required=True)
    parser.add_argument('--wandb_entity', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='nadam')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--weight_init', type=str, default='xavier')
    parser.add_argument('--weight_decay', type=float, default=0.0)

    args = parser.parse_args()
    train(args)
