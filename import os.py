import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Fonctions d'activation
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Classe du MLP avec régularisation L2 et optimiseur Adam
class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, lambda_reg=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.weights, self.biases = [], []

        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

        # Adam initialisation
        self.m_W = [np.zeros_like(w) for w in self.weights]
        self.v_W = [np.zeros_like(w) for w in self.weights]
        self.m_B = [np.zeros_like(b) for b in self.biases]
        self.v_B = [np.zeros_like(b) for b in self.biases]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def forward(self, X):
        self.activations, self.z_values = [X], []
        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            self.activations.append(relu(z))
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(softmax(z))
        return self.activations[-1]

    def compute_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        reg_term = (self.lambda_reg / 2) * sum(np.sum(w**2) for w in self.weights)
        return loss + reg_term

    def compute_accuracy(self, y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

    def backward(self, X, y, outputs):
        m = X.shape[0]
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]

        dZ = outputs - y
        self.d_weights[-1] = self.activations[-2].T @ dZ / m + (self.lambda_reg / m) * self.weights[-1]
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m

        for i in range(len(self.weights) - 2, -1, -1):
            dZ = (dZ @ self.weights[i+1].T) * relu_derivative(self.z_values[i])
            self.d_weights[i] = self.activations[i].T @ dZ / m + (self.lambda_reg / m) * self.weights[i]
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m

        self.t += 1
        for i in range(len(self.weights)):
            # Poids
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * self.d_weights[i]
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (self.d_weights[i] ** 2)
            m_hat_w = self.m_W[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_W[i] / (1 - self.beta2 ** self.t)
            self.weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

            # Biais
            self.m_B[i] = self.beta1 * self.m_B[i] + (1 - self.beta1) * self.d_biases[i]
            self.v_B[i] = self.beta2 * self.v_B[i] + (1 - self.beta2) * (self.d_biases[i] ** 2)
            m_hat_b = self.m_B[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_B[i] / (1 - self.beta2 ** self.t)
            self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled, y_shuffled = X[indices], y[indices]
            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                outputs = self.forward(X_batch)
                epoch_loss += self.compute_loss(y_batch, outputs)
                self.backward(X_batch, y_batch, outputs)
            train_pred = self.forward(X)
            val_pred = self.forward(X_val)
            train_losses.append(epoch_loss / (X.shape[0] // batch_size))
            val_losses.append(self.compute_loss(y_val, val_pred))
            train_accuracies.append(self.compute_accuracy(y, train_pred))
            val_accuracies.append(self.compute_accuracy(y_val, val_pred))
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")
        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
