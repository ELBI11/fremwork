import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Fonctions d'activation
def relu(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU: 1 if x > 0, else 0"""
    return (x > 0).astype(float)

def softmax(x):
    """Softmax activation: exp(x) / sum(exp(x))"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, lambda_reg=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.lambda_reg = lambda_reg
        
        # Initialize weights and biases
        np.random.seed(42)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01 
                       for i in range(len(layer_sizes)-1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) 
                      for i in range(len(layer_sizes)-1)]
        
        # Adam optimizer states
        self.m_W = [np.zeros_like(w) for w in self.weights]
        self.v_W = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        # Hidden layers with ReLU
        for i in range(len(self.weights)-1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            self.activations.append(relu(z))
        
        # Output layer with softmax
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        output = softmax(z)
        self.activations.append(output)
        
        return output

    def compute_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cross_entropy = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        # L2 regularization
        l2_penalty = 0
        for w in self.weights:
            l2_penalty += np.sum(w ** 2)
        l2_penalty = (self.lambda_reg / (2 * y_true.shape[0])) * l2_penalty
        
        return cross_entropy + l2_penalty

    def compute_accuracy(self, y_true, y_pred):
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == true_labels)

    def backward(self, X, y, outputs):
        m = X.shape[0]
        grads_W = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        dZ = outputs - y
        grads_W[-1] = (self.activations[-2].T @ dZ) / m
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Hidden layers gradients
        for i in range(len(self.weights)-2, -1, -1):
            dZ = (dZ @ self.weights[i+1].T) * relu_derivative(self.z_values[i])
            grads_W[i] = (self.activations[i].T @ dZ) / m
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Add L2 regularization term
        for i in range(len(grads_W)):
            grads_W[i] += (self.lambda_reg / m) * self.weights[i]
        
        # Adam optimizer update
        self.t += 1
        for i in range(len(self.weights)):
            # Update moments for weights
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grads_W[i]
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (grads_W[i] ** 2)
            
            # Update moments for biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)
            
            # Bias correction
            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            self.weights[i] -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def train(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32, patience=5):
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_loss = np.inf
        best_weights, best_biases = None, None
        wait = 0
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                outputs = self.forward(X_batch)
                epoch_loss += self.compute_loss(y_batch, outputs)
                self.backward(X_batch, y_batch, outputs)
            
            # Calculate metrics
            train_loss = epoch_loss / (X.shape[0] // batch_size)
            train_pred = self.forward(X)
            train_acc = self.compute_accuracy(y, train_pred)
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                val_acc = self.compute_accuracy(y_val, val_pred)
                
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        self.weights, self.biases = best_weights, best_biases
                        break
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        return train_losses, val_losses, train_accs, val_accs

    def predict(self, X):
        outputs = self.forward(X)
        return np.argmax(outputs, axis=1)

# Fonction pour charger et prétraiter les images avec augmentation
def load_and_preprocess_image(image_path, target_size=(32, 32), augment=True):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    
    if augment and np.random.rand() > 0.5:  # 50% chance to augment
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((16, 16), angle, 1)
        img = cv2.warpAffine(img, M, (32, 32))
        
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
    
    img = img.astype(np.float32) / 255.0
    return img.flatten()

# Chargement des données
data_dir = 'amhcd-data-64/tifinagh-images/'
try:
    labels_df = pd.read_csv(os.path.join(data_dir, 'labels-map.csv'))
except FileNotFoundError:
    image_paths = []
    labels = []
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, img_name))
                labels.append(label_dir)
    labels_df = pd.DataFrame({'image_path': image_paths, 'label': labels})

# Encodage des labels
label_encoder = LabelEncoder()
labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])
num_classes = len(label_encoder.classes_)

# Chargement et prétraitement des images
X = np.array([load_and_preprocess_image(os.path.join(data_dir, path), augment=False) 
              for path in labels_df['image_path']])
y = labels_df['label_encoded'].values

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

# One-hot encoding
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train_one_hot = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
y_val_one_hot = one_hot_encoder.transform(y_val.reshape(-1, 1))
y_test_one_hot = one_hot_encoder.transform(y_test.reshape(-1, 1))

# K-Fold Validation
print("\nK-Fold Cross Validation:")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\nFold {fold+1}")
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train_one_hot[train_idx], y_train_one_hot[val_idx]
    
    model = MultiClassNeuralNetwork([1024, 64, 32, num_classes], learning_rate=0.001, lambda_reg=0.01)
    _, _, _, val_acc = model.train(X_fold_train, y_fold_train, X_fold_val, y_fold_val, 
                                 epochs=50, batch_size=64, patience=3)
    fold_accuracies.append(np.mean(val_acc[-5:]))
    print(f"Fold {fold+1} Val Accuracy: {fold_accuracies[-1]:.4f}")

print(f"\nMean K-Fold Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

# Entraînement final
print("\nTraining Final Model...")
final_model = MultiClassNeuralNetwork([1024, 64, 32, num_classes], learning_rate=0.001, lambda_reg=0.01)
train_losses, val_losses, train_accs, val_accs = final_model.train(
    X_train, y_train_one_hot, X_val, y_val_one_hot, 
    epochs=100, batch_size=64, patience=5
)

# Évaluation
test_pred = final_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=label_encoder.classes_))

# Matrice de confusion
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.close()

# Courbes d'apprentissage
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png', bbox_inches='tight')
plt.close()

print("\nTraining completed! Check confusion_matrix.png and training_curves.png for results.")