import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## 1. Chargement et Prétraitement des Données RGB
def load_rgb_image(image_path, target_size=(32, 32)):
    """Charge et prétraite une image RGB"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Conversion BGR vers RGB
    img = cv2.resize(img, target_size)
    return img.astype(np.float32) / 255.0  # Normalisation [0,1]

# Chargement du dataset
data_dir = 'amhcd-data-rgb/tifinagh-images/'
labels_df = pd.read_csv(os.path.join(data_dir, 'labels-map.csv'))

# Encodage des labels
label_encoder = LabelEncoder()
labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])
num_classes = len(label_encoder.classes_)

# Chargement des images avec vérification
X = []
valid_indices = []
for idx, path in enumerate(labels_df['image_path']):
    try:
        img = load_rgb_image(os.path.join(data_dir, path))
        X.append(img)
        valid_indices.append(idx)
    except Exception as e:
        print(f"Erreur avec {path}: {str(e)}")

X = np.array(X)
y = labels_df.iloc[valid_indices]['label_encoded'].values

# Conversion one-hot
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_one_hot = one_hot_encoder.fit_transform(y.reshape(-1, 1))

## 2. Augmentation des Données
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

## 3. Architecture du Modèle avec Régularisation L2
class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.001, lambda_reg=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.weights = []
        self.biases = []
        
        # Initialisation des paramètres
        np.random.seed(42)
        for i in range(len(layer_sizes)-1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
    
    def forward(self, X):
        self.activations = [X.reshape(X.shape[0], -1)]  # Aplatissement des images RGB
        self.z_values = []
        
        for i in range(len(self.weights)-1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            self.activations.append(np.maximum(0, z))  # ReLU
            
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.activations.append(exp_z / np.sum(exp_z, axis=1, keepdims=True))
        return self.activations[-1]
    
    def compute_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
        cross_entropy = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        l2_penalty = sum(np.sum(w**2) for w in self.weights) * (self.lambda_reg/(2*y_true.shape[0]))
        return cross_entropy + l2_penalty
    
    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        
        dZ = y_pred - y_true
        grads_w[-1] = (self.activations[-2].T @ dZ)/m + (self.lambda_reg/m)*self.weights[-1]
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True)/m
        
        for l in range(len(self.weights)-2, -1, -1):
            dZ = (dZ @ self.weights[l+1].T) * (self.z_values[l] > 0)
            grads_w[l] = (self.activations[l].T @ dZ)/m + (self.lambda_reg/m)*self.weights[l]
            grads_b[l] = np.sum(dZ, axis=0, keepdims=True)/m
        
        return grads_w, grads_b
    
    def update_params(self, grads_w, grads_b, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if not hasattr(self, 'm_w'):
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]
        
        for i in range(len(self.weights)):
            # Mise à jour des moments pour Adam
            self.m_w[i] = beta1*self.m_w[i] + (1-beta1)*grads_w[i]
            self.v_w[i] = beta2*self.v_w[i] + (1-beta2)*(grads_w[i]**2)
            self.m_b[i] = beta1*self.m_b[i] + (1-beta1)*grads_b[i]
            self.v_b[i] = beta2*self.v_b[i] + (1-beta2)*(grads_b[i]**2)
            
            # Correction de biais
            m_w_hat = self.m_w[i]/(1-beta1**t)
            v_w_hat = self.v_w[i]/(1-beta2**t)
            m_b_hat = self.m_b[i]/(1-beta1**t)
            v_b_hat = self.v_b[i]/(1-beta2**t)
            
            # Mise à jour des paramètres
            self.weights[i] -= self.learning_rate * m_w_hat/(np.sqrt(v_w_hat)+epsilon)
            self.biases[i] -= self.learning_rate * m_b_hat/(np.sqrt(v_b_hat)+epsilon)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(1, epochs+1):
            # Entraînement par batch avec augmentation
            epoch_loss = 0
            for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_batch, y_pred)
                grads_w, grads_b = self.backward(X_batch, y_batch, y_pred)
                self.update_params(grads_w, grads_b, epoch)
                epoch_loss += loss
            
            # Évaluation
            train_pred = self.forward(X_train)
            train_loss = self.compute_loss(y_train, train_pred)
            train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
            
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f} | Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        return train_losses, val_losses, train_accs, val_accs

## 4. Entraînement avec Validation Croisée
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Configuration du modèle
layer_sizes = [32*32*3, 128, 64, num_classes]  # 3072 entrées pour RGB
model = MultiClassNeuralNetwork(layer_sizes, learning_rate=0.001, lambda_reg=0.01)

# Entraînement
history = model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=64)

## 5. Évaluation et Visualisation
# Matrice de confusion
y_pred = model.forward(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Matrice de Confusion (Test Set)')
plt.savefig('confusion_matrix_rgb.png')

# Courbes d'apprentissage
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history[0], label='Train')
plt.plot(history[1], label='Validation')
plt.title('Courbe de Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history[2], label='Train')
plt.plot(history[3], label='Validation')
plt.title('Courbe de Précision')
plt.legend()

plt.savefig('training_curves_rgb.png')