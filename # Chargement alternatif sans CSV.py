import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List

# Fonctions d'activation
def relu(x):
    """
    ReLU activation: max(0, x)
    """
    assert isinstance(x, np.ndarray), "Input to ReLU must be a numpy array"
    result = np.maximum(0, x)
    assert np.all(result >= 0), "ReLU output must be non-negative"
    return result

def relu_derivative(x):
    """
    Derivative of ReLU: 1 if x > 0, else 0
    """
    assert isinstance(x, np.ndarray), "Input to ReLU derivative must be a numpy array"
    result = (x > 0).astype(float)
    assert np.all((result == 0) | (result == 1)), "ReLU derivative must be 0 or 1"
    return result

def softmax(x):
    """
    Softmax activation: exp(x) / sum(exp(x))
    """
    assert isinstance(x, np.ndarray), "Input to softmax must be a numpy array"
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
    result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    assert np.all((result >= 0) & (result <= 1)), "Softmax output must be in [0, 1]"
    assert np.allclose(np.sum(result, axis=1), 1), "Softmax output must sum to 1 per sample"
    return result

# Data augmentation functions
def augment_image(img, target_size=(32, 32)):
    """
    Apply data augmentation to an image
    """
    # Original image
    augmented = [img]
    
    # Horizontal flip
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)
    
    # Small rotation (-10 to 10 degrees)
    angle = np.random.uniform(-10, 10)
    center = (target_size[0] // 2, target_size[1] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, target_size)
    augmented.append(rotated)
    
    # Small translation
    tx, ty = np.random.randint(-3, 4, 2)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, translation_matrix, target_size)
    augmented.append(translated)
    
    return augmented

# Classe MultiClassNeuralNetwork avec améliorations
class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, lambda_reg=0.01, use_adam=True):
        """
        Initialize the neural network with given layer sizes and learning rate.
        layer_sizes: List of integers [input_size, hidden1_size, ..., output_size]
        lambda_reg: L2 regularization parameter
        use_adam: Whether to use Adam optimizer
        """
        assert isinstance(layer_sizes, list) and len(layer_sizes) >= 2, "layer_sizes must be a list with at least 2 elements"
        assert all(isinstance(size, int) and size > 0 for size in layer_sizes), "All layer sizes must be positive integers"
        assert isinstance(learning_rate, (int, float)) and learning_rate > 0, "Learning rate must be a positive number"

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.use_adam = use_adam
        self.weights = []
        self.biases = []

        # Adam optimizer parameters
        if self.use_adam:
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.m_weights = []
            self.v_weights = []
            self.m_biases = []
            self.v_biases = []
            self.t = 0  # time step

        # Initialisation des poids et biais (Xavier initialization)
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization for better convergence
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            assert w.shape == (layer_sizes[i], layer_sizes[i+1]), f"Weight matrix {i+1} has incorrect shape"
            assert b.shape == (1, layer_sizes[i+1]), f"Bias vector {i+1} has incorrect shape"
            self.weights.append(w)
            self.biases.append(b)
            
            if self.use_adam:
                self.m_weights.append(np.zeros_like(w))
                self.v_weights.append(np.zeros_like(w))
                self.m_biases.append(np.zeros_like(b))
                self.v_biases.append(np.zeros_like(b))

    def forward(self, X):
        """
        Forward propagation: Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}, A^{[l]} = g(Z^{[l]})
        """
        assert isinstance(X, np.ndarray), "Input X must be a numpy array"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"

        self.activations = [X]
        self.z_values = []

        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = self.activations[i] @ self.weights[i] + self.biases[i]
            assert z.shape == (X.shape[0], self.layer_sizes[i+1]), f"Z^{[i+1]} has incorrect shape"
            self.z_values.append(z)
            self.activations.append(relu(z))

        # Output layer with softmax
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        assert z.shape == (X.shape[0], self.layer_sizes[-1]), "Output Z has incorrect shape"
        self.z_values.append(z)
        output = softmax(z)
        assert output.shape == (X.shape[0], self.layer_sizes[-1]), "Output A has incorrect shape"
        self.activations.append(output)

        return self.activations[-1]

    def compute_loss(self, y_true, y_pred):
        """
        Categorical Cross-Entropy with L2 regularization: J = -1/m * sum(y_true * log(y_pred)) + lambda/2m * sum(W^2)
        """
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "Inputs to loss must be numpy arrays"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cross_entropy_loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        # L2 regularization term
        l2_loss = 0
        for w in self.weights:
            l2_loss += np.sum(w ** 2)
        l2_loss = (self.lambda_reg / (2 * m)) * l2_loss
        
        total_loss = cross_entropy_loss + l2_loss
        assert not np.isnan(total_loss), "Loss computation resulted in NaN"
        return total_loss

    def compute_accuracy(self, y_true, y_pred):
        """
        Compute accuracy: proportion of correct predictions
        """
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "Inputs to accuracy must be numpy arrays"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(predictions == true_labels)
        assert 0 <= accuracy <= 1, "Accuracy must be between 0 and 1"
        return accuracy

    def backward(self, X, y, outputs):
        """
        Backpropagation with L2 regularization and Adam optimizer
        """
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray) and isinstance(outputs, np.ndarray), "Inputs to backward must be numpy arrays"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y.shape == outputs.shape, "y and outputs must have the same shape"

        m = X.shape[0]
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]

        # Output layer gradient (softmax + cross-entropy)
        dZ = outputs - y
        assert dZ.shape == outputs.shape, "dZ for output layer has incorrect shape"
        self.d_weights[-1] = (self.activations[-2].T @ dZ) / m
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m

        # Hidden layers gradients
        for i in range(len(self.weights) - 2, -1, -1):
            dZ = (dZ @ self.weights[i+1].T) * relu_derivative(self.z_values[i])
            assert dZ.shape == (X.shape[0], self.layer_sizes[i+1]), f"dZ^{[i+1]} has incorrect shape"
            self.d_weights[i] = (self.activations[i].T @ dZ) / m
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m

        # Add L2 regularization to weight gradients
        for i in range(len(self.weights)):
            self.d_weights[i] += (self.lambda_reg / m) * self.weights[i]

        # Update parameters
        if self.use_adam:
            self._adam_update()
        else:
            self._sgd_update()

    def _sgd_update(self):
        """Standard SGD parameter update"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def _adam_update(self):
        """Adam optimizer parameter update"""
        self.t += 1
        
        for i in range(len(self.weights)):
            # Update biased first moment estimate
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * self.d_weights[i]
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * self.d_biases[i]
            
            # Update biased second moment estimate
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (self.d_weights[i] ** 2)
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (self.d_biases[i] ** 2)
            
            # Compute bias-corrected first and second moment estimates
            m_w_hat = self.m_weights[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_biases[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_weights[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_biases[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def train(self, X, y, X_val, y_val, epochs, batch_size, early_stopping_patience=10):
        """
        Train the neural network using mini-batch SGD with early stopping
        """
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y must be numpy arrays"
        assert isinstance(X_val, np.ndarray) and isinstance(y_val, np.ndarray), "X_val and y_val must be numpy arrays"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y.shape[1] == self.layer_sizes[-1], f"Output dimension ({y.shape[1]}) must match output layer size ({self.layer_sizes[-1]})"
        assert X_val.shape[1] == self.layer_sizes[0], f"Validation input dimension ({X_val.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y_val.shape[1] == self.layer_sizes[-1], f"Validation output dimension ({y_val.shape[1]}) must match output layer size ({self.layer_sizes[-1]})"
        assert isinstance(epochs, int) and epochs > 0, "Epochs must be a positive integer"
        assert isinstance(batch_size, int) and batch_size > 0, "Batch size must be a positive integer"

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                outputs = self.forward(X_batch)
                epoch_loss += self.compute_loss(y_batch, outputs)
                self.backward(X_batch, y_batch, outputs)
                num_batches += 1

            # Calculate metrics
            train_loss = epoch_loss / num_batches
            train_pred = self.forward(X)
            train_accuracy = self.compute_accuracy(y, train_pred)
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_accuracy = self.compute_accuracy(y_val, val_pred)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        """
        Predict class labels
        """
        assert isinstance(X, np.ndarray), "Input X must be a numpy array"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"

        outputs = self.forward(X)
        predictions = np.argmax(outputs, axis=1)
        assert predictions.shape == (X.shape[0],), "Predictions have incorrect shape"
        return predictions

    def cross_validate(self, X, y, k_folds=5, epochs=50, batch_size=32):
        """
        Perform k-fold cross-validation
        """
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Training fold {fold + 1}/{k_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Reset network for each fold
            self.__init__(self.layer_sizes, self.learning_rate, self.lambda_reg, self.use_adam)
            
            # Train on this fold
            _, _, _, val_accuracies = self.train(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                epochs, batch_size, early_stopping_patience=5
            )
            
            # Get best validation accuracy
            best_acc = max(val_accuracies)
            cv_scores.append(best_acc)
            print(f"Fold {fold + 1} best accuracy: {best_acc:.4f}")
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        print(f"Cross-validation results: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        
        return cv_scores

def load_and_preprocess_image(image_path, target_size=(32, 32)):
    """
    Load and preprocess an image: convert to grayscale, resize, normalize
    """
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to load image: {image_path}"
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0  # Normalization
    return img.flatten()  # Flatten for neural network

def load_data_with_augmentation(data_dir, labels_df, use_augmentation=True):
    """
    Load and optionally augment the dataset
    """
    X = []
    y = []
    
    for idx, row in labels_df.iterrows():
        image_path = os.path.join(data_dir, row['image_path'])
        
        if os.path.exists(image_path):
            # Load original image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (32, 32))
                
                if use_augmentation:
                    # Apply data augmentation
                    augmented_images = augment_image(img)
                    for aug_img in augmented_images:
                        aug_img = aug_img.astype(np.float32) / 255.0
                        X.append(aug_img.flatten())
                        y.append(row['label_encoded'])
                else:
                    # Just original image
                    img = img.astype(np.float32) / 255.0
                    X.append(img.flatten())
                    y.append(row['label_encoded'])
    
    return np.array(X), np.array(y)

# Main execution function
def main():
    # Configuration
    TARGET_SIZE = (32, 32)
    USE_AUGMENTATION = True
    USE_ADAM = True
    LAMBDA_REG = 0.01
    LEARNING_RATE = 0.001
    EPOCHS = 100
    BATCH_SIZE = 32
    
    # Define path to decompressed folder
    data_dir = os.path.join(os.getcwd(), 'amhcd-data-64/tifinagh-images/')
    print(f"Data directory: {data_dir}")
    
    # Load CSV file containing labels
    try:
        labels_df = pd.read_csv(os.path.join(data_dir, 'amhcd-data-64/labels-map.csv'))
        assert 'image_path' in labels_df.columns and 'label' in labels_df.columns, "CSV must contain 'image_path' and 'label' columns"
    except FileNotFoundError:
        print("labels-map.csv not found. Building DataFrame from folders...")
        image_paths = []
        labels = []
        for label_dir in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_dir)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    image_paths.append(os.path.join(label_path, img_name))
                    labels.append(label_dir)
        labels_df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    
    # Verify DataFrame
    assert not labels_df.empty, "No data loaded. Check dataset files."
    print(f"Loaded {len(labels_df)} samples with {labels_df['label'].nunique()} unique classes.")
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    X, y = load_data_with_augmentation(data_dir, labels_df, USE_AUGMENTATION)
    
    # Verify dimensions
    assert X.shape[0] == y.shape[0], "Mismatch between number of images and labels"
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    
    print(f"Train: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # One-hot encode labels
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    y_train_one_hot = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_val_one_hot = one_hot_encoder.transform(y_val.reshape(-1, 1))
    y_test_one_hot = one_hot_encoder.transform(y_test.reshape(-1, 1))
    
    # Create and train model
    layer_sizes = [X_train.shape[1], 128, 64, num_classes]
    print(f"Network architecture: {layer_sizes}")
    
    nn = MultiClassNeuralNetwork(
        layer_sizes, 
        learning_rate=LEARNING_RATE, 
        lambda_reg=LAMBDA_REG, 
        use_adam=USE_ADAM
    )
    
    print("Training neural network...")
    train_losses, val_losses, train_accuracies, val_accuracies = nn.train(
        X_train, y_train_one_hot, X_val, y_val_one_hot, 
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )
    
    # Test set evaluation
    print("\nEvaluating on test set...")
    y_pred = nn.predict(X_test)
    test_accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\nClassification Report (Test set):")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix (Test set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curve
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(train_accuracies, label='Train Accuracy', linewidth=2)
    ax2.plot(val_accuracies, label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Optional: Perform cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = nn.cross_validate(X_train, y_train_one_hot, k_folds=5, epochs=50, batch_size=32)
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"CV Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    print(f"Best Validation Accuracy: {max(val_accuracies):.4f}")

if __name__ == "__main__":
    main()