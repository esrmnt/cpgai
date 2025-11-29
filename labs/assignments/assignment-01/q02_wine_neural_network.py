import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the wine dataset
wine = load_wine(as_frame=True)
X = wine.frame.drop('target', axis=1)
y = wine.frame['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to numpy arrays for Keras
X_train_scaled = np.array(X_train_scaled)
X_test_scaled = np.array(X_test_scaled)
y_train = np.array(y_train)
y_test = np.array(y_test)


print("WINE CLASSIFICATION WITH FEED-FORWARD NEURAL NETWORKS")
print(f"\nDataset shape: {X.shape}")
print(f"Training set size: {X_train_scaled.shape[0]}")
print(f"Testing set size: {X_test_scaled.shape[0]}")
print(f"Number of features: {X_train_scaled.shape[1]}")
print(f"Number of classes: {len(np.unique(y_train))}")

# ============================================================================
# MODEL 1: Simple Feed-Forward Neural Network (3 layers)
# ============================================================================
print("MODEL 1: Simple Feed-Forward Neural Network")
print("Architecture: Input(13) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(3, Softmax)")

model1 = Sequential([
    layers.Input(shape=(13,)),
    layers.Dense(64, activation='relu', name='hidden_1'),
    layers.Dense(32, activation='relu', name='hidden_2'),
    layers.Dense(3, activation='softmax', name='output')
])

model1.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel 1 Summary:")
model1.summary()

# Train Model 1
print("\nTraining Model 1...")
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history1 = model1.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# Evaluate Model 1
y_pred_prob1 = model1.predict(X_test_scaled, verbose=0)
y_pred1 = np.argmax(y_pred_prob1, axis=1)

accuracy1 = accuracy_score(y_test, y_pred1)
f1_1 = f1_score(y_test, y_pred1, average='weighted')

print(f"\nModel 1 - Accuracy: {accuracy1:.4f}")
print(f"Model 1 - F1 Score (weighted): {f1_1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred1))

# ============================================================================
# MODEL 2: Deeper Feed-Forward Neural Network with Dropout (5 layers)
# ============================================================================
print("MODEL 2: Deeper Feed-Forward Neural Network with Dropout")
print("Architecture: Input(13) -> Dense(128, ReLU) -> Dropout(0.3) ->")
print("              Dense(64, ReLU) -> Dropout(0.3) -> Dense(32, ReLU) ->")
print("              Dropout(0.2) -> Output(3, Softmax)")

model2 = Sequential([
    layers.Input(shape=(13,)),
    layers.Dense(128, activation='relu', name='hidden_1'),
    layers.Dropout(0.3, name='dropout_1'),
    layers.Dense(64, activation='relu', name='hidden_2'),
    layers.Dropout(0.3, name='dropout_2'),
    layers.Dense(32, activation='relu', name='hidden_3'),
    layers.Dropout(0.2, name='dropout_3'),
    layers.Dense(3, activation='softmax', name='output')
])

model2.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel 2 Summary:")
model2.summary()

# Train Model 2
print("\nTraining Model 2...")
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history2 = model2.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# Evaluate Model 2
y_pred_prob2 = model2.predict(X_test_scaled, verbose=0)
y_pred2 = np.argmax(y_pred_prob2, axis=1)

accuracy2 = accuracy_score(y_test, y_pred2)
f1_2 = f1_score(y_test, y_pred2, average='weighted')

print(f"\nModel 2 - Accuracy: {accuracy2:.4f}")
print(f"Model 2 - F1 Score (weighted): {f1_2:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred2))
