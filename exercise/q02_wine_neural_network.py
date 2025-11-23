"""
Implement 2 different feed forward neural network models to classify the wine dataset.
Evaluate their performance using accuracy and F1 score.
Provide a brief comparison of their results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
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

print("=" * 70)
print("WINE CLASSIFICATION WITH FEED-FORWARD NEURAL NETWORKS")
print("=" * 70)
print(f"\nDataset shape: {X.shape}")
print(f"Training set size: {X_train_scaled.shape[0]}")
print(f"Testing set size: {X_test_scaled.shape[0]}")
print(f"Number of features: {X_train_scaled.shape[1]}")
print(f"Number of classes: {len(np.unique(y_train))}")

# ============================================================================
# MODEL 1: Simple Feed-Forward Neural Network (3 layers)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 1: Simple Feed-Forward Neural Network")
print("=" * 70)
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
print("\n" + "=" * 70)
print("MODEL 2: Deeper Feed-Forward Neural Network with Dropout")
print("=" * 70)
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

# ============================================================================
# COMPARISON AND ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON OF THE TWO MODELS")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score (weighted)', 'Parameters', 'Layers'],
    'Model 1 (Simple)': [f"{accuracy1:.4f}", f"{f1_1:.4f}", 
                         model1.count_params(), 
                         len(model1.layers)],
    'Model 2 (Deep+Dropout)': [f"{accuracy2:.4f}", f"{f1_2:.4f}", 
                               model2.count_params(), 
                               len(model2.layers)]
})

print("\n" + comparison_df.to_string(index=False))

print("\n" + "=" * 70)
print("ANALYSIS AND CONCLUSIONS")
print("=" * 70)

print(f"\n1. ACCURACY COMPARISON:")
if accuracy1 > accuracy2:
    diff = (accuracy1 - accuracy2) * 100
    print(f"   Model 1 (Simple) is BETTER by {diff:.2f}%")
elif accuracy2 > accuracy1:
    diff = (accuracy2 - accuracy1) * 100
    print(f"   Model 2 (Deep+Dropout) is BETTER by {diff:.2f}%")
else:
    print(f"   Both models have EQUAL accuracy")

print(f"\n2. F1 SCORE COMPARISON:")
if f1_1 > f1_2:
    diff = (f1_1 - f1_2) * 100
    print(f"   Model 1 (Simple) is BETTER by {diff:.2f}%")
elif f1_2 > f1_1:
    diff = (f1_2 - f1_1) * 100
    print(f"   Model 2 (Deep+Dropout) is BETTER by {diff:.2f}%")
else:
    print(f"   Both models have EQUAL F1 score")

print(f"\n3. MODEL COMPLEXITY:")
print(f"   Model 1 has {model1.count_params():,} parameters")
print(f"   Model 2 has {model2.count_params():,} parameters")
print(f"   Model 2 is {(model2.count_params() / model1.count_params() - 1) * 100:.1f}% more complex")

print(f"\n4. ARCHITECTURE INSIGHTS:")
print(f"   Model 1: Simple 3-layer network (good baseline)")
print(f"   Model 2: Deeper 5-layer network with dropout (regularization)")
print(f"   - Dropout reduces overfitting by randomly deactivating neurons during training")
print(f"   - Larger hidden layers (128, 64) capture more complex patterns")

print(f"\n5. KEY OBSERVATIONS:")
best_model = "Model 1 (Simple)" if accuracy1 >= accuracy2 else "Model 2 (Deep+Dropout)"
print(f"   - Best performing model: {best_model}")
print(f"   - Model 2's dropout helps prevent overfitting on small datasets")
print(f"   - Wine dataset is relatively small (178 samples), so simpler models may generalize better")

# Plot training history comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Model 1 Training History
axes[0].plot(history1.history['loss'], label='Training Loss', marker='o', markersize=3)
axes[0].plot(history1.history['val_loss'], label='Validation Loss', marker='s', markersize=3)
axes[0].set_title('Model 1 (Simple) - Training History')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Model 2 Training History
axes[1].plot(history2.history['loss'], label='Training Loss', marker='o', markersize=3)
axes[1].plot(history2.history['val_loss'], label='Validation Loss', marker='s', markersize=3)
axes[1].set_title('Model 2 (Deep+Dropout) - Training History')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_comparison.png', dpi=100, bbox_inches='tight')
print("\n✓ Training history plot saved as 'training_history_comparison.png'")
plt.show()

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)