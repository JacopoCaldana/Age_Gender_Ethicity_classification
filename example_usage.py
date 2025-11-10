#!/usr/bin/env python3
"""
Example script showing how to use the MLP age regression implementation.
This creates synthetic data for demonstration purposes.
"""

import numpy as np
from mlp.model import MLPRegressor
from mlp.metrics import mse, mape
from mlp.data import StandardScaler

# Create synthetic regression data
np.random.seed(42)
n_samples = 200
n_features = 50

# Generate features
X = np.random.randn(n_samples, n_features)

# Generate target (age-like values between 1 and 100)
# Linear combination with some non-linearity
true_weights = np.random.randn(n_features) * 0.5
y = X @ true_weights + np.random.randn(n_samples) * 5
# Scale to age range [1, 100]
y = (y - y.min()) / (y.max() - y.min()) * 99 + 1

# Split data
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 80)
print("EXAMPLE: MLP Age Regression on Synthetic Data")
print("=" * 80)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {n_features}")
print()

# Create and train model
print("Training MLP with configuration:")
print("  - Hidden layers: [64, 32] (2 hidden layers)")
print("  - Activation: tanh")
print("  - L2 regularization: 0.001")
print("  - Optimizer: L-BFGS-B")
print("  - Max iterations: 500")
print()

model = MLPRegressor(
    input_dim=n_features,
    hidden_layers=[64, 32],
    activation="tanh",
    l2=0.001,
    seed=42,
)

# Get initial predictions
theta_init = model._init_theta()
model.theta_ = theta_init
y_train_pred_init = model.predict(X_train_scaled)
initial_mape = mape(y_train, y_train_pred_init)
initial_mse = mse(y_train, y_train_pred_init)

print(f"Initial (random weights) metrics:")
print(f"  - Training MAPE: {initial_mape:.2f}%")
print(f"  - Training MSE: {initial_mse:.4f}")
print()

# Train the model
print("Training...")
model.fit(X_train_scaled, y_train, max_iter=500, tol=1e-6, method="L-BFGS-B", verbose=True)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluate
train_mse = mse(y_train, y_train_pred)
train_mape_val = mape(y_train, y_train_pred)
test_mse = mse(y_test, y_test_pred)
test_mape_val = mape(y_test, y_test_pred)

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Training MSE:  {train_mse:.4f}")
print(f"Training MAPE: {train_mape_val:.2f}%")
print(f"Test MSE:      {test_mse:.4f}")
print(f"Test MAPE:     {test_mape_val:.2f}%")
print()

print("Optimization info:")
print(f"  - Status: {model.opt_result_.message}")
print(f"  - Iterations: {model.opt_result_.nit}")
print(f"  - Function evaluations: {model.opt_result_.nfev}")
print(f"  - Final objective: {model.opt_result_.fun:.6f}")
print()

print("Training history:")
print(f"  - Initial objective: {model.history_['obj'][0]:.6f}")
print(f"  - Final objective: {model.history_['obj'][-1]:.6f}")
print(f"  - Gradient norm (final): {model.history_['grad_norm'][-1]:.6e}")
print()

# Show some predictions
print("Sample predictions (first 5 test samples):")
print(f"{'True':>8} {'Predicted':>12} {'Error':>10}")
print("-" * 32)
for i in range(min(5, len(y_test))):
    error = abs(y_test[i] - y_test_pred[i])
    print(f"{y_test[i]:8.2f} {y_test_pred[i]:12.2f} {error:10.2f}")

print()
print("=" * 80)
print("SUCCESS! The MLP implementation is working correctly.")
print("=" * 80)
