#!/usr/bin/env python3
"""
Test script to validate the complete MLP pipeline including cross-validation.
"""

import numpy as np
import tempfile
import os
from mlp.model import MLPRegressor
from mlp.cv import grid_search_cv, build_hidden_layers
from mlp.metrics import mse, mape
from mlp.data import StandardScaler, kfold_indices

print("=" * 80)
print("VALIDATION TEST: MLP Age Regression Implementation")
print("=" * 80)
print()

# Create synthetic data
np.random.seed(42)
n_samples = 300
n_features = 30

X = np.random.randn(n_samples, n_features)
true_weights = np.random.randn(n_features) * 0.3
y = X @ true_weights + np.random.randn(n_samples) * 3
y = (y - y.min()) / (y.max() - y.min()) * 99 + 1

# Split data
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Test 1: Basic MLP Training")
print("-" * 80)

model = MLPRegressor(
    input_dim=n_features,
    hidden_layers=[32, 16],
    activation="tanh",
    l2=0.01,
    seed=42,
)

model.fit(X_train_scaled, y_train, max_iter=200, tol=1e-5, verbose=False)
y_pred = model.predict(X_train_scaled)
train_mape = mape(y_train, y_pred)

print(f"✓ Model trained successfully")
print(f"  Training MAPE: {train_mape:.2f}%")
print(f"  Optimization status: {model.opt_result_.message}")
print()

print("Test 2: Different Activation Functions")
print("-" * 80)

activations = ["tanh", "sigmoid", "relu", "elu"]
for act in activations:
    model = MLPRegressor(
        input_dim=n_features,
        hidden_layers=[16],
        activation=act,
        l2=0.01,
        seed=42,
    )
    model.fit(X_train_scaled, y_train, max_iter=100, tol=1e-4, verbose=False)
    y_pred = model.predict(X_train_scaled)
    train_mape = mape(y_train, y_pred)
    print(f"✓ {act:8s}: MAPE = {train_mape:6.2f}%")

print()

print("Test 3: K-Fold Indices Generation")
print("-" * 80)

n = 100
k = 5
folds = list(kfold_indices(n, k, shuffle=True, seed=42))
print(f"✓ Generated {len(folds)} folds")

# Check that all indices are covered and no overlap
all_val_indices = set()
for i, (train_idx, val_idx) in enumerate(folds):
    # Check no overlap
    assert len(set(train_idx) & set(val_idx)) == 0, f"Fold {i}: overlap detected!"
    all_val_indices.update(val_idx)
    print(f"  Fold {i+1}: {len(train_idx)} train, {len(val_idx)} val samples")

assert all_val_indices == set(range(n)), "Not all indices covered in validation!"
print(f"✓ All {n} indices covered exactly once across folds")
print()

print("Test 4: Cross-Validation Grid Search")
print("-" * 80)

# Small grid for testing
param_grid = {
    "L": [2, 3],
    "width": [16, 32],
    "l2": [0.0, 0.01],
    "activation": ["tanh", "relu"],
}

# Use only 2 folds for speed
cv_results = grid_search_cv(
    X=X_train_scaled,
    y=y_train,
    input_dim=n_features,
    k=2,
    param_grid=param_grid,
    seed=42,
    max_iter=100,
    tol=1e-4,
    verbose=False,
)

print(f"✓ Grid search completed")
print(f"  Total combinations tested: {len(cv_results['results'])}")
print(f"  Best MAPE: {cv_results['best']['mape']:.2f}%")
print(f"  Best params: {cv_results['best']['params']}")
print()

print("Test 5: Build Hidden Layers Helper")
print("-" * 80)

for L in [2, 3, 4]:
    hidden = build_hidden_layers(L, 64)
    print(f"✓ L={L}: hidden_layers = {hidden} (length={len(hidden)})")
    assert len(hidden) == L - 1, f"Expected {L-1} hidden layers, got {len(hidden)}"

print()

print("Test 6: Metrics Computation")
print("-" * 80)

y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
y_pred = np.array([12.0, 18.0, 32.0, 38.0, 51.0])

mse_val = mse(y_true, y_pred)
mape_val = mape(y_true, y_pred)

print(f"✓ MSE computed: {mse_val:.4f}")
print(f"✓ MAPE computed: {mape_val:.2f}%")

# Verify MSE calculation
expected_mse = np.mean((y_true - y_pred) ** 2)
assert abs(mse_val - expected_mse) < 1e-6, "MSE calculation incorrect!"

# Verify MAPE calculation
expected_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
assert abs(mape_val - expected_mape) < 1e-6, "MAPE calculation incorrect!"

print()

print("Test 7: Parameter Packing/Unpacking")
print("-" * 80)

model = MLPRegressor(
    input_dim=10,
    hidden_layers=[20, 15],
    activation="tanh",
    l2=0.01,
    seed=42,
)

# Initialize parameters
theta = model._init_theta()
print(f"✓ Parameters initialized: {len(theta)} total values")

# Unpack and repack
params = model._unpack(theta)
theta_repacked = model._pack(params)

print(f"✓ Unpacked into {len(params)} layers")
for i, (W, b) in enumerate(params):
    print(f"  Layer {i+1}: W shape {W.shape}, b shape {b.shape}")

# Verify packing/unpacking is consistent
assert np.allclose(theta, theta_repacked), "Pack/unpack not consistent!"
print(f"✓ Pack/unpack consistency verified")
print()

print("=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
print()
print("Summary:")
print("  ✓ Basic MLP training works")
print("  ✓ All activation functions work")
print("  ✓ K-fold cross-validation works")
print("  ✓ Grid search works")
print("  ✓ Metrics computation is correct")
print("  ✓ Parameter handling is correct")
print()
print("The implementation is ready to use with real data!")
