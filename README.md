# Age_Gender_Ethicity_classification
UTKFace Dataset

## Age Regression with Multi-Layer Perceptron

This repository implements a Multi-Layer Perceptron (MLP) for age regression using the UTKFace dataset, as part of an assignment on neural network optimization.

### Features

- **Custom MLP Implementation**: Fully-connected MLP built from scratch
- **Manual Backpropagation**: Gradients computed manually without automatic differentiation
- **SciPy Optimization**: Uses `scipy.optimize` (L-BFGS-B) for parameter optimization
- **L2 Regularization**: Prevents overfitting with configurable regularization strength
- **K-Fold Cross-Validation**: Automated hyperparameter search with cross-validation
- **Multiple Activations**: Supports tanh, sigmoid, ReLU, and ELU activation functions

### Installation

```bash
pip install numpy scipy pandas matplotlib
```

### Usage

Train the model with k-fold cross-validation:

```bash
python train_mlp_age_regression.py --data path/to/AGE_REGRESSION.csv --k 5 --outdir runs/age_mlp
```

#### Command Line Arguments

- `--data`: Path to the AGE_REGRESSION.csv dataset (required)
- `--k`: Number of folds for cross-validation (default: 5)
- `--test-size`: Fraction of data for test set (default: 0.2)
- `--seed`: Random seed for reproducibility (default: 42)
- `--max-iter`: Maximum iterations for optimizer (default: 800)
- `--tol`: Gradient tolerance for convergence (default: 1e-6)
- `--scale-y`: Standardize target variable during training
- `--outdir`: Output directory for results (default: runs/age_mlp)

#### Hyperparameter Grid

- `--L`: Number of layers including hidden + output (default: [2, 3, 4])
- `--width`: Hidden layer width (default: [64, 128, 256])
- `--l2`: L2 regularization values (default: [0.0, 1e-4, 1e-3, 1e-2])
- `--activation`: Activation functions (default: ["tanh", "relu"])

### Output

The training script generates:

1. **cv_results.json**: Cross-validation results for all hyperparameter combinations
2. **final_metrics.json**: Final performance metrics on train/validation/test sets
3. **train_history.png**: Training objective and gradient norm over iterations
4. **scatter_train.png**: Scatter plot of predicted vs true ages on training set
5. **scatter_test.png**: Scatter plot of predicted vs true ages on test set
6. **cv_results.png**: Bar chart comparing all CV configurations
7. **report_tables.txt**: LaTeX-formatted tables for the assignment report

### Assignment Requirements

This implementation satisfies the following requirements:

1. ✅ Neural network with at least 2 hidden layers (configurable 2-4)
2. ✅ L2 regularized loss function minimization
3. ✅ SciPy optimization routine (L-BFGS-B)
4. ✅ Manual gradient computation (no automatic differentiation)
5. ✅ K-fold cross-validation for hyperparameter selection
6. ✅ MAPE metric tracking on train/validation/test sets
7. ✅ Automatic generation of required tables for the report

### Project Structure

```
.
├── mlp/
│   ├── __init__.py          # Package initialization
│   ├── model.py             # MLPRegressor class
│   ├── activations.py       # Activation functions and weight initialization
│   ├── cv.py                # Cross-validation utilities
│   ├── data.py              # Data loading and preprocessing
│   ├── metrics.py           # Evaluation metrics (MSE, MAPE)
│   ├── plotting.py          # Visualization utilities
│   └── report_utils.py      # Report table generation
├── train_mlp_age_regression.py  # Main training script
└── README.md                # This file
```

### Example

```bash
# Run with default grid search
python train_mlp_age_regression.py \
    --data data/AGE_REGRESSION.csv \
    --k 5 \
    --max-iter 800 \
    --outdir results/experiment_1

# Run with custom hyperparameter grid
python train_mlp_age_regression.py \
    --data data/AGE_REGRESSION.csv \
    --L 3 4 \
    --width 128 256 512 \
    --l2 0.0 1e-3 1e-2 \
    --activation tanh sigmoid \
    --outdir results/experiment_2
```

### Report Tables

The script automatically generates LaTeX-formatted tables showing:
- Performance metrics (initial/final MAPE on train/validation/test)
- Best model configuration (L, neurons, activation, λ)
- Regularized error values

These can be directly copied into your assignment report PDF.
