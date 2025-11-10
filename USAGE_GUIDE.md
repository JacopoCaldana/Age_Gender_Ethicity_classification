# MLP Age Regression - Usage Guide

This guide explains how to use the MLP implementation for the age regression assignment.

## Quick Start

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy (>= 1.20.0)
- scipy (>= 1.7.0)
- pandas (>= 1.3.0)
- matplotlib (>= 3.4.0)

### 2. Verify Installation

Run the validation tests to ensure everything is working:

```bash
python test_implementation.py
```

You should see all tests passing with a ✓ mark.

### 3. Run with Your Data

Run the training script with your AGE_REGRESSION.csv file:

```bash
python train_mlp_age_regression.py --data path/to/AGE_REGRESSION.csv
```

## Command Line Options

### Required Arguments

- `--data PATH`: Path to the AGE_REGRESSION.csv file

### Optional Arguments

#### Cross-Validation Settings
- `--k INT`: Number of folds for cross-validation (default: 5)
- `--test-size FLOAT`: Fraction of data to hold out for testing (default: 0.2)
- `--seed INT`: Random seed for reproducibility (default: 42)

#### Optimization Settings
- `--max-iter INT`: Maximum iterations for scipy optimizer (default: 800)
- `--tol FLOAT`: Gradient tolerance for convergence (default: 1e-6)
- `--scale-y`: Flag to standardize target values during training

#### Output Settings
- `--outdir PATH`: Directory to save results (default: runs/age_mlp)

#### Hyperparameter Grid
- `--L INT [INT ...]`: Number of layers to try (default: 2 3 4)
- `--width INT [INT ...]`: Hidden layer widths to try (default: 64 128 256)
- `--l2 FLOAT [FLOAT ...]`: L2 regularization values (default: 0.0 1e-4 1e-3 1e-2)
- `--activation STR [STR ...]`: Activation functions (default: tanh relu)

## Example Usage

### Basic Usage

Run with default settings:

```bash
python train_mlp_age_regression.py --data data/AGE_REGRESSION.csv
```

### Custom Hyperparameter Grid

Try specific hyperparameters:

```bash
python train_mlp_age_regression.py \
    --data data/AGE_REGRESSION.csv \
    --L 3 4 \
    --width 128 256 512 \
    --l2 0.0 1e-3 1e-2 \
    --activation tanh sigmoid relu \
    --k 5 \
    --max-iter 1000
```

### Quick Testing

For quick testing with a smaller grid:

```bash
python train_mlp_age_regression.py \
    --data data/AGE_REGRESSION.csv \
    --L 2 \
    --width 64 \
    --l2 0.001 \
    --activation tanh \
    --k 3 \
    --max-iter 300
```

### With Y Standardization

Enable target standardization for potentially better optimization:

```bash
python train_mlp_age_regression.py \
    --data data/AGE_REGRESSION.csv \
    --scale-y
```

## Output Files

The script generates the following files in the output directory:

### Metrics and Results

1. **cv_results.json**: Cross-validation results for all hyperparameter combinations
   - Contains validation MAPE for each configuration
   - Best parameters identified

2. **final_metrics.json**: Final performance metrics
   - Initial training metrics (random weights)
   - Final training metrics
   - Test metrics
   - Best hyperparameters

3. **report_tables.txt**: LaTeX-formatted tables for the assignment report
   - Performance table (Figure 1)
   - Configuration table (Figure 2)
   - Summary of results

### Visualizations

4. **train_history.png**: Training objective and gradient norm over iterations

5. **scatter_train.png**: Scatter plot of predicted vs true ages on training set

6. **scatter_test.png**: Scatter plot of predicted vs true ages on test set

7. **cv_results.png**: Bar chart comparing all cross-validation configurations

## Understanding the Output

### Console Output

During training, you'll see:

```
Best hyperparameters (by avg val MAPE): {'L': 3, 'width': 128, 'l2': 0.001, 'activation': 'tanh'} with avg MAPE = 8.45

Initial train metrics: {'mse': 1234.56, 'mape': 98.76}
Final train metrics: {'mse': 23.45, 'mape': 7.89}
Test metrics: {'mse': 28.91, 'mape': 8.32}

Generating report tables...
Report tables saved to runs/age_mlp/report_tables.txt

================================================================================
BEST MODEL CONFIGURATION
================================================================================
Number of layers (L): 3
Hidden neurons per layer: 128
Activation function: tanh
L2 regularization (λ): 0.001

================================================================================
FINAL PERFORMANCE (MAPE)
================================================================================
Training:   7.89%
Validation: 8.45%
Test:       8.32%
```

### Interpreting Results

- **Lower MAPE is better**: Values below 10% are generally good for age prediction
- **Training vs Test**: Small gap indicates good generalization
- **Initial vs Final**: Large improvement shows effective optimization
- **Validation MAPE**: Used to select best hyperparameters

## Assignment Report

### Required Information

The assignment requires reporting:

1. **Hyperparameter Selection**
   - Final values of L, Nl (neurons), λ, activation
   - Justification for choices (based on CV results)

2. **Optimization Details**
   - Method: L-BFGS-B (from scipy.optimize)
   - Settings: max iterations, tolerance
   - Result: convergence status, iterations, objective values

3. **Performance Metrics**
   - Initial and final regularized error (MSE + L2 term)
   - Initial and final MAPE on training set
   - Final MAPE on validation set (average of k-folds)
   - Final MAPE on test set

### Using Generated Tables

The file `report_tables.txt` contains LaTeX-formatted tables that you can copy directly into your report. It includes:

- Figure 1: Performance table with MAPE values
- Figure 2: Configuration table with hyperparameters

### Writing the Report

Example structure:

```
1. Hyperparameter Selection
   - We tested L ∈ {2,3,4}, width ∈ {64,128,256}, λ ∈ {0, 10^-4, 10^-3, 10^-2}
   - Best configuration: L=3, width=128, λ=0.001, activation=tanh
   - Selected based on minimum average validation MAPE across 5 folds
   - Reasoning: This configuration balanced complexity and regularization

2. Optimization Setup
   - Method: L-BFGS-B from scipy.optimize
   - Max iterations: 800
   - Gradient tolerance: 10^-6
   - Result: Converged in X iterations
   - Initial objective: Y.YY
   - Final objective: Z.ZZ

3. Performance Results
   [Include Table 1: Performance metrics]
   [Include Table 2: Configuration]
```

## Tips for Better Results

1. **Start with a coarse grid**: Test broad ranges first
2. **Refine around best**: Once you find good hyperparameters, search nearby values
3. **Use more folds for final evaluation**: k=5 or k=10 gives more reliable estimates
4. **Monitor training history**: Check if optimization converged or hit iteration limit
5. **Compare multiple activations**: Different activations work better for different data
6. **Balance regularization**: Too much L2 underfits, too little overfits

## Troubleshooting

### Model doesn't converge

- Increase `--max-iter`
- Try different activations (tanh is often more stable than relu)
- Enable `--scale-y` to standardize targets
- Reduce network complexity (smaller L or width)

### Poor performance

- Try larger networks (more layers or neurons)
- Adjust L2 regularization
- Check if data is properly normalized (features should be standardized)
- Verify data quality (no missing values, correct format)

### Out of memory

- Reduce grid size (fewer hyperparameter combinations)
- Use smaller k (fewer folds)
- Process data in smaller batches (for very large datasets)

### Slow training

- Reduce `--max-iter` for initial exploration
- Use smaller k for initial grid search
- Reduce grid size (test fewer combinations)
- Use fast_cv option in code if doing extensive searches

## Advanced Usage

### Programmatic Usage

You can also use the MLP classes directly in your own scripts:

```python
from mlp.model import MLPRegressor
from mlp.data import StandardScaler
import numpy as np

# Load and prepare your data
X, y = load_data()  # Your data loading function
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Create and train model
model = MLPRegressor(
    input_dim=X.shape[1],
    hidden_layers=[128, 64],
    activation="tanh",
    l2=0.001,
    seed=42
)

model.fit(X_scaled, y, max_iter=800, tol=1e-6, verbose=True)

# Make predictions
y_pred = model.predict(X_scaled)
```

See `example_usage.py` for a complete example.

## Citation

If you use this implementation, please cite the assignment:

```
Age Regression with Multi-Layer Perceptron
UTKFace Dataset
Course: [Your Course Name]
```
