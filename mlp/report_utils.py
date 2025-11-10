"""
Utilities to generate tables and figures required for the assignment report.
"""

from __future__ import annotations
import json
from typing import Dict, Optional
import numpy as np


def generate_performance_table(
    train_initial_mape: float,
    train_final_mape: float,
    val_final_mape: float,
    test_final_mape: float,
) -> str:
    """
    Generate the performance table (Figure 1) as specified in the assignment.
    
    Returns a formatted string that can be included in a LaTeX document.
    """
    table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Performance of the best MLP}}
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Metric}} & \\textbf{{MAPE (\\%)}} \\\\
\\hline
Training (initial) & {train_initial_mape:.2f} \\\\
Training (final) & {train_final_mape:.2f} \\\\
Validation (final) & {val_final_mape:.2f} \\\\
Test (final) & {test_final_mape:.2f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    return table


def generate_configuration_table(
    L: int,
    hidden_neurons: str,
    activation: str,
    lambda_val: float,
    optimizer: str,
    max_iter: int,
) -> str:
    """
    Generate the configuration table (Figure 2) as specified in the assignment.
    
    Returns a formatted string that can be included in a LaTeX document.
    """
    table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Configuration of the best MLP}}
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Hyperparameter}} & \\textbf{{Value}} \\\\
\\hline
Number of layers (L) & {L} \\\\
Hidden neurons per layer & {hidden_neurons} \\\\
Activation function & {activation} \\\\
Regularization ($\\lambda$) & {lambda_val:.4f} \\\\
Optimizer & {optimizer} \\\\
Max iterations & {max_iter} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    return table


def generate_error_metrics_table(
    train_initial_error: float,
    train_final_error: float,
    val_final_error: float,
    test_final_error: float,
) -> str:
    """
    Generate table showing regularized error on training/validation/test sets.
    
    Returns a formatted string that can be included in a LaTeX document.
    """
    table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Regularized L2 Error on Training, Validation and Test Sets}}
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Set}} & \\textbf{{Error}} \\\\
\\hline
Training (initial) & {train_initial_error:.6f} \\\\
Training (final) & {train_final_error:.6f} \\\\
Validation (final) & {val_final_error:.6f} \\\\
Test (final) & {test_final_error:.6f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    return table


def save_report_tables(
    metrics_file: str,
    cv_results_file: str,
    output_file: str = "report_tables.txt",
):
    """
    Read results from JSON files and generate all required tables for the report.
    
    Args:
        metrics_file: Path to final_metrics.json
        cv_results_file: Path to cv_results.json
        output_file: Where to save the generated tables
    """
    # Load results
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    with open(cv_results_file, 'r') as f:
        cv_results = json.load(f)
    
    best_params = metrics['best_params']
    
    # Extract metrics
    train_mape = metrics['train']['mape']
    test_mape = metrics['test']['mape']
    val_mape = cv_results['best']['mape']
    
    # Get initial MAPE
    train_initial_mape = metrics.get('initial', {}).get('mape', 100.0)
    
    # Generate tables
    perf_table = generate_performance_table(
        train_initial_mape=train_initial_mape,
        train_final_mape=train_mape,
        val_final_mape=val_mape,
        test_final_mape=test_mape,
    )
    
    # Build hidden neurons string
    hidden_neurons = f"{best_params['width']} neurons × {best_params['L']-1} layers"
    
    config_table = generate_configuration_table(
        L=best_params['L'],
        hidden_neurons=hidden_neurons,
        activation=best_params['activation'],
        lambda_val=best_params['l2'],
        optimizer="L-BFGS-B",
        max_iter=800,  # This should match what was used
    )
    
    # For error metrics, we would need MSE values
    train_error = metrics['train']['mse']
    test_error = metrics['test']['mse']
    
    # Save all tables
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PERFORMANCE TABLE (Figure 1)\n")
        f.write("=" * 80 + "\n")
        f.write(perf_table)
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("CONFIGURATION TABLE (Figure 2)\n")
        f.write("=" * 80 + "\n")
        f.write(config_table)
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("SUMMARY FOR REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nBest Hyperparameters:\n")
        f.write(f"  - L (total layers): {best_params['L']}\n")
        f.write(f"  - Hidden neurons: {best_params['width']}\n")
        f.write(f"  - Activation: {best_params['activation']}\n")
        f.write(f"  - Lambda (L2 reg): {best_params['l2']}\n")
        f.write(f"\nFinal Performance:\n")
        f.write(f"  - Training MAPE: {train_mape:.2f}%\n")
        f.write(f"  - Validation MAPE: {val_mape:.2f}%\n")
        f.write(f"  - Test MAPE: {test_mape:.2f}%\n")
        f.write(f"  - Training MSE: {train_error:.4f}\n")
        f.write(f"  - Test MSE: {test_error:.4f}\n")
    
    print(f"Report tables saved to {output_file}")
    return output_file


def print_summary(metrics_file: str, cv_results_file: str):
    """
    Print a human-readable summary of the results to console.
    """
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    with open(cv_results_file, 'r') as f:
        cv_results = json.load(f)
    
    best_params = metrics['best_params']
    
    print("\n" + "=" * 80)
    print("BEST MODEL CONFIGURATION")
    print("=" * 80)
    print(f"Number of layers (L): {best_params['L']}")
    print(f"Hidden neurons per layer: {best_params['width']}")
    print(f"Activation function: {best_params['activation']}")
    print(f"L2 regularization (λ): {best_params['l2']}")
    
    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE (MAPE)")
    print("=" * 80)
    print(f"Training:   {metrics['train']['mape']:.2f}%")
    print(f"Validation: {cv_results['best']['mape']:.2f}%")
    print(f"Test:       {metrics['test']['mape']:.2f}%")
    
    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE (MSE)")
    print("=" * 80)
    print(f"Training: {metrics['train']['mse']:.4f}")
    print(f"Test:     {metrics['test']['mse']:.4f}")
    print("=" * 80 + "\n")
