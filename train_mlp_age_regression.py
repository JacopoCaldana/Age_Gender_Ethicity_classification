
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List

import numpy as np
from mlp.data import load_age_regression_csv, StandardScaler, ScalarStandardizer, train_test_split
from mlp.cv import grid_search_cv, build_hidden_layers
from mlp.model import MLPRegressor
from mlp.metrics import mse, mape
from mlp.plotting import plot_training_history, plot_scatter_pred_vs_true, plot_cv_bars
from mlp.report_utils import save_report_tables, print_summary


def parse_args():
    p = argparse.ArgumentParser(description="Age Regression with manual MLP + SciPy optimizer")
    p.add_argument("--data", type=str, required=True, help="Path to AGE_REGRESSION.csv")
    p.add_argument("--k", type=int, default=5, help="Number of folds for CV")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (held out)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--max-iter", type=int, default=800, help="Max iterations for SciPy optimizer")
    p.add_argument("--tol", type=float, default=1e-6, help="Gradient tolerance for SciPy optimizer")
    p.add_argument("--scale-y", action="store_true", help="Standardize target y during training")
    p.add_argument("--outdir", type=str, default="runs/age_mlp", help="Output directory for artifacts")
    # Grid defaults: sensible but small; adjust as needed
    p.add_argument("--L", type=int, nargs="+", default=[2,3,4], help="Total layers (hidden + output)")
    p.add_argument("--width", type=int, nargs="+", default=[64,128,256], help="Hidden layer width candidates")
    p.add_argument("--l2", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2], help="L2 regularization candidates")
    p.add_argument("--activation", type=str, nargs="+", default=["tanh", "relu"], help="Activation candidates")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    X, y = load_age_regression_csv(args.data)
    n, d = X.shape

    # Split train/test
    tr_idx, te_idx = train_test_split(n, test_size=args.test_size, seed=args.seed)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    # Standardize features (always recommended)
    xsc = StandardScaler().fit(Xtr)
    Xtr_s = xsc.transform(Xtr)
    Xte_s = xsc.transform(Xte)

    # Optionally standardize y for optimization stability
    if args.scale_y:
        ysc = ScalarStandardizer().fit(ytr)
        ytr_s = ysc.transform(ytr)
        # CV and training will use scaled y, but evaluation will inverse-transform
        y_eval_inv = lambda yp: ysc.inverse_transform(yp)
    else:
        ysc = None
        ytr_s = ytr
        y_eval_inv = lambda yp: yp

    # Grid search with K-fold CV on training set
    param_grid = {"L": args.L, "width": args.width, "l2": args.l2, "activation": args.activation}
    cv_out = grid_search_cv(
        X=Xtr_s, y=ytr_s, input_dim=d, k=args.k,
        param_grid=param_grid, seed=args.seed,
        max_iter=args.max_iter, tol=args.tol
    )

    # Save and plot CV results
    with open(os.path.join(args.outdir, "cv_results.json"), "w") as f:
        json.dump(cv_out, f, indent=2)
    plot_cv_bars(cv_out["results"], os.path.join(args.outdir, "cv_results.png"))

    best = cv_out["best"]["params"]
    print("Best hyperparameters (by avg val MAPE):", best, "with avg MAPE =", cv_out["best"]["mape"])

    # Train final model on all training data with best params
    hidden = build_hidden_layers(best["L"], best["width"])
    model = MLPRegressor(
        input_dim=d,
        hidden_layers=hidden,
        activation=best["activation"],
        l2=best["l2"],
        seed=args.seed,
    )
    
    # Get initial predictions for initial MAPE (before training)
    # Initialize model with random weights
    theta_init = model._init_theta()
    model.theta_ = theta_init
    ytr_pred_init_s = model.predict(Xtr_s)
    ytr_pred_init = y_eval_inv(ytr_pred_init_s)
    initial_train_mape = mape(ytr, ytr_pred_init)
    initial_train_mse = mse(ytr, ytr_pred_init)
    
    # Now fit the model
    model.fit(Xtr_s, ytr_s, max_iter=args.max_iter, tol=args.tol, method="L-BFGS-B", verbose=False)
    plot_training_history(model.history_, os.path.join(args.outdir, "train_history.png"))

    # Evaluate on train and test in original y units
    ytr_pred_s = model.predict(Xtr_s)
    yte_pred_s = model.predict(Xte_s)
    ytr_pred = y_eval_inv(ytr_pred_s)
    yte_pred = y_eval_inv(yte_pred_s)

    train_metrics = {"mse": mse(ytr, ytr_pred), "mape": mape(ytr, ytr_pred)}
    test_metrics = {"mse": mse(yte, yte_pred), "mape": mape(yte, yte_pred)}
    initial_metrics = {"mse": initial_train_mse, "mape": initial_train_mape}

    with open(os.path.join(args.outdir, "final_metrics.json"), "w") as f:
        json.dump({
            "train": train_metrics,
            "test": test_metrics,
            "initial": initial_metrics,
            "best_params": best,
            "validation": {"mape": cv_out["best"]["mape"]},
        }, f, indent=2)

    print("Initial train metrics:", initial_metrics)
    print("Final train metrics:", train_metrics)
    print("Test metrics:", test_metrics)

    # Catchy plots
    plot_scatter_pred_vs_true(ytr, ytr_pred, os.path.join(args.outdir, "scatter_train.png"), title="Train: Pred vs True")
    plot_scatter_pred_vs_true(yte, yte_pred, os.path.join(args.outdir, "scatter_test.png"), title="Test: Pred vs True")
    
    # Generate report tables
    print("\nGenerating report tables...")
    save_report_tables(
        metrics_file=os.path.join(args.outdir, "final_metrics.json"),
        cv_results_file=os.path.join(args.outdir, "cv_results.json"),
        output_file=os.path.join(args.outdir, "report_tables.txt"),
    )
    print_summary(
        metrics_file=os.path.join(args.outdir, "final_metrics.json"),
        cv_results_file=os.path.join(args.outdir, "cv_results.json"),
    )


if __name__ == "__main__":
    main()