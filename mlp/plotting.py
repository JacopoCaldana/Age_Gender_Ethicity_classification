from __future__ import annotations
import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_training_history(history: Dict[str, list], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(history["obj"], label="Objective", color="tab:blue")
    ax1.set_xlabel("Callback evaluations")
    ax1.set_ylabel("Objective", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    if "grad_norm" in history and len(history["grad_norm"]) == len(history["obj"]):
        ax2 = ax1.twinx()
        ax2.plot(history["grad_norm"], label="Grad norm", color="tab:orange")
        ax2.set_ylabel("Grad norm", color="tab:orange")
        ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title="Predicted vs True"):
    ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("True age")
    ax.set_ylabel("Predicted age")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cv_bars(results: List[Dict], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    # Sort by avg_val_mape ascending
    results_sorted = sorted(results, key=lambda r: r["avg_val_mape"])
    labels = [f"L={r['L']},w={r['width']},λ={r['l2']},act={r['activation']}" for r in results_sorted]
    mape_vals = [r["avg_val_mape"] for r in results_sorted]

    fig, ax = plt.subplots(figsize=(min(10, max(6, 0.4*len(labels))), 4))
    ax.bar(range(len(labels)), mape_vals)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha='right', fontsize=8)
    ax.set_ylabel("Avg Val MAPE (%)")
    ax.set_title("K-fold CV results (lower is better)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)