from __future__ import annotations
import itertools
import time
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

from data import kfold_indices
from model import MLPRegressor
from metrics import mape, mse

Array = np.ndarray

#Dopo un primo tentativo di prova con cv fast si è notato un andamento costante del MAPE il che ha fatto sorgere il sosspetto che stessimo
# calcolando la mape su y standardizzata, per cui ora riportiamo alla scala normale con inverse

def build_hidden_layers(L: int, width: int) -> List[int]:
    assert 2 <= L <= 4
    return [width] * (L - 1)


def _compute_metrics(
    y_true: Array,
    y_pred: Array,
    y_inverse: Optional[Callable[[Array], Array]] = None,
) -> Tuple[float, float]:
    """
    Calcola MSE e MAPE. Se y_inverse è fornito, riporta prima y e y_hat alla scala originale.
    """
    if y_inverse is not None:
        y_true = y_inverse(y_true)
        y_pred = y_inverse(y_pred)
    return float(mse(y_true, y_pred)), float(mape(y_true, y_pred))


def _evaluate_combo(
    X: Array,
    y: Array,
    input_dim: int,
    combo: Tuple[int, int, float, str],  # (L, width, l2, activation)
    folds: List[Tuple[np.ndarray, np.ndarray]],
    max_iter: int,
    tol: float,
    seed: int,
    y_inverse: Optional[Callable[[Array], Array]] = None,
) -> Dict:
    """
    Valuta una singola combinazione di iperparametri sui fold forniti.
    Le metriche (MSE, MAPE) sono calcolate su scala originale se y_inverse è passato.
    """
    L, width, l2, activation = combo
    fold_metrics: List[Dict[str, float]] = []

    for fi, (tr_idx, va_idx) in enumerate(folds, 1):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        model = MLPRegressor(
            input_dim=input_dim,
            hidden_layers=build_hidden_layers(int(L), int(width)),
            activation=str(activation),
            l2=float(l2),
            seed=seed + fi,
        )
        model.fit(Xtr, ytr, max_iter=max_iter, tol=tol, method="L-BFGS-B", verbose=False)
        yva_pred = model.predict(Xva)

        mse_v, mape_v = _compute_metrics(yva, yva_pred, y_inverse=y_inverse)
        fold_metrics.append({"mse": mse_v, "mape": mape_v})

    avg_mape = float(np.mean([fm["mape"] for fm in fold_metrics]))
    avg_mse = float(np.mean([fm["mse"] for fm in fold_metrics]))
    return {
        "L": int(L),
        "width": int(width),
        "l2": float(l2),
        "activation": str(activation),
        "avg_val_mape": avg_mape,
        "avg_val_mse": avg_mse,
    }


def grid_search_cv(
    X: Array,
    y: Array,
    input_dim: int,
    k: int,
    param_grid: Dict,
    seed: int = 0,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = True,
    folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    y_inverse: Optional[Callable[[Array], Array]] = None,
) -> Dict:
    """
    Grid search classica: valuta tutte le combinazioni su K fold.
    Se y_inverse è fornito, le metriche (MSE, MAPE) sono calcolate su scala originale.
    """
    # Validazione griglia
    keys = ["L", "width", "l2", "activation"]
    for key in keys:
        if key not in param_grid:
            raise ValueError(f"Missing '{key}' in param_grid")

    combos = list(
        itertools.product(
            param_grid["L"], param_grid["width"], param_grid["l2"], param_grid["activation"]
        )
    )

    # Folds precomputati per riproducibilità e velocità
    if folds is None:
        folds = list(kfold_indices(len(X), k, shuffle=True, seed=seed))

    results: List[Dict] = []
    best = {"mape": np.inf, "params": None}

    t0 = time.time()
    for ci, combo in enumerate(combos, 1):
        if verbose:
            L, width, l2, activation = combo
            print(f"[{ci}/{len(combos)}] L={L}, w={width}, λ={l2}, act={activation}", flush=True)

        row = _evaluate_combo(
            X=X,
            y=y,
            input_dim=input_dim,
            combo=combo,
            folds=folds,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            y_inverse=y_inverse,
        )
        results.append(row)

        if row["avg_val_mape"] < best["mape"]:
            best = {
                "mape": row["avg_val_mape"],
                "params": {
                    "L": row["L"],
                    "width": row["width"],
                    "l2": row["l2"],
                    "activation": row["activation"],
                },
            }

        if verbose:
            elapsed = time.time() - t0
            eta = elapsed / ci * (len(combos) - ci)
            print(
                f"    avg MAPE={row['avg_val_mape']:.3f} | elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m",
                flush=True,
            )

    return {"best": best, "results": results}


def grid_search_cv_fast(
    X: Array,
    y: Array,
    input_dim: int,
    k_full: int,
    param_grid: Dict,
    seed: int = 0,
    # Stage 1 (cheap)
    k_stage1: Optional[int] = 3,
    max_iter_stage1: int = 200,
    # Stage 2 (refine)
    top_n: int = 5,
    max_iter_stage2: int = 800,
    tol: float = 1e-6,
    verbose: bool = True,
    y_inverse: Optional[Callable[[Array], Array]] = None,
) -> Dict:
    """
    Grid search in 2 fasi:
      - Stage 1: tutte le combinazioni con k ridotto e max_iter ridotto
      - Stage 2: solo le top-N con k pieno e max_iter pieno
    Le metriche sono su scala originale se y_inverse è passato.
    """
    # Combinazioni
    keys = ["L", "width", "l2", "activation"]
    for key in keys:
        if key not in param_grid:
            raise ValueError(f"Missing '{key}' in param_grid")
    combos = list(
        itertools.product(
            param_grid["L"], param_grid["width"], param_grid["l2"], param_grid["activation"]
        )
    )

    # Stage 1
    if k_stage1 is None or k_stage1 <= 0:
        k_stage1 = min(3, k_full)
    folds_stage1 = list(kfold_indices(len(X), k_stage1, shuffle=True, seed=seed))

    stage1_results: List[Dict] = []
    t0 = time.time()
    if verbose:
        print(
            f"FAST Stage1: {len(combos)} combos, k={k_stage1}, max_iter={max_iter_stage1}",
            flush=True,
        )

    for ci, combo in enumerate(combos, 1):
        if verbose:
            L, width, l2, activation = combo
            print(
                f"[S1 {ci}/{len(combos)}] L={L}, w={width}, λ={l2}, act={activation}",
                flush=True,
            )
        row = _evaluate_combo(
            X=X,
            y=y,
            input_dim=input_dim,
            combo=combo,
            folds=folds_stage1,
            max_iter=max_iter_stage1,
            tol=tol,
            seed=seed,
            y_inverse=y_inverse,
        )
        stage1_results.append(row)
        if verbose:
            elapsed = time.time() - t0
            eta = elapsed / ci * (len(combos) - ci)
            print(
                f"    S1 avg MAPE={row['avg_val_mape']:.3f} | elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m",
                flush=True,
            )

    # Ordina e scegli finalisti
    stage1_sorted = sorted(stage1_results, key=lambda r: r["avg_val_mape"])
    finalists = stage1_sorted[: min(top_n, len(stage1_sorted))]

    # Stage 2
    folds_full = list(kfold_indices(len(X), k_full, shuffle=True, seed=seed))
    stage2_results: List[Dict] = []
    if verbose:
        print(
            f"FAST Stage2: refining top-{len(finalists)} with k={k_full}, max_iter={max_iter_stage2}",
            flush=True,
        )
    t1 = time.time()
    for ci, row in enumerate(finalists, 1):
        combo = (row["L"], row["width"], row["l2"], row["activation"])
        if verbose:
            L, width, l2, activation = combo
            print(f"[S2 {ci}/{len(finalists)}] L={L}, w={width}, λ={l2}, act={activation}", flush=True)
        row2 = _evaluate_combo(
            X=X,
            y=y,
            input_dim=input_dim,
            combo=combo,
            folds=folds_full,
            max_iter=max_iter_stage2,
            tol=tol,
            seed=seed,
            y_inverse=y_inverse,
        )
        stage2_results.append(row2)
        if verbose:
            elapsed = time.time() - t1
            eta = elapsed / ci * (len(finalists) - ci)
            print(
                f"    S2 avg MAPE={row2['avg_val_mape']:.3f} | elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m",
                flush=True,
            )

    # Risultati finali
    final_results = stage2_results if stage2_results else stage1_results
    best_row = min(final_results, key=lambda r: r["avg_val_mape"])
    best = {
        "mape": best_row["avg_val_mape"],
        "params": {
            "L": best_row["L"],
            "width": best_row["width"],
            "l2": best_row["l2"],
            "activation": best_row["activation"],
        },
    }

    return {"best": best, "results": final_results, "stage1_results": stage1_results}

