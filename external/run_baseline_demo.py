"""
run_baseline_demo.py -- compare GNAR-edge against ARIMA / SARIMA / Naive baselines

This is the script a thesis chapter needs: every model goes through the SAME
expanding-window rolling backtest, so the numbers are directly comparable.

    python run_baseline_demo.py

By default it runs on simulated data so it works out of the box. Point `main()`
at your own panel (numpy array, DataFrame, Excel file, or an EdgeGraph) to run it
on real data -- see `load_my_data()` below.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from edge_graph import ArrayEdgeGraph
from edge_baselines import (ARIMAEdgeBaseline, NaiveEdgeBaseline,
                            SARIMAEdgeBaseline, compare_baselines, to_edge_graph)
from BaseEdgeGNAR_edge import GNAREdgeLearner
from BaseEdgeGNAR_edge_global import GNAREdgeGlobalLearner

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Data: swap this for your own panel
# ----------------------------------------------------------------------
def simulate_edge_graph(K: int = 6, T: int = 60, seed: int = 0) -> ArrayEdgeGraph:
    """Edge series generated with GNAR-like dynamics (own lag + neighbour mean)."""
    rng = np.random.default_rng(seed)
    edges = [(0, 0), (0, 1), (1, 2), (2, 1), (2, 3), (3, 4), (4, 5), (5, 0), (1, 3), (3, 1)]
    E = len(edges)
    nb = [[j for j, (a, b) in enumerate(edges) if j != i and ({u, v} & {a, b})]
          for i, (u, v) in enumerate(edges)]

    X = np.zeros((T, E))
    X[0] = rng.normal(size=E)
    for t in range(1, T):
        neigh = np.array([X[t - 1, nb[i]].mean() if nb[i] else 0.0 for i in range(E)])
        X[t] = 0.55 * X[t - 1] + 0.25 * neigh + 0.3 * rng.normal(size=E)

    months = list(pd.period_range("2020-01", periods=T, freq="M"))
    return ArrayEdgeGraph.from_edge_panel(X, edges, n_nodes=K, time_labels=months,
                                          node_labels=[f"N{i}" for i in range(K)])


def load_my_data():
    """
    Replace the body with your own data. Anything below works:

        return to_edge_graph("panel.xlsx")                       # Excel: rows=time, cols="u->v"
        return to_edge_graph(df)                                 # DataFrame
        return to_edge_graph(X, edges=edges, n_nodes=K,          # numpy (T, E)
                             time_labels=windows, node_labels=names)
        return to_edge_graph(V)                                  # numpy (T, K, K)
        return to_edge_graph(tsg)                                # legacy TSG / ipg
    """
    return simulate_edge_graph()


# ----------------------------------------------------------------------
def main():
    g = load_my_data()
    print(f"EdgeGraph: E={g.n_edges} edges, K={g.n_nodes} nodes, T={g.n_times} periods\n")

    # seasonal period in WINDOWS: 12 = monthly data with a yearly cycle,
    # 7 = daily data with a weekly cycle. Set to your own sampling.
    s = 12

    table = compare_baselines(
        g,
        models={
            "naive":                    (NaiveEdgeBaseline,     {}),
            "arima(1,0,0)":             (ARIMAEdgeBaseline,     dict(order=(1, 0, 0))),
            "arima(auto)":              (ARIMAEdgeBaseline,     dict(order="auto")),
            f"sarima(1,0,0)(1,0,0,{s})": (SARIMAEdgeBaseline,   dict(order=(1, 0, 0),
                                                                    seasonal_order=(1, 0, 0, s))),
            "gnar_global":              (GNAREdgeGlobalLearner, dict(L=1, stages_per_lag=[[1]],
                                                                    use_ols=True, verbose_fit=False)),
            "gnar_local":               (GNAREdgeLearner,       dict(L=1, stages_per_lag=[[1]],
                                                                    use_ols=True, verbose_fit=False)),
        },
        fit_kwargs_by_model={
            "gnar_global": dict(use_ols=True, verbose=False),
            "gnar_local":  dict(use_ols=True, verbose=False),
        },
        val_frac=0.2,          # last 20% of the time axis is the rolling test set
    )

    print("\nRolling one-step-ahead comparison (identical harness for every model)")
    print("=" * 78)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 78)
    print("skill_vs_reference = 1 - MAE_model / MAE_reference;  > 0 beats the reference.")
    print("Reference: 'naive' (persistence) for a LEVEL panel; use ZeroEdgeBaseline")
    print("and reference='zero' if your panel holds differences.")

    # per-model predictions are kept for further analysis / plots
    preds = table.attrs["predictions"]
    best = table.iloc[0]["model"]
    print(f"\nBest model: {best}")
    print(f"Its prediction frame: {preds[best]['pred_df'].shape} (test periods x edges)")


if __name__ == "__main__":
    main()
