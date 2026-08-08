"""
run_simulated.py  —— Minimal example (1): run GNAR-edge on "simulated data" (no external data required)

Data is generated according to GNAR dynamics (each edge = its own lag + mean of adjacent edge lags + noise),
so the model should be able to fit it reasonably well. The whole pipeline depends only on the pure-algorithm files in this subfolder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edge_graph import ArrayEdgeGraph
from BaseEdgeGNAR_edge import GNAREdgeLearner
from RollingEdgePredict import RollingEdgePredict


def simulate_edge_graph(K: int = 6, T: int = 48, seed: int = 0) -> ArrayEdgeGraph:
    rng = np.random.default_rng(seed)
    edges = [(0, 0), (0, 1), (1, 2), (2, 1), (2, 3), (3, 4), (4, 5), (5, 0), (1, 3), (3, 1)]
    E = len(edges)

    # Edge-adjacency (sharing an endpoint makes them neighbours), used to build the neural auto-regressive signal
    nb = [[] for _ in range(E)]
    for i, (u, v) in enumerate(edges):
        for j, (a, b) in enumerate(edges):
            if j != i and ({u, v} & {a, b}):
                nb[i].append(j)

    X = np.zeros((T, E))
    X[0] = rng.normal(size=E)
    for t in range(1, T):
        neigh = np.array([X[t - 1, nb[i]].mean() if nb[i] else 0.0 for i in range(E)])
        X[t] = 0.6 * X[t - 1] + 0.25 * neigh + 0.3 * rng.normal(size=E)

    months = list(pd.period_range("2020-01", periods=T, freq="M"))
    return ArrayEdgeGraph.from_edge_panel(
        X, edges, n_nodes=K, time_labels=months, node_labels=[f"N{i}" for i in range(K)])


def main():
    g = simulate_edge_graph()
    print(f"simulated EdgeGraph: E={g.n_edges} K={g.n_nodes} T={g.n_times}")

    roller = RollingEdgePredict(
        graph=g,
        learner_cls=GNAREdgeLearner,
        train_periods=g.time_labels[:-6],
        val_periods=g.time_labels[-6:],
        learner_kwargs=dict(L=3, stages_per_lag=[[1], [1, 2], [1, 2]],
                            use_ols=True, device="cpu", verbose_fit=False),
        fit_kwargs=dict(use_ols=True, ols_rcond=1e-8, verbose=False),
        graph_builder=None,            # Fixed graph: A stays constant, not rebuilt
        strict_next_match=True,
    ).run()

    print(roller.summary().to_string(index=False))
    out = roller.to_prediction_dfs(edge_mode="intersection")
    print(f"mean R² over edges = {np.nanmean(out['r2_per_edge'].to_numpy()):.4f}")


if __name__ == "__main__":
    main()
