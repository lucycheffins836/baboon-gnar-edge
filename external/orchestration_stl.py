"""
orchestration_stl.py  —— STL preprocessing in the orchestration layer
============================================

STL (seasonal-trend decomposition) is **data preprocessing**, not part of the
GNAR-edge algorithm, so it lives in the orchestration layer. The algorithm only
accepts an already-processed EdgeGraph.

Two usage options:
  A) Transform the entire series in one pass to obtain a new EdgeGraph, then feed it
     into the fixed-graph rolling backtest:
        Xs = stl_transform_panel(g.edge_series(), g.time_labels, ...)
        g_stl = ArrayEdgeGraph.from_edge_panel(Xs, g.edges, n_nodes=g.n_nodes, ...)

  B) prefix-rolling STL (at each step, redo STL using only the "up-to-current" prefix
     to avoid look-ahead bias):
        builder = make_prefix_stl_graph_builder(g, ...)
        RollingEdgePredict(graph=g, ..., graph_builder=builder)   # rebuild the graph each step with builder
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

from edge_graph import ArrayEdgeGraph, EdgeGraph


def _components_to_idx(components: str) -> List[int]:
    s = str(components).lower().replace(" ", "")
    if s == "all":
        return [0, 1, 2]
    out = []
    if "trend" in s:
        out.append(0)
    if "seasonal" in s:
        out.append(1)
    if "resid" in s:
        out.append(2)
    return out or [2]


def _time_axis(time_labels: List[Any]) -> pd.Index:
    """STL needs an indexed series; convert monthly Period to timestamp, otherwise use integer positions."""
    if len(time_labels) and all(isinstance(t, pd.Period) for t in time_labels):
        return pd.PeriodIndex(time_labels, freq=time_labels[0].freq).to_timestamp()
    return pd.RangeIndex(len(time_labels))


def stl_transform_panel(
    X: np.ndarray,
    time_labels: List[Any],
    *,
    period: int = 12,
    mode: str = "auto",                 # "auto" | "additive" | "log-additive"
    components: str = "resid",          # "trend"/"seasonal"/"resid"/"all"/combination
    robust: bool = True,
    stl_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Apply STL column-by-column to the (T,E) edge panel, returning the (T,E) made up of the sum of the selected components."""
    from statsmodels.tsa.seasonal import STL

    Xc = np.asarray(X, dtype=float)
    T, E = Xc.shape
    mode_eff = mode
    if mode == "auto":
        mode_eff = "additive" if np.any(Xc <= 0.0) else "log-additive"
    Y = np.log1p(np.clip(Xc, 0.0, None)) if mode_eff == "log-additive" else Xc.copy()

    idx = _time_axis(time_labels)
    comps = _components_to_idx(components)
    kw = dict(period=int(period), robust=bool(robust))
    kw.update(stl_kwargs or {})

    out = np.zeros((T, E), dtype=float)
    for k in range(E):
        res = STL(pd.Series(Y[:, k], index=idx), **kw).fit()
        parts = [np.asarray(res.trend, float),
                 np.asarray(res.seasonal, float),
                 np.asarray(res.resid, float)]
        out[:, k] = sum(parts[i] for i in comps)
    return out


def make_prefix_stl_graph_builder(
    base_graph: EdgeGraph,
    *,
    period: int = 12,
    mode: str = "auto",
    components: str = "resid",
    robust: bool = True,
    stl_kwargs: Optional[dict] = None,
) -> Callable[[List[Any]], ArrayEdgeGraph]:
    """
    Return a graph_builder(observed_times) -> ArrayEdgeGraph:
      take the **raw** prefix edge weights of base_graph over observed_times, apply STL,
      and rebuild a post-STL graph using the same edges/A. The structure A is unchanged; only the edge weights are replaced.
    """
    edges = list(base_graph.edges)
    n_nodes = int(base_graph.n_nodes)
    node_labels = list(base_graph.node_labels)

    def builder(observed: List[Any]) -> ArrayEdgeGraph:
        X_raw = base_graph.edge_series(times=observed)            # (len, E) raw
        Xs = stl_transform_panel(X_raw, list(observed), period=period, mode=mode,
                                 components=components, robust=robust, stl_kwargs=stl_kwargs)
        return ArrayEdgeGraph.from_edge_panel(
            Xs, edges, n_nodes=n_nodes, time_labels=list(observed), node_labels=node_labels)

    return builder
