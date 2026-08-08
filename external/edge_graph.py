"""
edge_graph.py
=============

A general-purpose "fixed graph + time-varying edge weights" data base class, used to decouple
algorithms (BaseEdge / GNAR-edge / Rolling) from concrete data sources (TSG / ipg / Excel / numpy / ...).

Design principle (minimalism):
    What the algorithm truly needs is just pure functions over 4 pieces of state:
        A           K×K adjacency matrix (defines which edges exist -> edges / n_nodes / W^(r))
        X           T×E edge panel (the value of each edge over time, the "time-varying edge weights" in the paper)
        time_labels time labels of length T (optional, purely for display / seasonality)
        node_labels node names of length K (optional)

    So this module defines:
        EdgeGraph       abstract contract: the only interface the algorithm depends on (kept extremely small)
        ArrayEdgeGraph  default in-memory implementation: directly holds (A, X, labels), and provides various factory methods
                        (from numpy / from Excel-csv / from explicit adjacency / from threshold rule / from TSG)

    The time axis uses "integer positions 0..T-1", completely removing the pandas.Period coupling;
    when dates/months are needed, a separate optional time_labels is kept for display and STL use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int]


# =====================================================================
# 1) Abstract contract layer: the only interface the algorithm depends on
# =====================================================================

class EdgeGraph(ABC):
    """
    Minimal contract for a fixed directed graph + time-varying edge weights.

    Algorithms (BaseEdge etc.) are only allowed to call the members below; everything else (how to read files,
    how to build edges by threshold, how to cache W) is an implementation detail the algorithm is unaware of.
    """

    # ---- Graph structure ----
    @property
    @abstractmethod
    def edges(self) -> List[Edge]:
        """Ordered edge list [(u, v), ...], where u/v are integer node indices."""

    @property
    @abstractmethod
    def n_nodes(self) -> int:
        """Number of nodes K."""

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def edge_to_idx(self) -> Dict[Edge, int]:
        """Edge -> column index; derived automatically from edges, subclasses generally need not override."""
        return {e: i for i, e in enumerate(self.edges)}

    @property
    def node_labels(self) -> List[str]:
        """Node names; defaults to '0','1',..., subclasses may override to give real SIC codes etc."""
        return [str(i) for i in range(self.n_nodes)]

    # ---- Time axis ----
    @property
    @abstractmethod
    def n_times(self) -> int:
        """Number of time steps T."""

    @property
    def time_index(self) -> List[int]:
        """Integer-position time axis 0..T-1. This is the algorithm's only frame of reference for time."""
        return list(range(self.n_times))

    @property
    def time_labels(self) -> List[Any]:
        """Optional time labels (date / pd.Period / string); defaults to equal the integer positions."""
        return list(range(self.n_times))

    def position_of(self, t: Any) -> int:
        """Uniformly translate a "time label or integer position" into an integer position."""
        if isinstance(t, (int, np.integer)):
            return int(t)
        # Fall back: look up by time_labels
        labels = list(self.time_labels)
        return labels.index(t)

    # ---- Core data access (all the algorithm's data fetching goes through this single entry point) ----
    @abstractmethod
    def edge_series(
        self,
        times: Optional[Sequence[Any]] = None,
        edges: Optional[Sequence[Edge]] = None,
    ) -> np.ndarray:
        """
        Fetch a (len(times), len(edges)) edge-value matrix.
            times default = all times (0..T-1), may pass integer positions or time labels.
            edges default = all edges.
        """

    # ---- Graph neighbour structure (needed by GNAR; not needed by pure AR/MLP) ----
    @abstractmethod
    def neighbor_matrix(self, r: int) -> np.ndarray:
        """
        r-th order adjacency weight matrix W^(r), shape (E, E).
            W[a, b] = 1/|N^r(a)|  when edge b is exactly an r-th order neighbour of edge a, otherwise 0.
        See ArrayEdgeGraph for the default implementation (BFS over A + row normalisation), subclasses may override to add caching.
        """

    # ---- Copy ----
    @abstractmethod
    def copy(self) -> "EdgeGraph":
        ...


# =====================================================================
# 2) Default in-memory implementation + various factory methods
# =====================================================================

def presence_selector(min_obs: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    """
    The most general one-dimensional "build edges by observation condition" rule:
        edge (u, v) exists  <=>  the count of finite and non-zero entries in V[:, u, v] >= min_obs.

    Returns a selector(V) -> A (boolean K×K).
    (The vmin / nmin scheme is a more specialised rule; whoever needs it writes their own selector and passes it in,
      without polluting the base class.)
    """
    def _sel(V: np.ndarray) -> np.ndarray:
        finite_nonzero = np.isfinite(V) & (V != 0.0)
        counts = finite_nonzero.sum(axis=0)          # (K, K)
        return counts >= int(min_obs)
    return _sel


class ArrayEdgeGraph(EdgeGraph):
    """
    Simplest in-memory implementation: directly holds (A, X[T,E], time_labels, node_labels).

    A is used only to "define the edge set + compute W^(r)", looking only at 0/non-zero (presence);
    the actual numerical edge weights over time live in X.
    """

    def __init__(
        self,
        A: np.ndarray,
        X: np.ndarray,
        edges: List[Edge],
        *,
        time_labels: Optional[Sequence[Any]] = None,
        node_labels: Optional[Sequence[str]] = None,
    ):
        A = np.asarray(A)
        X = np.asarray(X, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square (K,K), got {A.shape}.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (T,E), got {X.shape}.")
        if X.shape[1] != len(edges):
            raise ValueError(f"X has E={X.shape[1]} cols but len(edges)={len(edges)}.")

        self._A = A
        self._X = X
        self._edges = [(int(u), int(v)) for (u, v) in edges]
        self._edge_to_idx = {e: i for i, e in enumerate(self._edges)}
        self._K = int(A.shape[0])
        self._time_labels = None if time_labels is None else list(time_labels)
        self._node_labels = None if node_labels is None else [str(c) for c in node_labels]
        self._W_cache: Dict[int, np.ndarray] = {}
        # Optional: an externally provided W^(r) computer (e.g. reuse TSG.build_W's file cache).
        # When None, use this class's general BFS implementation.
        self._W_provider: Optional[Callable[[int], np.ndarray]] = None

        if self._time_labels is not None and len(self._time_labels) != X.shape[0]:
            raise ValueError("len(time_labels) must equal T.")
        if self._node_labels is not None and len(self._node_labels) != self._K:
            raise ValueError("len(node_labels) must equal K.")

    # ---------- Contract implementation ----------
    @property
    def edges(self) -> List[Edge]:
        return list(self._edges)

    @property
    def edge_to_idx(self) -> Dict[Edge, int]:
        return dict(self._edge_to_idx)

    @property
    def n_nodes(self) -> int:
        return self._K

    @property
    def node_labels(self) -> List[str]:
        return list(self._node_labels) if self._node_labels is not None else [str(i) for i in range(self._K)]

    @property
    def n_times(self) -> int:
        return int(self._X.shape[0])

    @property
    def time_labels(self) -> List[Any]:
        return list(self._time_labels) if self._time_labels is not None else list(range(self.n_times))

    def edge_series(self, times=None, edges=None) -> np.ndarray:
        if times is None:
            rows = slice(None)
        else:
            rows = [self.position_of(t) for t in times]
        if edges is None:
            cols = slice(None)
        else:
            cols = [self._edge_to_idx[(int(u), int(v))] for (u, v) in edges]
        sub = self._X[rows, :]
        sub = sub[:, cols] if not isinstance(cols, slice) else sub
        return np.asarray(sub, dtype=float)

    def neighbor_matrix(self, r: int) -> np.ndarray:
        r = int(r)
        if r in self._W_cache:
            return self._W_cache[r]
        W = self._W_provider(r) if self._W_provider is not None else self._compute_W(r)
        W = np.asarray(W, dtype=np.float64)
        self._W_cache[r] = W
        return W

    def copy(self) -> "ArrayEdgeGraph":
        g = ArrayEdgeGraph(
            self._A.copy(),
            self._X.copy(),
            list(self._edges),
            time_labels=None if self._time_labels is None else list(self._time_labels),
            node_labels=None if self._node_labels is None else list(self._node_labels),
        )
        g._W_cache = {k: v.copy() for k, v in self._W_cache.items()}
        g._W_provider = self._W_provider      # preserve the external W computer (e.g. TSG cache)
        return g

    # ---------- General computation of W^(r) (exactly r-th order neighbours + row normalisation) ----------
    def _incident_sets(self):
        out = [set() for _ in range(self._K)]
        inn = [set() for _ in range(self._K)]
        for (u, v) in self._edges:
            out[u].add((u, v))
            inn[v].add((u, v))
        return out, inn

    def _one_stage(self, e: Edge, out, inn) -> set:
        u, v = e
        return (out[u] | inn[u] | out[v] | inn[v]) - {e}

    def _compute_W(self, r: int) -> np.ndarray:
        edges = self._edges
        idx = self._edge_to_idx
        Ke = len(edges)
        W = np.zeros((Ke, Ke), dtype=np.float64)
        out, inn = self._incident_sets()

        for i, e in enumerate(edges):
            visited = {e}
            frontier = {e}
            layer_r: set = set()
            for depth in range(1, r + 1):
                cand: set = set()
                for f in frontier:
                    cand |= self._one_stage(f, out, inn)
                layer = cand - visited
                if depth == r:
                    layer_r = layer
                visited |= layer
                frontier = layer
                if not frontier:
                    break
            if layer_r:
                w = 1.0 / len(layer_r)
                for f in layer_r:
                    j = idx.get(f)
                    if j is not None:
                        W[i, j] = w
        return W

    # =================================================================
    # Factory methods: cover all the "input / generation" ways you asked for
    # =================================================================

    @classmethod
    def from_edge_panel(
        cls,
        X: np.ndarray,
        edges: List[Edge],
        *,
        n_nodes: Optional[int] = None,
        time_labels=None,
        node_labels=None,
    ) -> "ArrayEdgeGraph":
        """Given an existing edge list + (T,E) panel, load it directly. A is inferred from the presence of edges."""
        edges = [(int(u), int(v)) for (u, v) in edges]
        if n_nodes is None:
            n_nodes = (max((max(u, v) for (u, v) in edges)) + 1) if edges else 0
        A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        for (u, v) in edges:
            A[u, v] = 1.0
        return cls(A, X, edges, time_labels=time_labels, node_labels=node_labels)

    @classmethod
    def from_value_tensor(
        cls,
        V: np.ndarray,
        *,
        A: Optional[np.ndarray] = None,
        edge_selector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        time_labels=None,
        node_labels=None,
    ) -> "ArrayEdgeGraph":
        """
        Construct from a numpy (T,K,K) data stream.
            A given           -> use A to define edges;
            A not given, selector given -> A = edge_selector(V);
            neither given     -> default presence_selector() (build an edge wherever a non-zero has appeared).
        X is obtained from the slice of V over those edges.
        """
        V = np.asarray(V, dtype=float)
        if V.ndim != 3 or V.shape[1] != V.shape[2]:
            raise ValueError(f"V must be (T,K,K), got {V.shape}.")
        K = V.shape[1]

        if A is None:
            sel = edge_selector or presence_selector()
            A = sel(V)
        A = np.asarray(A)
        if A.shape != (K, K):
            raise ValueError(f"A shape {A.shape} != (K,K)={(K, K)}.")

        ei, ej = np.nonzero(A)
        edges = [(int(u), int(v)) for u, v in zip(ei, ej)]
        X = V[:, ei, ej].astype(float)               # (T, E)
        A01 = np.zeros((K, K), dtype=np.float64)
        A01[ei, ej] = 1.0
        return cls(A01, X, edges, time_labels=time_labels, node_labels=node_labels)

    @classmethod
    def from_adjacency(
        cls,
        A: np.ndarray,
        values: np.ndarray,
        *,
        time_labels=None,
        node_labels=None,
    ) -> "ArrayEdgeGraph":
        """
        A = explicit adjacency (the meta matrix you mentioned), defines the edge set (in A's row-major order).
        values may be a (T,K,K) tensor, or a (T,E) panel already aligned to A's edge order.
        """
        A = np.asarray(A)
        K = A.shape[0]
        ei, ej = np.nonzero(A)
        edges = [(int(u), int(v)) for u, v in zip(ei, ej)]
        values = np.asarray(values, dtype=float)
        if values.ndim == 3:
            X = values[:, ei, ej].astype(float)
        elif values.ndim == 2:
            if values.shape[1] != len(edges):
                raise ValueError(f"values (T,E) has E={values.shape[1]} but A has {len(edges)} edges.")
            X = values
        else:
            raise ValueError("values must be (T,K,K) or (T,E).")
        A01 = np.zeros((K, K), dtype=np.float64)
        A01[ei, ej] = 1.0
        return cls(A01, X, edges, time_labels=time_labels, node_labels=node_labels)

    @classmethod
    def from_files(
        cls,
        panel_path: str,
        *,
        adjacency_path: Optional[str] = None,
        edge_selector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        sep: str = ",",
    ) -> "ArrayEdgeGraph":
        """
        A thin layer of file reading (xlsx / csv). Conventions:
            panel_path : a table, rows=time, columns=edges, with column names like "u->v" (u,v are node indices).
                         the row index (if any) serves as time_labels.
            adjacency_path (optional): a K×K adjacency meta matrix (also xlsx/csv).
        When adjacency_path is not given, A is defined directly by the columns (edges) that appear.
        Requires pandas.
        """
        import pandas as pd

        def _read(path):
            if str(path).lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(path, index_col=0)
            return pd.read_csv(path, sep=sep, index_col=0)

        panel = _read(panel_path)
        edges = []
        for col in panel.columns:
            s = str(col)
            u, v = s.split("->")
            edges.append((int(u), int(v)))
        X = panel.to_numpy(dtype=float)
        time_labels = list(panel.index)

        if adjacency_path is not None:
            A = _read(adjacency_path).to_numpy()
            return cls.from_adjacency(A, _panel_to_tensor(X, edges, A.shape[0]),
                                      time_labels=time_labels)
        return cls.from_edge_panel(X, edges, time_labels=time_labels)

    @classmethod
    def from_tsg(cls, tsg, *, use_tsg_W: bool = True) -> "ArrayEdgeGraph":
        """
        Adapter: turn an existing ThresholdStaticGraphTS into a "source" for this base class,
        without changing TSG at all. This makes the statement "TSG is just one implementation of this base class" concrete.

        When use_tsg_W=True, neighbor_matrix directly reuses tsg.build_W (with its cache);
        otherwise it recomputes using this base class's general BFS (the result should be identical).
        """
        periods = list(tsg.get_selected_periods())
        edges = [(int(u), int(v)) for (u, v) in tsg.edges]
        ipg = tsg.ipg
        K = int(ipg.K)
        p2i = ipg.period_to_idx
        idx = [p2i[p] for p in periods]
        ei = np.array([u for (u, v) in edges], dtype=int)
        ej = np.array([v for (u, v) in edges], dtype=int)
        X = ipg.value[idx][:, ei, ej].astype(float)        # (T, E)
        A01 = np.zeros((K, K), dtype=np.float64)
        A01[ei, ej] = 1.0
        node_labels = list(getattr(ipg, "codes", [])) or None

        g = cls(A01, X, edges, time_labels=periods, node_labels=node_labels)
        if use_tsg_W:
            # Reuse TSG.build_W (with its file cache); the result matches the general BFS but is faster.
            g._W_provider = lambda r, _t=tsg: np.asarray(_t.build_W(int(r)), dtype=np.float64)
        return g


def _panel_to_tensor(X: np.ndarray, edges: List[Edge], K: int) -> np.ndarray:
    """(T,E) panel -> (T,K,K) tensor, for factories that need a tensor."""
    T = X.shape[0]
    V = np.zeros((T, K, K), dtype=float)
    for j, (u, v) in enumerate(edges):
        V[:, u, v] = X[:, j]
    return V
