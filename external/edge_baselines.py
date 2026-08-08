"""
edge_baselines.py -- baselines for edge time series (ARIMA / SARIMA / DFM / Naive)
========================================================================================

These baselines answer the question a GNAR-edge paper must answer:

    "Does modelling the NETWORK actually help, compared with modelling each edge
     on its own?"

Every baseline produces a one-step-ahead forecast for every edge from exactly the
same input GNAR-edge takes, so the comparison is fair by construction. They differ
in how much information they are allowed to borrow across edges:

    Naive / Zero      no model at all (reference points)
    ARIMA / SARIMA    one univariate model PER EDGE -- no cross-edge information
    DFM               the WHOLE panel via r estimated latent factors -- cross-edge
                      information, but learned rather than given
    GNAR-edge         the whole panel via the KNOWN network structure

That ordering is the point: if DFM already matches GNAR-edge, then generic
co-movement explains the data and the known network is adding little.

Design
------
1. Same input as GNAR-edge. Every baseline takes an `EdgeGraph` (the same object
   `GNAREdgeLearner` takes). Use `to_edge_graph(...)` to build one from a numpy
   array, a DataFrame, an Excel/CSV file, or a legacy TSG/ipg object.

2. Same output as GNAR-edge. `predict_next()` returns a length-E vector of
   one-step-ahead predictions, aligned with `learner.edges`.

3. Drop-in for the rolling harness. Every baseline subclasses `BaseEdgeLearner`,
   so it can be passed straight to `RollingEdgePredict(learner_cls=...)`. The
   expanding-window backtest code is then *identical* across all models.

Quick start
-----------
    from edge_baselines import to_edge_graph, ARIMAEdgeBaseline, compare_baselines

    g = to_edge_graph("my_panel.xlsx")            # or a numpy array, DataFrame, TSG...

    # (a) single fit + next-step forecast
    m = ARIMAEdgeBaseline(graph=g, train_periods=g.time_labels, order=(1, 0, 0))
    m.fit()
    y_hat = m.predict_next()                       # shape (E,)

    # (b) rolling comparison against GNAR-edge, one table
    from BaseEdgeGNAR_edge_global import GNAREdgeGlobalLearner
    table = compare_baselines(
        g,
        models={
            "naive":  (NaiveEdgeBaseline, {}),
            "arima":  (ARIMAEdgeBaseline, dict(order=(1, 0, 0))),
            "sarima": (SARIMAEdgeBaseline, dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, 7))),
            "gnar":   (GNAREdgeGlobalLearner, dict(L=1, stages_per_lag=[[1]], use_ols=True)),
        },
        val_frac=0.2,
    )

Dependencies: numpy, pandas, statsmodels (ARIMA/SARIMA only).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from edge_graph import ArrayEdgeGraph, EdgeGraph
from BaseEdge import BaseEdgeLearner, EdgeTrainSet

try:                                                        # optional import
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX
    _HAS_STATSMODELS = True
except Exception:                                           # pragma: no cover
    _ARIMA = _SARIMAX = None
    _HAS_STATSMODELS = False


# =====================================================================
# 0) Universal input loader -- the same graph object GNAR-edge consumes
# =====================================================================

def to_edge_graph(
    source: Any,
    *,
    edges: Optional[Sequence[Tuple[int, int]]] = None,
    n_nodes: Optional[int] = None,
    time_labels: Optional[Sequence[Any]] = None,
    node_labels: Optional[Sequence[str]] = None,
    sheet_name: Any = 0,
) -> EdgeGraph:
    """
    Turn almost anything into the `EdgeGraph` that every model here accepts.

    Accepted sources
    ----------------
    EdgeGraph                 : returned unchanged.
    np.ndarray (T, E)         : edge panel; pass `edges` (list of (u, v)).
    np.ndarray (T, K, K)      : value tensor; edges inferred from non-zero entries.
    pd.DataFrame              : rows = time, columns = edges. Column names may be
                                "u->v", "u-v", or (u, v) tuples; otherwise pass `edges`.
    str / Path (.xlsx, .csv)  : read into a DataFrame, then as above.
    ThresholdStaticGraphTS    : legacy TSG/ipg object (adapted via from_tsg).

    Everything downstream only sees an EdgeGraph, so the model code never needs
    to know where the numbers came from.
    """
    # --- already the right thing ---
    if isinstance(source, EdgeGraph):
        return source

    # --- legacy TSG / ipg (duck-typed, same rule BaseEdge uses) ---
    if hasattr(source, "ipg") and hasattr(source, "build_W") and hasattr(source, "edges"):
        return ArrayEdgeGraph.from_tsg(source)

    # --- file path -> DataFrame ---
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        path = str(source)
        if path.lower().endswith((".xlsx", ".xls")):
            source = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
        elif path.lower().endswith((".csv", ".txt", ".tsv")):
            sep = "\t" if path.lower().endswith((".txt", ".tsv")) else ","
            source = pd.read_csv(path, sep=sep, index_col=0)
        else:
            raise ValueError(f"unsupported file type: {path}")

    # --- DataFrame ---
    if isinstance(source, pd.DataFrame):
        df = source
        if edges is None:
            edges = _parse_edge_columns(df.columns)
        if time_labels is None:
            time_labels = list(df.index)
        X = df.to_numpy(dtype=float)
        return ArrayEdgeGraph.from_edge_panel(
            X, list(edges), n_nodes=n_nodes,
            time_labels=time_labels, node_labels=node_labels)

    # --- numpy ---
    X = np.asarray(source, dtype=float)
    if X.ndim == 3:
        return ArrayEdgeGraph.from_value_tensor(
            X, time_labels=time_labels, node_labels=node_labels)
    if X.ndim == 2:
        if edges is None:
            raise ValueError("a (T, E) array needs `edges=[(u, v), ...]`.")
        return ArrayEdgeGraph.from_edge_panel(
            X, list(edges), n_nodes=n_nodes,
            time_labels=time_labels, node_labels=node_labels)

    raise TypeError(f"cannot build an EdgeGraph from {type(source).__name__}.")


def _parse_edge_columns(columns) -> List[Tuple[int, int]]:
    """Read edge ids from DataFrame column names ("u->v", "u-v", or (u, v))."""
    out: List[Tuple[int, int]] = []
    for col in columns:
        if isinstance(col, tuple) and len(col) == 2:
            out.append((int(col[0]), int(col[1])))
            continue
        s = str(col)
        for sep in ("->", "_", "-"):
            if sep in s:
                a, b = s.split(sep, 1)
                try:
                    out.append((int(a), int(b)))
                    break
                except ValueError:
                    raise ValueError(
                        f"column {col!r} is not a numeric edge id; pass `edges=` explicitly "
                        f"(node names are not node indices).")
        else:
            raise ValueError(f"cannot parse an edge from column {col!r}; pass `edges=`.")
    return out


# =====================================================================
# 1) Shared base: one univariate model per edge
# =====================================================================

@dataclass
class _PerEdgeBaseline(BaseEdgeLearner):
    """
    Base class for "fit one univariate model per edge" baselines.

    Subclasses only implement `_forecast_one(y)`: given one edge's history as a
    pandas Series, return (one_step_forecast, info_dict). Everything else --
    graph handling, the rolling contract, error tolerance -- lives here.

    Attributes
    ----------
    min_train_obs : minimum usable observations per edge; below this the edge
                    falls back to the last observed value (and is counted in
                    `n_fallback_`).
    verbose       : print progress every `print_every` edges.
    """

    min_train_obs: int = 5
    verbose: bool = False
    print_every: int = 200

    # populated by fit()
    next_pred_: Optional[np.ndarray] = field(init=False, default=None)
    fit_info_: List[Dict[str, Any]] = field(init=False, default_factory=list)
    n_fallback_: int = field(init=False, default=0)

    # ---- to be provided by subclasses -------------------------------
    def _forecast_one(self, y: pd.Series) -> Tuple[float, Dict[str, Any]]:
        raise NotImplementedError

    def model_name(self) -> str:
        return self.__class__.__name__

    # ---- core: forecast every edge from its own history --------------
    def _on_edge_start(self, edge_pos: int) -> None:
        """Hook called before each edge is fitted (subclasses may use it)."""

    def _forecast_all(self, periods: Sequence[Any]) -> np.ndarray:
        """Fit one model per edge on `periods` and return the (E,) forecast."""
        X = self.graph.edge_series(times=list(periods))          # (T, E)
        preds = np.full(self.E, np.nan, dtype=float)
        self.fit_info_ = []
        self.n_fallback_ = 0

        for e in range(self.E):
            self._on_edge_start(e)
            y = pd.Series(np.asarray(X[:, e], dtype=float))
            y = y.replace([np.inf, -np.inf], np.nan).dropna()
            info: Dict[str, Any] = {"edge_pos": e, "nobs": int(len(y))}

            # too short, or perfectly flat -> last value is the best guess
            if len(y) < int(self.min_train_obs) or y.nunique() <= 1:
                preds[e] = float(y.iloc[-1]) if len(y) else 0.0
                info["status"] = "short_or_constant"
                self.n_fallback_ += 1
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pred, extra = self._forecast_one(y)
                    if not np.isfinite(pred):
                        raise ValueError("non-finite forecast")
                    preds[e] = float(pred)
                    info.update(extra)
                    info.setdefault("status", "ok")
                except Exception as exc:
                    # one bad edge must never kill the whole run
                    preds[e] = float(y.iloc[-1])
                    info["status"] = "failed_fallback_last_value"
                    info["error"] = f"{type(exc).__name__}: {exc}"
                    self.n_fallback_ += 1

            self.fit_info_.append(info)
            if self.verbose and (e + 1) % max(1, self.print_every) == 0:
                print(f"[{self.model_name()}] {e + 1}/{self.E} edges", flush=True)

        return preds

    # ---- BaseEdgeLearner contract ------------------------------------
    def _fit_impl(self, train_set: EdgeTrainSet, **kwargs):
        self.next_pred_ = self._forecast_all(self.train_periods)
        resid = train_set.y[-1, :] - self.next_pred_ if train_set.n_samples else np.array([np.nan])
        self.last_fit_best_loss = float(np.nanmean(resid ** 2))
        self.last_fit_epochs = 1
        self.last_fit_best_epoch = 1
        self.last_fit_stop_reason = "per_edge_closed_form"
        if self.verbose:
            print(f"[{self.model_name()}] fitted {self.E} edges "
                  f"({self.n_fallback_} fallback)", flush=True)

    def forward(self, X: Any, **kwargs) -> np.ndarray:
        raise NotImplementedError(
            f"{self.model_name()} forecasts from each edge's full history, "
            f"not from a fixed lag window; use predict_next().")

    def predict_next(self, periods: Optional[List[Any]] = None, **kwargs) -> np.ndarray:
        """One-step-ahead forecast for every edge, shape (E,)."""
        if periods is None or list(periods) == list(self.train_periods):
            if self.next_pred_ is None:
                self.next_pred_ = self._forecast_all(self.train_periods)
            return np.asarray(self.next_pred_, dtype=float).reshape(-1)
        return np.asarray(self._forecast_all(periods), dtype=float).reshape(-1)

    # ---- optional diagnostics the rolling harness picks up -----------
    def calc_aic_bic(self, train_set: Optional[EdgeTrainSet] = None, **kwargs) -> Dict[str, Any]:
        """Mean AIC/BIC across edges (per-edge models have one IC each)."""
        aic = [i.get("aic", np.nan) for i in self.fit_info_]
        bic = [i.get("bic", np.nan) for i in self.fit_info_]
        ts = train_set if train_set is not None else self.train_set
        return {
            "AIC": float(np.nanmean(aic)) if len(aic) else np.nan,
            "BIC": float(np.nanmean(bic)) if len(bic) else np.nan,
            "k_params": int(self._num_mean_params()),
            "n_samples": int(ts.n_samples) if ts is not None else 0,
            "n_edges": int(self.E),
            "note": "mean of per-edge AIC/BIC",
        }

    def _num_mean_params(self) -> int:
        return 0

    def fit_report(self) -> pd.DataFrame:
        """Per-edge fit diagnostics (status, nobs, chosen order, AIC/BIC)."""
        df = pd.DataFrame(self.fit_info_)
        if not df.empty:
            labels = self.graph.node_labels
            df.insert(0, "edge", [f"{labels[u]}-{labels[v]}" for (u, v) in self.edges])
        return df


# =====================================================================
# 2) Naive / persistence baseline (the reference every paper needs)
# =====================================================================

@dataclass
class NaiveEdgeBaseline(_PerEdgeBaseline):
    """
    Persistence on the MODELLED series: forecast(t+1) = value at t. No parameters.

    Careful about what "persistence" means for your panel:
      * panel holds levels X_t      -> this predicts X_{t+1} = X_t   (the usual naive)
      * panel holds differences dX_t -> this predicts dX_{t+1} = dX_t,
                                        i.e. "the change repeats" -- a WEAK baseline.
    If your panel is already differenced, `ZeroEdgeBaseline` is the honest
    reference: predicting dX = 0 is exactly persistence in levels.
    """

    def model_name(self) -> str:
        return "Naive"

    def _forecast_one(self, y: pd.Series) -> Tuple[float, Dict[str, Any]]:
        return float(y.iloc[-1]), {"aic": np.nan, "bic": np.nan}


@dataclass
class ZeroEdgeBaseline(_PerEdgeBaseline):
    """
    Predict zero for every edge. No parameters.

    On a DIFFERENCED panel this is the standard "no change" reference and is
    identical to persistence in the original level scale -- the baseline a
    difference-target model must beat to be worth anything.
    """

    def model_name(self) -> str:
        return "Zero"

    def _forecast_one(self, y: pd.Series) -> Tuple[float, Dict[str, Any]]:
        return 0.0, {"aic": np.nan, "bic": np.nan}

    def _forecast_all(self, periods: Sequence[Any]) -> np.ndarray:
        # zero everywhere, including short/constant edges (no fallback to last value)
        self.fit_info_ = [{"edge_pos": e, "status": "zero", "nobs": len(list(periods))}
                          for e in range(self.E)]
        self.n_fallback_ = 0
        return np.zeros(self.E, dtype=float)


# =====================================================================
# 3) ARIMA baseline
# =====================================================================

def _normalize_order(value: Any) -> Union[Tuple[int, int, int], str]:
    """Accept (1,0,0), "1,0,0", "(1,0,0)" or "auto"."""
    if isinstance(value, str):
        if value.lower().strip() in {"auto", "auto_arima", "auto-arima"}:
            return "auto"
        parts = value.replace("(", "").replace(")", "").replace(",", " ").split()
        if len(parts) == 3:
            return tuple(int(x) for x in parts)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(x) for x in value)
    raise ValueError(f"order must look like (1, 0, 0) or 'auto', got {value!r}")


def _normalize_seasonal_order(value: Any) -> Tuple[int, int, int, int]:
    """Accept (1,0,0,12), "1,0,0,12" or "(1,0,0,12)"."""
    if isinstance(value, str):
        parts = value.replace("(", "").replace(")", "").replace(",", " ").split()
        if len(parts) == 4:
            value = [int(x) for x in parts]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        P, D, Q, s = (int(x) for x in value)
        if min(P, D, Q, s) < 0:
            raise ValueError(f"seasonal_order must be non-negative, got {value!r}")
        if (P > 0 or D > 0 or Q > 0) and s <= 1:
            raise ValueError(
                f"seasonal period s must be > 1 when P/D/Q are non-zero, got {value!r}")
        return (P, D, Q, s)
    raise ValueError(f"seasonal_order must look like (1, 0, 0, 12), got {value!r}")


def _trend_for(order, seasonal_order=None, trend: Optional[str] = "auto") -> Optional[str]:
    """
    Pick the deterministic trend term.

    With differencing (d > 0 or seasonal D > 0) a constant is not identified,
    so the default drops it ("n"); otherwise a constant ("c") is used.
    """
    d = int(order[1])
    D = int(seasonal_order[1]) if seasonal_order is not None else 0
    integrated = (d > 0) or (D > 0)
    if trend is None:
        return "n" if integrated else "c"
    t = str(trend).lower().strip()
    if t in ("auto", ""):
        return "n" if integrated else "c"
    if t in ("none", "no", "null"):
        return None
    if t not in ("n", "c", "t", "ct"):
        raise ValueError("trend must be one of auto, n, c, t, ct, none")
    return t


@dataclass
class ARIMAEdgeBaseline(_PerEdgeBaseline):
    """
    ARIMA(p, d, q) fitted independently on every edge.

    Parameters
    ----------
    order : (p, d, q), or "auto" to pick the order per edge by AIC/BIC from
            `auto_candidates`.
    trend : "auto" (default) | "n" | "c" | "t" | "ct" | None.
    auto_candidates : orders searched when order="auto".
    auto_ic : "aic" or "bic".
    auto_reselect_each_fit : if False (default) the order is chosen once per edge,
            on the first training window, then reused -- much faster in a rolling
            backtest and standard practice. Set True to reselect at every refit.
    """

    order: Any = (1, 0, 0)
    trend: Optional[str] = "auto"
    auto_candidates: Sequence[Tuple[int, int, int]] = (
        (0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1),
        (2, 0, 0), (0, 0, 2), (1, 1, 0), (0, 1, 1), (1, 1, 1),
    )
    auto_ic: str = "aic"
    auto_reselect_each_fit: bool = False
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True

    _auto_cache: Dict[int, Tuple[int, int, int]] = field(init=False, default_factory=dict)
    _edge_cursor: int = field(init=False, default=0)

    def __post_init__(self):
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels is required for ARIMA/SARIMA baselines.")
        self.order = _normalize_order(self.order)
        if str(self.auto_ic).lower() not in ("aic", "bic"):
            raise ValueError("auto_ic must be 'aic' or 'bic'")
        super().__post_init__()

    def model_name(self) -> str:
        return "ARIMA" if self.order != "auto" else "ARIMA(auto)"

    # -- statsmodels plumbing (shared with the SARIMA subclass) --------
    def _make_model(self, y: pd.Series, order, seasonal_order=None):
        trend_eff = _trend_for(order, seasonal_order, self.trend)
        if seasonal_order is None:
            return _ARIMA(y.astype(float), order=tuple(order), trend=trend_eff,
                          enforce_stationarity=bool(self.enforce_stationarity),
                          enforce_invertibility=bool(self.enforce_invertibility),
                          missing="drop")
        return _SARIMAX(y.astype(float), order=tuple(order),
                        seasonal_order=tuple(seasonal_order), trend=trend_eff,
                        enforce_stationarity=bool(self.enforce_stationarity),
                        enforce_invertibility=bool(self.enforce_invertibility),
                        missing="drop")

    def _select_order(self, y: pd.Series, seasonal_order=None) -> Tuple[Any, float]:
        """Pick the order minimising AIC/BIC; fall back to (1,0,0) if all fail."""
        best_order, best_ic = None, np.inf
        for cand in self.auto_candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = self._make_model(y, cand, seasonal_order).fit()
                ic = float(getattr(res, str(self.auto_ic).lower()))
                if np.isfinite(ic) and ic < best_ic:
                    best_ic, best_order = ic, tuple(cand)
            except Exception:
                continue
        if best_order is None:
            return (1, 0, 0), np.nan
        return best_order, best_ic

    def _resolve_order(self, y: pd.Series, seasonal_order=None) -> Tuple[Any, Dict[str, Any]]:
        if self.order != "auto":
            return self.order, {"order_source": "fixed"}
        e = self._edge_cursor
        if (not self.auto_reselect_each_fit) and e in self._auto_cache:
            return self._auto_cache[e], {"order_source": "auto_cached"}
        chosen, ic = self._select_order(y, seasonal_order)
        self._auto_cache[e] = chosen
        return chosen, {"order_source": "auto_selected", "selection_ic": ic}

    def _forecast_one(self, y: pd.Series) -> Tuple[float, Dict[str, Any]]:
        order, info = self._resolve_order(y)
        res = self._make_model(y, order).fit()
        pred = float(np.asarray(res.forecast(steps=1))[0])
        info.update({
            "order": tuple(order),
            "aic": float(res.aic) if np.isfinite(res.aic) else np.nan,
            "bic": float(res.bic) if np.isfinite(res.bic) else np.nan,
        })
        return pred, info

    def _on_edge_start(self, edge_pos: int) -> None:
        """Tell `_resolve_order` which edge we are on (auto-order cache is per edge)."""
        self._edge_cursor = edge_pos

    def _num_mean_params(self) -> int:
        if self.order == "auto":
            return int(sum(o[0] + o[2] for o in self._auto_cache.values()))
        p, _d, q = self.order
        return int(self.E * (p + q))


# =====================================================================
# 4) SARIMA baseline
# =====================================================================

@dataclass
class SARIMAEdgeBaseline(ARIMAEdgeBaseline):
    """
    Seasonal ARIMA(p,d,q)(P,D,Q,s) fitted independently on every edge.

    `seasonal_order=(P, D, Q, s)`; s is the seasonal period in *windows*, so it
    depends on how the panel was aggregated -- e.g. s=12 for monthly data with a
    yearly cycle, s=7 for daily data with a weekly cycle, s=3 for 8-hourly data
    with a daily cycle.
    """

    seasonal_order: Any = (0, 0, 0, 0)

    def __post_init__(self):
        self.seasonal_order = _normalize_seasonal_order(self.seasonal_order)
        super().__post_init__()

    def model_name(self) -> str:
        return "SARIMA" if self.order != "auto" else "SARIMA(auto)"

    def _forecast_one(self, y: pd.Series) -> Tuple[float, Dict[str, Any]]:
        order, info = self._resolve_order(y, self.seasonal_order)
        res = self._make_model(y, order, self.seasonal_order).fit(disp=False)
        pred = float(np.asarray(res.forecast(steps=1))[0])
        info.update({
            "order": tuple(order),
            "seasonal_order": tuple(self.seasonal_order),
            "aic": float(res.aic) if np.isfinite(res.aic) else np.nan,
            "bic": float(res.bic) if np.isfinite(res.bic) else np.nan,
        })
        return pred, info

    def _num_mean_params(self) -> int:
        base = super()._num_mean_params()
        P, _D, Q, _s = self.seasonal_order
        return int(base + self.E * (P + Q))


# =====================================================================
# 5) Dynamic Factor Model baseline (panel-wide, not per-edge)
# =====================================================================

def _standardize_columns(hist: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise z-score using the current window; non-finite entries become 0."""
    mu = np.nanmean(hist, axis=0)
    sd = np.nanstd(hist, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    z = (hist - mu) / sd
    return np.where(np.isfinite(z), z, 0.0), mu, sd


def _var_one_step(F: np.ndarray, p: int) -> np.ndarray:
    """Fit VAR(p) on the factors F (T x r) and return the next-period factors (r,)."""
    T, r = F.shape
    p = max(1, int(p))
    if T <= p + 1:
        return F[-1].copy()
    Y = F[p:]                                                              # (T-p, r)
    X = np.concatenate([F[p - k - 1: T - k - 1] for k in range(p)], axis=1)  # (T-p, r*p)
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)                              # Y ~ X @ B
    x_last = np.concatenate([F[T - k - 1] for k in range(p)])              # (r*p,)
    return x_last @ B


def dfm_pca_one_step(hist: np.ndarray, *, n_factors: int, factor_order: int,
                     idiosyncratic: str, standardize: bool) -> np.ndarray:
    """
    One-step forecast from a two-step (PCA) dynamic factor model.

    hist : (T, E) panel of the modelled target, may contain NaN.
    Returns the (E,) forecast for the next period.

    Steps: standardise -> SVD -> factors F and loadings L -> VAR(p) on F ->
    common component L @ f_next -> optional AR(1) idiosyncratic term -> unstandardise.
    """
    T, E = hist.shape
    if standardize:
        z, mu, sd = _standardize_columns(hist)
    else:
        z = np.where(np.isfinite(hist), hist, 0.0)
        mu, sd = np.zeros(E), np.ones(E)

    r_eff = int(max(1, min(int(n_factors), E, T - int(factor_order) - 1)))

    try:
        U, S, Vt = np.linalg.svd(z, full_matrices=False)
    except np.linalg.LinAlgError:
        U, S, Vt = np.linalg.svd(np.nan_to_num(z), full_matrices=False)
    F = U[:, :r_eff] * S[:r_eff]                    # (T, r_eff) factors
    L = Vt[:r_eff, :].T                             # (E, r_eff) loadings

    f_next = _var_one_step(F, factor_order)
    z_next = L @ f_next                             # common component, standardised scale

    if str(idiosyncratic).lower() == "ar1":
        e = z - F @ L.T                             # (T, E) idiosyncratic residuals
        num = np.sum(e[1:] * e[:-1], axis=0)
        den = np.sum(e[:-1] ** 2, axis=0)
        rho = np.clip(np.where(den > 1e-12, num / den, 0.0), -0.99, 0.99)
        z_next = z_next + rho * e[-1]

    return mu + sd * z_next


@dataclass
class DFMEdgeBaseline(BaseEdgeLearner):
    """
    Dynamic Factor Model over the WHOLE edge panel.

        measurement : y_t = L f_t + e_t          (r common factors, E x r loadings)
        state       : f_t = A_1 f_{t-1} + ... + A_p f_{t-p} + eta_t
        idiosyncratic: e_it white noise, or AR(1) per edge

    This sits between the per-edge baselines and GNAR-edge: it DOES borrow strength
    across edges, but through estimated latent factors rather than the known network.
    A useful third comparison -- if DFM already captures what GNAR-edge captures,
    the network structure is adding little beyond generic co-movement.

    Parameters
    ----------
    n_factors    : number of common factors r (clipped to what the window supports).
    factor_order : VAR order p on the factors (default 1).
    method       : "pca" (two-step SVD, fast, recommended) or
                   "em"  (statsmodels DynamicFactorMQ, exact ML, slow for large E).
    idiosyncratic: "ar1" (per-edge AR(1) on residuals) or "white".
    standardize  : z-score each edge on the training window.
    em_maxiter   : EM iterations when method="em".
    """

    n_factors: int = 5
    factor_order: int = 1
    method: str = "pca"
    idiosyncratic: str = "ar1"
    standardize: bool = True
    em_maxiter: int = 100
    em_large_e_warn: int = 1200
    verbose: bool = False

    next_pred_: Optional[np.ndarray] = field(init=False, default=None)
    fit_info_: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        m = str(self.method).lower().strip()
        if m not in ("pca", "em"):
            raise ValueError("method must be 'pca' or 'em'")
        self.method = m
        if str(self.idiosyncratic).lower().strip() not in ("ar1", "white"):
            raise ValueError("idiosyncratic must be 'ar1' or 'white'")
        super().__post_init__()

    def model_name(self) -> str:
        return f"DFM({self.method}, r={self.n_factors})"

    # ---- core forecast over the whole panel ----
    def _forecast_all(self, periods: Sequence[Any]) -> np.ndarray:
        hist = np.asarray(self.graph.edge_series(times=list(periods)), dtype=float)  # (T, E)
        T, E = hist.shape
        info: Dict[str, Any] = {"method": self.method, "n_obs": int(T), "n_edges": int(E),
                                "r_requested": int(self.n_factors)}

        if self.method == "em":
            pred, extra = self._forecast_em(hist, periods)
        else:
            pred = dfm_pca_one_step(hist, n_factors=self.n_factors,
                                    factor_order=self.factor_order,
                                    idiosyncratic=self.idiosyncratic,
                                    standardize=self.standardize)
            extra = {"r_used": int(max(1, min(self.n_factors, E, T - self.factor_order - 1))),
                     "status": "ok"}
        info.update(extra)
        self.fit_info_ = info

        # any edge that failed falls back to its last observed value
        bad = ~np.isfinite(pred)
        if bad.any():
            last = hist[-1, :]
            pred = np.where(bad, np.where(np.isfinite(last), last, 0.0), pred)
            info["n_fallback"] = int(bad.sum())
        return np.asarray(pred, dtype=float).reshape(-1)

    def _forecast_em(self, hist: np.ndarray, periods) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Exact-ML variant via statsmodels DynamicFactorMQ (refit on this window)."""
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

        T, E = hist.shape
        if E >= int(self.em_large_e_warn):
            print(f"[{self.model_name()}] warning: E={E} is large; DynamicFactorMQ EM will be "
                  f"slow and memory hungry. Consider method='pca' or filtering edges.", flush=True)

        idx = pd.PeriodIndex(list(periods)).to_timestamp() \
            if isinstance(periods[0], pd.Period) else pd.RangeIndex(T)
        df = pd.DataFrame(hist, index=idx)
        r_eff = int(max(1, min(int(self.n_factors), E - 1)))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = DynamicFactorMQ(
                    df, factors=r_eff, factor_orders=int(max(1, self.factor_order)),
                    idiosyncratic_ar1=(self.idiosyncratic == "ar1"),
                    standardize=bool(self.standardize))
                res = mod.fit(disp=False, maxiter=int(self.em_maxiter))
                fc = np.asarray(res.forecast(steps=1).iloc[0].to_numpy(), dtype=float)
            return fc, {"r_used": r_eff, "status": "ok"}
        except Exception as exc:
            # fall back to the PCA variant rather than losing the whole run
            pred = dfm_pca_one_step(hist, n_factors=self.n_factors,
                                    factor_order=self.factor_order,
                                    idiosyncratic=self.idiosyncratic,
                                    standardize=self.standardize)
            return pred, {"r_used": r_eff, "status": "em_failed_pca_fallback",
                          "error": f"{type(exc).__name__}: {exc}"}

    # ---- BaseEdgeLearner contract ----
    def _fit_impl(self, train_set: EdgeTrainSet, **kwargs):
        self.next_pred_ = self._forecast_all(self.train_periods)
        self.last_fit_epochs = 1
        self.last_fit_best_epoch = 1
        self.last_fit_stop_reason = "dfm_closed_form"
        if self.verbose:
            print(f"[{self.model_name()}] {self.fit_info_}", flush=True)

    def forward(self, X: Any, **kwargs) -> np.ndarray:
        raise NotImplementedError(
            f"{self.model_name()} forecasts from the whole panel history, not from a fixed "
            f"lag window; use predict_next().")

    def predict_next(self, periods: Optional[List[Any]] = None, **kwargs) -> np.ndarray:
        """One-step-ahead forecast for every edge, shape (E,)."""
        if periods is None or list(periods) == list(self.train_periods):
            if self.next_pred_ is None:
                self.next_pred_ = self._forecast_all(self.train_periods)
            return np.asarray(self.next_pred_, dtype=float).reshape(-1)
        return self._forecast_all(periods)

    def calc_aic_bic(self, train_set: Optional[EdgeTrainSet] = None, **kwargs) -> Dict[str, Any]:
        ts = train_set if train_set is not None else self.train_set
        r = int(self.fit_info_.get("r_used", self.n_factors))
        return {"AIC": np.nan, "BIC": np.nan,
                "k_params": int(self.E * r + r * r * max(1, self.factor_order)),
                "n_samples": int(ts.n_samples) if ts is not None else 0,
                "n_edges": int(self.E),
                "note": "DFM parameter count is approximate (loadings + factor VAR)"}

    def fit_report(self) -> pd.DataFrame:
        """One row describing the panel-level fit (DFM is not fitted per edge)."""
        return pd.DataFrame([self.fit_info_])


# =====================================================================
# 6) Rolling comparison: every model through the identical harness
# =====================================================================

def compare_baselines(
    graph: Any,
    models: Dict[str, Tuple[type, Dict[str, Any]]],
    *,
    val_frac: float = 0.2,
    train_periods: Optional[Sequence[Any]] = None,
    val_periods: Optional[Sequence[Any]] = None,
    fit_kwargs_by_model: Optional[Dict[str, Dict[str, Any]]] = None,
    reference: str = "auto",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run several models through the SAME expanding-window rolling backtest and
    return one tidy table -- the comparison a paper needs.

    Parameters
    ----------
    graph  : anything `to_edge_graph` accepts (EdgeGraph, array, DataFrame, file...).
    models : {name: (learner_cls, learner_kwargs)}. Mix baselines and GNAR-edge freely.
    val_frac : last fraction of the time axis used as the rolling test set (0.2 = 80/20).
    reference : which model's MAE the skill score is measured against.
            "auto"  -> use "zero" if present, else "naive", else compute persistence.
            Pass an explicit model name to override. On a DIFFERENCED panel use
            "zero" (predicting no change); on a LEVEL panel use "naive".

    Returns
    -------
    DataFrame with MAE / RMSE / skill_vs_reference per model, plus the per-model
    prediction frames in `.attrs["predictions"]`.

    Note on order="auto": the rolling harness builds a fresh learner at every
    step, so an automatic order is re-selected on each training window. That is
    slower but statistically cleaner (no look-ahead); use a fixed order for speed.
    """
    from RollingEdgePredict import RollingEdgePredict          # local import: optional dep

    g = to_edge_graph(graph)
    times = list(g.time_labels)
    if train_periods is None or val_periods is None:
        n_val = max(1, round(float(val_frac) * len(times)))
        train_periods, val_periods = times[:len(times) - n_val], times[len(times) - n_val:]

    rows, preds = [], {}
    for name, (cls, kw) in models.items():
        if verbose:
            print(f"[compare] running {name} ({cls.__name__}) ...", flush=True)
        fit_kw = (fit_kwargs_by_model or {}).get(name, {})
        roller = RollingEdgePredict(
            graph=g, learner_cls=cls,
            train_periods=list(train_periods), val_periods=list(val_periods),
            learner_kwargs=dict(kw), fit_kwargs=dict(fit_kw),
            graph_builder=None, strict_next_match=True,
        ).run()

        out = roller.to_prediction_dfs(edge_mode="intersection")
        p, r = out["pred_df"], out["real_df"]
        err = np.abs(p.values - r.values)
        rows.append({
            "model": name,
            "n_test": len(r),
            "n_edges": p.shape[1],
            "MAE": float(np.nanmean(err)),
            "RMSE": float(np.sqrt(np.nanmean((p.values - r.values) ** 2))),
        })
        preds[name] = out

    table = pd.DataFrame(rows)

    # ---- reference model for the skill score ----
    ref_val, ref_name = None, None
    if reference != "auto":
        if reference not in table["model"].values:
            raise ValueError(f"reference={reference!r} is not one of the models run.")
        ref_name = reference
    else:
        for cand in ("zero", "Zero", "naive", "persistence", "Naive"):
            if cand in table["model"].values:
                ref_name = cand
                break
    if ref_name is not None:
        ref_val = float(table.loc[table["model"] == ref_name, "MAE"].iloc[0])
    else:                                   # no reference model run: compute persistence
        real = next(iter(preds.values()))["real_df"]
        X = g.edge_series()
        tidx = {t: i for i, t in enumerate(times)}
        base = np.vstack([X[tidx[w] - 1, :] for w in real.index])
        ref_val, ref_name = float(np.nanmean(np.abs(base - real.values))), "persistence(computed)"

    table["reference"] = ref_name
    table["reference_MAE"] = ref_val
    table["skill_vs_reference"] = 1.0 - table["MAE"] / ref_val
    table = table.sort_values("skill_vs_reference", ascending=False).reset_index(drop=True)
    table.attrs["predictions"] = preds
    return table


__all__ = [
    "to_edge_graph",
    "NaiveEdgeBaseline",
    "ZeroEdgeBaseline",
    "ARIMAEdgeBaseline",
    "SARIMAEdgeBaseline",
    "DFMEdgeBaseline",
    "dfm_pca_one_step",
    "compare_baselines",
]
