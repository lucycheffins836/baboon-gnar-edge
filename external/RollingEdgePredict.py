"""
RollingEdgePredict.py
=====================

Expanding-window one-step rolling backtest container -- **pure algorithm, depends only on EdgeGraph**.

In essence: run the same algorithm independently n times. At step k, train using "the time observed
up to the present", predict the next period, and record the prediction / ground truth / metrics.

Two graph modes:
  1) Fixed graph (default, graph_builder=None):
     Assume the network structure A stays unchanged over the whole backtest period, only the edge
     weights vary over time.
     => Do not rebuild the graph; each step simply trains on a longer time slice. W^(r) is computed
        only once (cached and reused).
  2) Rebuild every step (pass graph_builder):
     graph_builder(observed_times) -> EdgeGraph
     Used for cases where "the graph structure changes with the data" or "each step needs
     data-dependent preprocessing such as prefix-STL".
     This is the entry point that keeps all of ONS's ipg/TSG/STL etc. in the orchestration layer.

Completely independent of any specific data source (ipg / TSG / Excel / numpy).
"""

from __future__ import annotations

import os
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from edge_graph import EdgeGraph
from BaseEdge import BaseEdgeLearner


def _r2_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return np.nan
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    if sst == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / sst


# -------------------------------------------------
# 1) Single-step result
# -------------------------------------------------

@dataclass
class RollingStepResult:
    step: int
    observed_periods: List[Any]
    fit_periods: List[Any]
    predict_periods: List[Any]
    target_period: Any
    expected_target_period: Optional[Any]

    edges: List[Tuple[int, int]]
    pred_value: np.ndarray                       # (E,)
    true_value: Optional[np.ndarray] = None      # (E,) or None

    n_train_samples: int = 0
    train_set_meta: Dict[str, Any] = field(default_factory=dict)

    mse: float = np.nan
    mae: float = np.nan
    aic_bic: Optional[Dict[str, Any]] = None

    pred_matrix: Optional[np.ndarray] = None
    true_matrix: Optional[np.ndarray] = None

    graph: Optional[EdgeGraph] = None
    learner: Optional[BaseEdgeLearner] = None

    def pred_dict(self) -> Dict[Tuple[int, int], float]:
        return {e: float(v) for e, v in zip(self.edges, self.pred_value)}

    def true_dict(self) -> Optional[Dict[Tuple[int, int], float]]:
        if self.true_value is None:
            return None
        return {e: float(v) for e, v in zip(self.edges, self.true_value)}


# -------------------------------------------------
# 2) Rolling container
# -------------------------------------------------

@dataclass
class RollingEdgePredict:
    graph: EdgeGraph
    learner_cls: Type[BaseEdgeLearner]

    train_periods: List[Any]
    val_periods: Optional[List[Any]] = None

    learner_kwargs: Optional[Dict[str, Any]] = None
    fit_kwargs: Optional[Dict[str, Any]] = None
    predict_kwargs: Optional[Dict[str, Any]] = None

    # Callback that rebuilds the graph at each step; None => fixed graph, no rebuild (assume A unchanged).
    #   graph_builder(observed_periods) -> EdgeGraph
    graph_builder: Optional[Callable[[List[Any]], EdgeGraph]] = None

    # Custom ground-truth getter: (graph, target_period, predict_periods) -> np.ndarray | None
    true_value_getter: Optional[
        Callable[[EdgeGraph, Any, List[Any]], Optional[np.ndarray]]
    ] = None

    # Custom builder of the periods passed to predict_next: (step, fit_periods, observed_periods, learner) -> List
    predict_periods_builder: Optional[
        Callable[[int, List[Any], List[Any], BaseEdgeLearner], List[Any]]
    ] = None

    include_final_forecast: bool = False
    max_train_size: Optional[int] = None
    strict_next_match: bool = False

    return_matrix: bool = False
    matrix_fill_value: float = 0.0

    store_graphs: bool = False
    store_learners: bool = False

    export_learner_params: bool = False
    learner_params_filepath: str = "gnar_params.xlsx"

    step_callbacks: Optional[Any] = None
    step_callback_every: int = 1
    step_callback_on_error: str = "warn"     # "warn" / "raise" / "ignore"

    verbose: bool = False

    steps: List[RollingStepResult] = field(init=False, default_factory=list)

    def __post_init__(self):
        if not isinstance(self.graph, EdgeGraph):
            raise TypeError("graph must be an EdgeGraph instance.")

        self.train_periods = self._normalize_periods(self.train_periods)
        self.val_periods = self._normalize_periods(self.val_periods or [])

        self.learner_kwargs = dict(self.learner_kwargs or {})
        self.fit_kwargs = dict(self.fit_kwargs or {})
        self.predict_kwargs = dict(self.predict_kwargs or {})

        for k in ("graph", "train_periods", "val_periods", "test_periods"):
            self.learner_kwargs.pop(k, None)
        self.fit_kwargs.pop("train_set", None)
        self.predict_kwargs.pop("periods", None)

        if len(self.train_periods) == 0:
            raise ValueError("train_periods must not be empty.")
        if self.max_train_size is not None and int(self.max_train_size) <= 0:
            raise ValueError("max_train_size must be positive.")

        if self.step_callbacks is None:
            self.step_callbacks = []
        elif callable(self.step_callbacks):
            self.step_callbacks = [self.step_callbacks]
        else:
            self.step_callbacks = list(self.step_callbacks)
        self.step_callback_every = max(1, int(self.step_callback_every))
        if self.step_callback_on_error not in {"warn", "raise", "ignore"}:
            raise ValueError("step_callback_on_error must be 'warn'/'raise'/'ignore'.")

    # ---------------- helpers ----------------
    @staticmethod
    def _normalize_periods(periods: List[Any]) -> List[Any]:
        return BaseEdgeLearner._normalize_periods(periods)

    def _build_learner(self, graph: EdgeGraph, fit_periods: List[Any]) -> BaseEdgeLearner:
        return self.learner_cls(
            graph=graph,
            train_periods=fit_periods,
            val_periods=[],
            test_periods=[],
            **self.learner_kwargs,
        )

    @staticmethod
    def _min_predict_history_length(learner: BaseEdgeLearner) -> int:
        return int(getattr(learner, "lookback", 1))

    @staticmethod
    def _time_after(graph: EdgeGraph, t: Any, horizon: int) -> Optional[Any]:
        labels = list(graph.time_labels)
        try:
            i = labels.index(t)
        except ValueError:
            try:
                i = graph.position_of(t)
            except Exception:
                return None
        j = i + int(horizon)
        return labels[j] if 0 <= j < len(labels) else None

    def _resolve_predict_periods(self, step_idx, fit_periods, observed_periods, learner):
        if self.predict_periods_builder is None:
            periods = list(fit_periods)
        else:
            periods = list(self.predict_periods_builder(
                step_idx, list(fit_periods), list(observed_periods), learner))
        periods = self._normalize_periods(periods)
        if len(periods) == 0:
            raise ValueError(f"step={step_idx}: predict_periods is empty.")
        need = self._min_predict_history_length(learner)
        if len(periods) < need:
            raise ValueError(
                f"step={step_idx}: predict_periods length={len(periods)} < required={need}.")
        return periods

    def _default_true_value_getter(self, graph, target_period, predict_periods):
        if target_period is None:
            return None
        # First look for the target in "this step's graph" (transformed ground truth takes priority);
        # if not found, fall back to the global reference graph self.graph (which holds the full time axis).
        for g in (graph, self.graph):
            if target_period in list(g.time_labels):
                return np.asarray(g.edge_series(times=[target_period]), dtype=float).reshape(-1)
        return None

    def _edge_vec_to_matrix(self, graph, edges, edge_vec, *, fill_value=0.0):
        edge_vec = np.asarray(edge_vec, dtype=float).reshape(-1)
        K = int(graph.n_nodes)
        mat = np.full((K, K), fill_value, dtype=float)
        if len(edges) == 0:
            return mat
        ei = np.asarray(edges, dtype=int)
        mat[ei[:, 0], ei[:, 1]] = edge_vec
        return mat

    def _export_step_learner_params(self, learner, step_idx, target_period=None):
        if not self.export_learner_params:
            return
        export_fn = getattr(learner, "export_parameters_to_excel", None)
        if not callable(export_fn):
            return
        kwargs = {"filepath": self.learner_params_filepath, "sheet_name": f"step_{step_idx:03d}"}
        try:
            params = inspect.signature(export_fn).parameters
            if "target_period" in params and target_period is not None:
                kwargs["target_period"] = target_period
        except (TypeError, ValueError):
            pass
        export_fn(**kwargs)

    def _run_step_callbacks(self, step_result, n_steps):
        cbs = list(self.step_callbacks or [])
        if not cbs:
            return
        step_idx = int(step_result.step)
        due = ((step_idx + 1) % self.step_callback_every == 0) or (step_idx == n_steps - 1)
        if not due:
            return
        for cb in cbs:
            try:
                cb(roller=self, step_result=step_result, n_steps=n_steps)
            except Exception as e:
                if self.step_callback_on_error == "raise":
                    raise
                if self.step_callback_on_error == "warn":
                    print(f"[rolling callback] step={step_idx} failed: {type(e).__name__}: {e}", flush=True)

    # ---------------- main loop ----------------
    def run(self) -> "RollingEdgePredict":
        self.steps = []
        fixed_graph = self.graph_builder is None

        n_steps = len(self.val_periods)
        if self.include_final_forecast or n_steps == 0:
            n_steps += 1

        if self.export_learner_params:
            out_dir = os.path.dirname(os.path.abspath(self.learner_params_filepath))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        for step_idx in range(n_steps):
            observed = list(self.train_periods) + list(self.val_periods[:step_idx])
            fit_periods = observed if self.max_train_size is None \
                else observed[-int(self.max_train_size):]

            # --- Obtain the graph for this step ---
            if fixed_graph:
                step_graph = self.graph
            else:
                step_graph = self.graph_builder(list(observed))

            # Build a new learner at every step (guarantees the n iterations are mutually independent, so SGD does not warm-start).
            learner = self._build_learner(step_graph, fit_periods)

            # Fixed-graph optimization: cache the W^(r) computed at the first step back into the main graph,
            # so that when the learner copies the main graph at each subsequent step it brings W along and W is no longer recomputed.
            if fixed_graph and hasattr(self.graph, "_W_cache") and hasattr(learner.graph, "_W_cache"):
                self.graph._W_cache.update(learner.graph._W_cache)

            train_set = learner.build_train_set()
            if train_set.n_samples <= 0:
                raise ValueError(
                    f"step={step_idx}: empty train_set. fit_periods={fit_periods}. "
                    f"Check L / lookback / max_train_size.")

            learner.fit(train_set=train_set, **self.fit_kwargs)

            aic_bic = None
            fn = getattr(learner, "calc_aic_bic", None)
            if callable(fn):
                try:
                    aic_bic = fn(train_set=train_set)
                except TypeError:
                    aic_bic = fn()

            predict_periods = self._resolve_predict_periods(step_idx, fit_periods, observed, learner)
            # Use the global reference graph's time axis for the target time (step_graph may have been truncated by graph_builder to only contain observed periods)
            target_period = self._time_after(self.graph, predict_periods[-1], int(getattr(learner, "horizon", 1)))
            expected_target = self.val_periods[step_idx] if step_idx < len(self.val_periods) else None

            if self.strict_next_match and expected_target is not None and target_period != expected_target:
                raise ValueError(
                    f"step={step_idx}: target={target_period} != expected={expected_target}.")

            pred_value = np.asarray(
                learner.predict_next(periods=predict_periods, **self.predict_kwargs),
                dtype=float).reshape(-1)
            if len(pred_value) != len(learner.edges):
                raise ValueError(
                    f"step={step_idx}: pred dim {len(pred_value)} != n_edges {len(learner.edges)}.")

            getter = self.true_value_getter or self._default_true_value_getter
            true_value = getter(step_graph, target_period, predict_periods)

            mse = mae = np.nan
            if true_value is not None:
                true_value = np.asarray(true_value, dtype=float).reshape(-1)
                if len(true_value) != len(pred_value):
                    raise ValueError(
                        f"step={step_idx}: true dim {len(true_value)} != pred dim {len(pred_value)}.")
                mse = float(np.mean((pred_value - true_value) ** 2))
                mae = float(np.mean(np.abs(pred_value - true_value)))

            pred_matrix = true_matrix = None
            if self.return_matrix:
                pred_matrix = self._edge_vec_to_matrix(step_graph, learner.edges, pred_value,
                                                       fill_value=self.matrix_fill_value)
                if true_value is not None:
                    true_matrix = self._edge_vec_to_matrix(step_graph, learner.edges, true_value,
                                                           fill_value=self.matrix_fill_value)

            self._export_step_learner_params(learner, step_idx, target_period)

            step_result = RollingStepResult(
                step=step_idx,
                observed_periods=list(observed),
                fit_periods=list(fit_periods),
                predict_periods=list(predict_periods),
                target_period=target_period,
                expected_target_period=expected_target,
                edges=list(learner.edges),
                pred_value=pred_value,
                true_value=true_value,
                n_train_samples=int(train_set.n_samples),
                train_set_meta=dict(train_set.meta),
                mse=mse, mae=mae, aic_bic=aic_bic,
                pred_matrix=pred_matrix, true_matrix=true_matrix,
                graph=step_graph if self.store_graphs else None,
                learner=learner if self.store_learners else None,
            )
            self.steps.append(step_result)
            self._run_step_callbacks(step_result, n_steps)

            if self.verbose:
                print(f"[rolling step={step_idx}] fit_end={fit_periods[-1]} "
                      f"target={target_period} n_edges={len(learner.edges)} "
                      f"n_samples={train_set.n_samples} mse={mse}", flush=True)

        return self

    # ---------------- summary ----------------
    def summary(self) -> pd.DataFrame:
        rows = []
        for s in self.steps:
            rows.append({
                "step": s.step,
                "fit_start": s.fit_periods[0], "fit_end": s.fit_periods[-1],
                "target_period": s.target_period,
                "expected_target_period": s.expected_target_period,
                "expected_match": (s.expected_target_period is None
                                   or s.target_period == s.expected_target_period),
                "n_fit_periods": len(s.fit_periods),
                "n_edges": len(s.edges), "n_train_samples": s.n_train_samples,
                "mse": s.mse, "mae": s.mae,
                "AIC": None if s.aic_bic is None else s.aic_bic.get("AIC"),
                "BIC": None if s.aic_bic is None else s.aic_bic.get("BIC"),
            })
        return pd.DataFrame(rows)

    def pred_by_target_period(self) -> Dict[Any, Dict[str, Any]]:
        return {s.target_period: {"edges": s.edges, "pred_value": s.pred_value,
                                  "true_value": s.true_value} for s in self.steps}

    def to_prediction_dfs(self, *, edge_mode: str = "intersection", require_true: bool = True) -> Dict[str, Any]:
        if not self.steps:
            raise RuntimeError("No rolling steps. Run the roller first.")
        edge_sets = [set(s.edges) for s in self.steps]
        if edge_mode == "intersection":
            selected = sorted(set.intersection(*edge_sets)) if edge_sets else []
        elif edge_mode == "union":
            selected = sorted(set().union(*edge_sets)) if edge_sets else []
        else:
            raise ValueError("edge_mode must be 'intersection' or 'union'.")
        if not selected:
            raise RuntimeError("selected_edges is empty.")

        labels = list(self.graph.node_labels)
        cols = pd.MultiIndex.from_tuples([(labels[u], labels[v]) for (u, v) in selected],
                                         names=["payer", "payee"])
        idx = pd.Index([s.target_period for s in self.steps], name="period")
        edge_to_col = {e: j for j, e in enumerate(selected)}

        pred_mat = np.full((len(self.steps), len(selected)), np.nan)
        real_mat = np.full((len(self.steps), len(selected)), np.nan) if require_true else None
        for t, s in enumerate(self.steps):
            for e, v in s.pred_dict().items():
                j = edge_to_col.get(e)
                if j is not None:
                    pred_mat[t, j] = v
            if real_mat is not None and s.true_value is not None:
                for e, v in s.true_dict().items():
                    j = edge_to_col.get(e)
                    if j is not None:
                        real_mat[t, j] = v

        out = {"selected_edges": selected, "pred_df": pd.DataFrame(pred_mat, index=idx, columns=cols),
               "real_df": None, "r2_per_edge": None}
        if real_mat is not None:
            out["real_df"] = pd.DataFrame(real_mat, index=idx, columns=cols)
            r2 = np.array([_r2_metric(real_mat[:, j], pred_mat[:, j]) for j in range(len(selected))])
            out["r2_per_edge"] = pd.Series(r2, index=cols, name="R2")
        return out


# -------------------------------------------------
# 3) Top-level functions
# -------------------------------------------------

def RollingPredict(**kwargs) -> RollingEdgePredict:
    return RollingEdgePredict(**kwargs).run()


def rolling_predict(**kwargs) -> RollingEdgePredict:
    return RollingPredict(**kwargs)
