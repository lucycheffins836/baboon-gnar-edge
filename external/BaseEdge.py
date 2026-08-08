from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

from edge_graph import EdgeGraph, ArrayEdgeGraph


def _coerce_to_graph(obj) -> EdgeGraph:
    """
    Normalise the incoming object into an EdgeGraph:
      - if it is already an EdgeGraph, use it directly;
      - if it is the legacy ThresholdStaticGraphTS (duck-typed: has .ipg + .build_W), wrap it with an adapter.
    This way the algorithm only depends on the EdgeGraph interface, and the TSG is just one of the possible sources.
    """
    if isinstance(obj, EdgeGraph):
        return obj
    if hasattr(obj, "ipg") and hasattr(obj, "build_W") and hasattr(obj, "edges"):
        return ArrayEdgeGraph.from_tsg(obj)
    raise TypeError(
        f"learner expects an EdgeGraph (or a legacy TSG), got {type(obj).__name__}."
    )

# Optional: only the Torch subclasses need this
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


# ---------------------------
# 1) Standard training-set container
# ---------------------------

@dataclass
class EdgeTrainSet:
    """
    Standard machine-learning training set:
      X: (N, lookback, E)
      y: (N, E)

    Here each sample is:
      input  = all edge values over the past `lookback` periods
      label  = all edge values at the next period (or `horizon` periods later)
    """
    X: np.ndarray
    y: np.ndarray
    edges: List[Tuple[int, int]]
    sample_target_periods: List[pd.Period] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def flatten_X(self) -> np.ndarray:
        return self.X.reshape(self.X.shape[0], -1)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.X.shape[2])


# ---------------------------
# 2) Framework-agnostic base class
# ---------------------------

@dataclass
class BaseEdgeLearner(ABC):
    """
    The unified base class for all learning models built on a "fixed graph + time-varying edge weights".

    It only depends on the EdgeGraph interface (edges / n_nodes / time / edge_series / neighbor_matrix),
    and is fully decoupled from the concrete data source (TSG / ipg / Excel / numpy).
    """
    graph: EdgeGraph

    train_periods: Optional[List[pd.Period]] = None
    val_periods: Optional[List[pd.Period]] = None
    test_periods: Optional[List[pd.Period]] = None

    measure: str = "value"
    lookback: int = 1
    horizon: int = 1

    is_fitted: bool = field(init=False, default=False)

    edges: List[Tuple[int, int]] = field(init=False)
    edge_to_idx: Dict[Tuple[int, int], int] = field(init=False)
    E: int = field(init=False)

    train_set: Optional[EdgeTrainSet] = field(init=False, default=None)

    # Training debug information
    fit_history: List[Dict[str, Any]] = field(init=False, default_factory=list)
    last_train_set_build_seconds: float = field(init=False, default=np.nan)
    last_fit_seconds: float = field(init=False, default=0.0)
    last_fit_epochs: int = field(init=False, default=0)
    last_fit_best_loss: Optional[float] = field(init=False, default=None)
    last_fit_best_epoch: int = field(init=False, default=0)
    last_fit_stop_reason: str = field(init=False, default="not_started")

    def __post_init__(self):
        self.graph = _coerce_to_graph(self.graph).copy()

        self.train_periods = self._normalize_periods(
            self.train_periods if self.train_periods is not None else self.graph.time_labels
        )
        self.val_periods = self._normalize_periods(self.val_periods or [])
        self.test_periods = self._normalize_periods(self.test_periods or [])

        self.edges = list(self.graph.edges)
        self.edge_to_idx = dict(self.graph.edge_to_idx)
        self.E = len(self.edges)

    # ---------- Basic data interface ----------

    @staticmethod
    def _normalize_periods(periods: Iterable[Any]) -> List[Any]:
        # Keep the time labels in their original type (pd.Period / int / date are all fine); only sort them.
        # No longer force-convert to monthly-frequency Period, thereby removing the coupling to monthly data.
        return sorted(periods)

    def _extract_edge_matrix(
        self,
        periods: List[pd.Period],
        *,
        edges: Optional[List[Tuple[int, int]]] = None,
        measure: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extract from the graph the edge matrix corresponding to the given periods and given edges:
            X.shape = (T, E)

        It now uniformly goes through EdgeGraph.edge_series; the `measure` parameter is kept only for
        backward compatibility with old calls, and under the single-channel (1-D) case it no longer
        distinguishes value / transactions.
        """
        periods = self._normalize_periods(periods)
        edges = self.edges if edges is None else edges
        return self.graph.edge_series(times=periods, edges=edges)

    def build_history_input(
        self,
        periods: List[pd.Period],
    ) -> np.ndarray:
        """
        The input used by predict_next:
            shape = (1, lookback, E)
        """
        X = self._extract_edge_matrix(periods)
        return X[-self.lookback:, :][None, :, :]

    # ---------- Standard training-set generation ----------

    def build_train_set(self) -> EdgeTrainSet:
        """
        Default training-set generation logic:
            take the edge matrix over train_periods and slice it into standard supervised-learning samples.
        """
        periods = list(self.train_periods)
        X_full = self._extract_edge_matrix(periods)

        X_rows = []
        y_rows = []
        y_periods = []

        for end in range(self.lookback, len(periods) - self.horizon + 1):
            x_win = X_full[end - self.lookback:end, :]
            y_val = X_full[end + self.horizon - 1, :]
            y_p = periods[end + self.horizon - 1]

            X_rows.append(x_win)
            y_rows.append(y_val)
            y_periods.append(y_p)

        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y_rows, dtype=float)

        return EdgeTrainSet(
            X=X,
            y=y,
            edges=list(self.edges),
            sample_target_periods=y_periods,
            meta={
                "measure": self.measure,
                "lookback": self.lookback,
                "horizon": self.horizon,
                "train_periods": list(periods),
            }
        )

    def make_train_set(self) -> EdgeTrainSet:
        """
        Training-set construction entry point with timing.
        """
        t0 = time.perf_counter()
        train_set = self.build_train_set()
        elapsed = time.perf_counter() - t0

        self.last_train_set_build_seconds = float(elapsed)

        meta = dict(train_set.meta or {})
        meta["train_set_build_seconds"] = float(elapsed)
        train_set.meta = meta

        return train_set

    # ---------- Unified training/prediction interface ----------

    def _reset_fit_stats(self):
        self.fit_history = []
        self.last_fit_seconds = 0.0
        self.last_fit_epochs = 0
        self.last_fit_best_loss = None
        self.last_fit_best_epoch = 0
        self.last_fit_stop_reason = "running"

    def _record_epoch(
        self,
        epoch: int,
        loss: float,
        epoch_seconds: float,
        *,
        abs_improve: Optional[float] = None,
        rel_improve: Optional[float] = None,
        best_loss_so_far: Optional[float] = None,
        no_improve_rounds: Optional[int] = None,
        low_gain_rounds: Optional[int] = None,
        verbose: bool = True,
        print_every: int = 10,
    ):
        row = {
            "epoch": int(epoch),
            "loss": float(loss),
            "epoch_seconds": float(epoch_seconds),
        }

        if abs_improve is not None and np.isfinite(abs_improve):
            row["abs_improve"] = float(abs_improve)
        if rel_improve is not None and np.isfinite(rel_improve):
            row["rel_improve"] = float(rel_improve)
        if best_loss_so_far is not None and np.isfinite(best_loss_so_far):
            row["best_loss_so_far"] = float(best_loss_so_far)
        if no_improve_rounds is not None:
            row["no_improve_rounds"] = int(no_improve_rounds)
        if low_gain_rounds is not None:
            row["low_gain_rounds"] = int(low_gain_rounds)

        self.fit_history.append(row)
        self.last_fit_epochs = int(epoch)

        if verbose and (epoch == 1 or epoch % max(1, int(print_every)) == 0):
            extra = ""
            if rel_improve is not None and np.isfinite(rel_improve):
                extra += f" rel_improve={float(rel_improve):.6f}"
            if no_improve_rounds is not None:
                extra += f" no_improve_rounds={int(no_improve_rounds)}"
            if low_gain_rounds is not None:
                extra += f" low_gain_rounds={int(low_gain_rounds)}"

            print(
                f"[{self.__class__.__name__}] "
                f"epoch={int(epoch):03d} "
                f"loss={float(loss):.8f} "
                f"time={float(epoch_seconds):.3f}s"
                f"{extra}"
            )

    def fit(self, train_set: Optional[EdgeTrainSet] = None, **kwargs):
        """
        Unified entry point:
          1) if no train_set is passed, automatically call make_train_set()
          2) then hand it off to the subclass-implemented _fit_impl(...)
        """
        if train_set is None:
            train_set = self.make_train_set()
        else:
            meta = dict(train_set.meta or {})
            build_seconds = meta.get("train_set_build_seconds", np.nan)
            try:
                self.last_train_set_build_seconds = float(build_seconds)
            except Exception:
                self.last_train_set_build_seconds = np.nan

        self.train_set = train_set
        self._reset_fit_stats()

        fit_t0 = time.perf_counter()
        self._fit_impl(train_set, **kwargs)
        self.last_fit_seconds = time.perf_counter() - fit_t0

        if self.last_fit_stop_reason == "running":
            self.last_fit_stop_reason = "completed"

        verbose = bool(kwargs.get("verbose", getattr(self, "verbose_fit", True)))
        if verbose:
            print(
                f"[{self.__class__.__name__}] "
                f"fit_done epochs={self.last_fit_epochs} "
                f"best_epoch={self.last_fit_best_epoch} "
                f"best_loss={self.last_fit_best_loss} "
                f"train_set_time={self.last_train_set_build_seconds:.3f}s "
                f"fit_time={self.last_fit_seconds:.3f}s "
                f"stop={self.last_fit_stop_reason}"
            )

        self.is_fitted = True
        return self

    @abstractmethod
    def _fit_impl(self, train_set: EdgeTrainSet, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, X: Any, **kwargs) -> Any:
        raise NotImplementedError

    def predict_next(self, periods: Optional[List[pd.Period]] = None, **kwargs) -> np.ndarray:
        """
        Predict the next period using the last `lookback` periods of the given periods.
        By default periods=train_periods.
        """
        if periods is None:
            periods = self.train_periods

        X_in = self.build_history_input(periods)

        if hasattr(self, "model") and self.model is not None:
            self.model.eval()

        if torch is not None:
            with torch.no_grad():
                out = self.forward(X_in, **kwargs)
        else:
            out = self.forward(X_in, **kwargs)

        if torch is not None and isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()

        out = np.asarray(out, dtype=float)
        return out.reshape(-1)

    # ---------- Some optional extension points ----------

    def loss(self, y_pred: Any, y_true: Any):
        raise NotImplementedError("BaseEdgeLearner.loss is not implemented.")

    def save_state(self) -> Dict[str, Any]:
        return {}

    def load_state(self, state: Dict[str, Any]):
        return self


# ---------------------------
# 3) Torch automatic-differentiation subclass
# ---------------------------

@dataclass
class TorchEdgeLearner(BaseEdgeLearner):
    """
    A general subclass that uses Torch for automatic differentiation.
    """
    device: str = "cpu"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 100

    # Existing early stopping
    use_early_stop: bool = True
    early_stop_epsilon: float = 1e-6
    early_stop_rounds: int = 10

    # New: convergence stopping based on "improvement rate too small"
    use_convergence_stop: bool = False
    convergence_rel_improve_threshold: float = 2e-3
    convergence_rounds: int = 5

    verbose_fit: bool = True
    print_every: int = 10
    restore_best_state: bool = True

    model: Any = field(init=False, default=None)
    optimizer: Any = field(init=False, default=None)
    criterion: Any = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()

        if torch is None or nn is None:
            raise ImportError("PyTorch is not installed, but TorchEdgeLearner was used.")

        self.device = str(self.device)
        self.model = self.build_module().to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.criterion = nn.MSELoss()

    @abstractmethod
    def build_module(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, X: Any, **kwargs) -> torch.Tensor:
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self.model(X)

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self.criterion(y_pred, y_true)

    def _fit_impl(self, train_set: EdgeTrainSet, **kwargs):
        max_epochs = int(kwargs.get("max_epochs", self.max_epochs))

        use_early_stop = bool(kwargs.get("use_early_stop", self.use_early_stop))
        early_stop_epsilon = float(
            kwargs.get("early_stop_epsilon", kwargs.get("epsilon", self.early_stop_epsilon))
        )
        early_stop_rounds = int(kwargs.get("early_stop_rounds", self.early_stop_rounds))

        use_convergence_stop = bool(kwargs.get("use_convergence_stop", self.use_convergence_stop))
        convergence_rel_improve_threshold = float(
            kwargs.get(
                "convergence_rel_improve_threshold",
                self.convergence_rel_improve_threshold,
            )
        )
        convergence_rounds = int(kwargs.get("convergence_rounds", self.convergence_rounds))

        verbose = bool(kwargs.get("verbose", self.verbose_fit))
        print_every = int(kwargs.get("print_every", self.print_every))
        restore_best_state = bool(kwargs.get("restore_best_state", self.restore_best_state))

        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if early_stop_epsilon < 0:
            raise ValueError("early_stop_epsilon must be >= 0.")
        if early_stop_rounds < 0:
            raise ValueError("early_stop_rounds must be >= 0.")
        if convergence_rel_improve_threshold < 0:
            raise ValueError("convergence_rel_improve_threshold must be >= 0.")
        if convergence_rounds < 0:
            raise ValueError("convergence_rounds must be >= 0.")
        if train_set.n_samples <= 0:
            raise ValueError("train_set is empty.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        X = torch.as_tensor(train_set.X, dtype=torch.float32)
        y = torch.as_tensor(train_set.y, dtype=torch.float32)

        effective_batch_size = max(1, min(int(self.batch_size), int(train_set.n_samples)))

        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=effective_batch_size,
            shuffle=True,
        )

        # best_state is saved using the true minimum loss
        best_loss_raw = float("inf")
        best_epoch = 0
        best_state = None

        # The existing early stopping accumulates based on "significant improvement"
        patience_best_loss = float("inf")
        no_improve_rounds = 0

        # The new convergence stopping accumulates based on "the improvement rate between two adjacent epochs is too small"
        prev_epoch_loss = None
        low_gain_rounds = 0

        self.model.train()

        for epoch in range(1, max_epochs + 1):
            epoch_t0 = time.perf_counter()
            loss_sum = 0.0
            sample_count = 0

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                y_pred = self.forward(xb)
                loss = self.loss(y_pred, yb)
                loss.backward()
                self.optimizer.step()

                bs = int(xb.shape[0])
                loss_sum += float(loss.detach().item()) * bs
                sample_count += bs

            epoch_loss = loss_sum / max(1, sample_count)
            epoch_seconds = time.perf_counter() - epoch_t0

            # 1) the best state is always saved by the true minimum loss
            if epoch_loss < best_loss_raw:
                best_loss_raw = float(epoch_loss)
                best_epoch = int(epoch)
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

            # 2) existing early stopping: judged relative to the "significant improvement threshold"
            if epoch_loss < patience_best_loss - early_stop_epsilon:
                patience_best_loss = float(epoch_loss)
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            # 3) new convergence stopping: look at the improvement rate between two adjacent epochs
            abs_improve = np.nan
            rel_improve = np.nan

            if prev_epoch_loss is not None:
                abs_improve = float(prev_epoch_loss - epoch_loss)
                effective_improve = max(abs_improve, 0.0)
                rel_improve = effective_improve / max(abs(float(prev_epoch_loss)), 1e-12)

                if use_convergence_stop and convergence_rounds > 0:
                    if rel_improve < convergence_rel_improve_threshold:
                        low_gain_rounds += 1
                    else:
                        low_gain_rounds = 0
            else:
                low_gain_rounds = 0

            self._record_epoch(
                epoch=epoch,
                loss=epoch_loss,
                epoch_seconds=epoch_seconds,
                abs_improve=abs_improve,
                rel_improve=rel_improve,
                best_loss_so_far=best_loss_raw,
                no_improve_rounds=no_improve_rounds,
                low_gain_rounds=low_gain_rounds,
                verbose=verbose,
                print_every=print_every,
            )

            prev_epoch_loss = float(epoch_loss)

            # Check "convergence stopping" first, since this is the new logic you wanted
            if use_convergence_stop and convergence_rounds > 0 and low_gain_rounds >= convergence_rounds:
                self.last_fit_stop_reason = (
                    "convergence_stop("
                    f"rel_improve<{convergence_rel_improve_threshold}, "
                    f"rounds={convergence_rounds})"
                )
                if verbose:
                    print(
                        f"[{self.__class__.__name__}] "
                        f"convergence_stop at epoch={epoch} "
                        f"best_epoch={best_epoch} "
                        f"best_loss={best_loss_raw:.8f}"
                    )
                break

            # Then check the existing early stopping
            if use_early_stop and early_stop_rounds > 0 and no_improve_rounds >= early_stop_rounds:
                self.last_fit_stop_reason = (
                    "early_stop("
                    f"epsilon={early_stop_epsilon}, "
                    f"rounds={early_stop_rounds})"
                )
                if verbose:
                    print(
                        f"[{self.__class__.__name__}] "
                        f"early_stop at epoch={epoch} "
                        f"best_epoch={best_epoch} "
                        f"best_loss={best_loss_raw:.8f}"
                    )
                break
        else:
            self.last_fit_stop_reason = "max_epochs"

        if restore_best_state and best_state is not None:
            self.model.load_state_dict(best_state)

        if np.isfinite(best_loss_raw):
            self.last_fit_best_loss = float(best_loss_raw)
            self.last_fit_best_epoch = int(best_epoch)

    def save_state(self) -> Dict[str, Any]:
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }

    def load_state(self, state: Dict[str, Any]):
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        return self

    def print_parameters(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__}.print_parameters is not implemented."
        )

    def export_parameters_to_excel(
            self,
            filepath: str,
            sheet_name: str,
            *args,
            **kwargs,
    ) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__}.export_parameters_to_excel is not implemented."
        )

    def get_parameters(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__}.get_parameters is not implemented."
        )

    def debug_exports(self,**kwargs) -> Any:
        pass


# ---------------------------
# 4) A simple example: MLP
# ---------------------------

@dataclass
class TorchMLPEdgeLearner(TorchEdgeLearner):
    hidden_dim: int = 128
    depth: int = 2
    dropout: float = 0.0

    def build_module(self) -> nn.Module:
        input_dim = self.lookback * self.E
        output_dim = self.E

        layers: List[nn.Module] = [nn.Flatten()]
        dim_in = input_dim

        for _ in range(self.depth):
            layers.append(nn.Linear(dim_in, self.hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            dim_in = self.hidden_dim

        layers.append(nn.Linear(dim_in, output_dim))
        return nn.Sequential(*layers)