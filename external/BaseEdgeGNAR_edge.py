from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn

from BaseEdge import EdgeTrainSet
from BaseEdge import BaseEdgeLearner
from BaseEdge import TorchEdgeLearner
import pandas as pd
import os

# Here we assume you already have these two base classes
# from your_module import TorchEdgeLearner, EdgeTrainSet


# -------------------------------------------------
# 1) Pure torch module: GNAR-edge forward formula
# -------------------------------------------------

class GNAREdgeModule(nn.Module):
    """
    Torch module version of GNAR-edge.

    Input:
        X: shape (N, L, E)
           Convention:
             X[:, -1, :] = lag1
             X[:, -2, :] = lag2
             ...
    Output:
        y_pred: shape (N, E)

    Formula:
        X_t = sum_{l=1}^L [ diag(alpha_l) X_{t-l} + sum_{r in R_l} beta_{l,r} W^(r) X_{t-l} ]
    """

    def __init__(
        self,
        *,
        E: int,
        L: int,
        R_by_l: List[List[int]],
        W_dict: Dict[int, np.ndarray],
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.E = int(E)
        self.L = int(L)
        self.R_by_l = [list(map(int, rs)) for rs in R_by_l]

        # Trainable parameters
        self.alpha = nn.Parameter(torch.zeros(self.L, self.E, dtype=dtype))

        self.beta_pairs: List[Tuple[int, int]] = []
        for l in range(1, self.L + 1):
            for r in self.R_by_l[l - 1]:
                self.beta_pairs.append((l, r))

        self.beta_index = {pair: j for j, pair in enumerate(self.beta_pairs)}
        self.beta = nn.Parameter(torch.zeros(len(self.beta_pairs), dtype=dtype))

        # Fixed W^(r), registered as a buffer
        needed_r = sorted({r for rs in self.R_by_l for r in rs})
        for r in needed_r:
            W = np.asarray(W_dict[r], dtype=np.float32)
            self.register_buffer(f"W_{r}", torch.as_tensor(W, dtype=dtype))

    def get_W(self, r: int) -> torch.Tensor:
        return getattr(self, f"W_{r}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X shape = (N, L, E)
        output shape = (N, E)
        """
        N, L_in, E_in = X.shape
        if L_in != self.L:
            raise ValueError(f"Expected lookback L={self.L}, got {L_in}")
        if E_in != self.E:
            raise ValueError(f"Expected E={self.E}, got {E_in}")

        pred = torch.zeros((N, self.E), dtype=X.dtype, device=X.device)

        # X[:, -1, :] = lag1
        # X[:, -2, :] = lag2
        # ...
        for l in range(1, self.L + 1):
            x_lag = X[:, -l, :]  # (N, E)

            # alpha part
            pred = pred + self.alpha[l - 1][None, :] * x_lag

            # beta part
            for r in self.R_by_l[l - 1]:
                beta_idx = self.beta_index[(l, r)]
                beta_lr = self.beta[beta_idx]
                W_r = self.get_W(r)             # (E, E)
                neigh = x_lag @ W_r.T           # (N, E)
                pred = pred + beta_lr * neigh

        return pred

    @torch.no_grad()
    def export_alpha_beta(self) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
        alpha_np = self.alpha.detach().cpu().numpy().copy()
        beta_np = self.beta.detach().cpu().numpy().copy()
        beta_dict = {pair: float(beta_np[j]) for j, pair in enumerate(self.beta_pairs)}
        return alpha_np, beta_dict

    @torch.no_grad()
    def export_psi_list(self) -> List[np.ndarray]:
        alpha_np, beta_dict = self.export_alpha_beta()
        psi_list = []

        for l in range(1, self.L + 1):
            M = np.diag(alpha_np[l - 1].copy())
            for r in self.R_by_l[l - 1]:
                W_r = self.get_W(r).detach().cpu().numpy()
                M += beta_dict[(l, r)] * W_r
            psi_list.append(M)

        return psi_list


# -------------------------------------------------
# 2) GNAR-edge learner
# -------------------------------------------------

@dataclass
class GNAREdgeLearner(TorchEdgeLearner):
    """
    Pure GNAR-edge learner (based on TorchEdgeLearner).

    Accepts only a single EdgeGraph as input:
      - lookback = L, horizon = 1
      - training set = graph.edge_series sliced into "past L periods -> next period"

    Note: this class is "pure algorithm" and contains no data preprocessing.
    STL / growth rate / threshold graph construction etc. all belong to the
    orchestration layer's responsibility -- please prepare the edge weights
    before constructing the EdgeGraph (see the orchestration layer example
    run_generic_example.py).
    """
    L: int = 1
    stages_per_lag: Optional[Union[List[int], List[List[int]]]] = None

    use_ols: bool = False

    def __post_init__(self):
        # A GNAR-edge training sample is exactly the past L periods -> next period
        self.lookback = int(self.L)
        self.horizon = 1

        self.R_by_l = self.normalize_R_by_l(self.stages_per_lag, self.L)

        super().__post_init__()

    # ---------- lag/stage specification ----------
    @staticmethod
    def normalize_R_by_l(
        stages_per_lag: Optional[Union[List[int], List[List[int]]]],
        L: int
    ) -> List[List[int]]:
        if stages_per_lag is None:
            stages_per_lag = [1]

        # List[int] -> maximum stage order
        if all(isinstance(x, (int, np.integer)) for x in stages_per_lag):
            spl = list(map(int, stages_per_lag))
            while len(spl) < L:
                spl.append(spl[-1])
            spl = spl[:L]
            return [list(range(1, r_max + 1)) if r_max > 0 else [] for r_max in spl]

        # List[List[int]] -> explicit set of stage orders
        R_by_l = [list(map(int, rs)) for rs in stages_per_lag]
        while len(R_by_l) < L:
            R_by_l.append(R_by_l[-1])
        R_by_l = R_by_l[:L]
        return [sorted({r for r in rs if r >= 1}) for rs in R_by_l]

    # ---------- model ----------
    def build_module(self) -> nn.Module:
        needed_r = sorted({r for rs in self.R_by_l for r in rs})
        W_dict = {r: self.graph.neighbor_matrix(r) for r in needed_r}

        return GNAREdgeModule(
            E=self.E,
            L=self.L,
            R_by_l=self.R_by_l,
            W_dict=W_dict,
        )

    # ---------- training set / prediction: pure algorithm, directly inherit base class (no STL) ----------
    # build_train_set / build_history_input use the default implementations of BaseEdgeLearner:
    #   use graph.edge_series to slice the past L periods into supervised samples, predicting the next period.
    # Any data preprocessing (STL / growth rate / threshold) is done in the orchestration layer before being fed into the EdgeGraph.

    # ---------- export parameters ----------
    @property
    def alpha(self) -> np.ndarray:
        return self.model.export_alpha_beta()[0]

    @property
    def beta(self) -> Dict[Tuple[int, int], float]:
        return self.model.export_alpha_beta()[1]

    @property
    def Psi_list(self) -> List[np.ndarray]:
        return self.model.export_psi_list()

    def _build_ols_beta_features(
        self,
        X: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        """
        Global beta features for joint OLS.

        Returns:
            beta_pairs: [(l, r), ...]
            Z_list[j]: shape = (N, E)
                Z_list[j][n, e] = (W^(r) x_{t-l})_e
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("X must have shape (N, L, E).")

        beta_pairs = list(self.model.beta_pairs)
        needed_r = sorted({r for (_, r) in beta_pairs})
        W_cache = {
            r: self.model.get_W(r).detach().cpu().numpy().astype(float, copy=False)
            for r in needed_r
        }

        Z_list: List[np.ndarray] = []
        for l, r in beta_pairs:
            x_lag = X[:, -l, :]   # (N, E)
            Z_list.append(x_lag @ W_cache[r].T)

        return beta_pairs, Z_list

    # Overrode the fit function to use the OLS algorithm for added speed and to use FWL and profile OLS
    def _fit_impl(self, train_set: EdgeTrainSet, **kwargs):
        use_ols = bool(kwargs.pop("use_ols", self.use_ols))
        if use_ols:
            return self._fit_impl_ols(train_set, **kwargs)

        return super()._fit_impl(train_set, **kwargs)
    def _fit_impl_ols(self, train_set: EdgeTrainSet, **kwargs):
        """
        Closed-form joint OLS:

            y[n,e]
              = sum_l alpha[l,e] * x_lag_l[n,e]
              + sum_(l,r) beta[l,r] * (W^r x_lag_l[n,:])[e]

        Note:
        - each sample = one complete time node, not a single edge
        - alpha is edge-specific
        - beta is shared across all edges
        So we must perform joint OLS; we cannot fully split each edge into an independent regression.
        """
        verbose = bool(kwargs.get("verbose", self.verbose_fit))
        ols_rcond = float(kwargs.get("ols_rcond", 1e-10))
        if ols_rcond < 0:
            raise ValueError("ols_rcond must be >= 0.")

        X = np.asarray(train_set.X, dtype=float)   # (N, L, E)
        y = np.asarray(train_set.y, dtype=float)   # (N, E)

        if X.ndim != 3:
            raise ValueError(f"train_set.X must have shape (N, L, E), got {X.shape}.")
        if y.ndim != 2:
            raise ValueError(f"train_set.y must have shape (N, E), got {y.shape}.")

        N, L_in, E_in = X.shape
        if N <= 0:
            raise ValueError("train_set is empty.")
        if L_in != self.L:
            raise ValueError(f"Expected lookback L={self.L}, got {L_in}.")
        if E_in != self.E:
            raise ValueError(f"Expected E={self.E}, got {E_in}.")
        if y.shape != (N, self.E):
            raise ValueError(f"Expected y shape {(N, self.E)}, got {y.shape}.")

        # Local alpha design matrix:
        # X_alpha[n, e, :] = [lag1, lag2, ..., lagL]
        X_alpha = np.transpose(X[:, ::-1, :], (0, 2, 1))   # (N, E, L)

        beta_pairs, Z_list = self._build_ols_beta_features(X)
        B = len(beta_pairs)
        Z_stack = np.stack(Z_list, axis=2) if B > 0 else None   # (N, E, B)

        # First cache each edge's local lag design matrix and its pinv
        # pinv is used so that even under rank deficiency / collinearity we still get a stable minimum-norm OLS solution
        X_e_list = [X_alpha[:, e, :] for e in range(self.E)]               # E * (N, L)
        X_pinv_list = [np.linalg.pinv(X_e, rcond=ols_rcond) for X_e in X_e_list]

        alpha_hat = np.zeros((self.L, self.E), dtype=float)
        beta_hat = np.zeros(B, dtype=float)

        if B > 0:
            # Step 1:
            # Use FWL / partial-out to estimate the shared beta first
            # i.e. remove each edge's own alpha part from y and Z first
            S = np.zeros((B, B), dtype=float)
            rhs = np.zeros(B, dtype=float)

            for e in range(self.E):
                X_e = X_e_list[e]               # (N, L)
                X_pinv = X_pinv_list[e]         # (L, N)
                y_e = y[:, e]                   # (N,)
                Z_e = Z_stack[:, e, :]          # (N, B)

                # residualize Z_e on X_e
                MZ = Z_e - X_e @ (X_pinv @ Z_e)

                # residualize y_e on X_e
                My = y_e - X_e @ (X_pinv @ y_e)

                S += MZ.T @ MZ
                rhs += MZ.T @ My

            beta_hat = np.linalg.pinv(S, rcond=ols_rcond) @ rhs
            self.ols_beta_rank_ = int(np.linalg.matrix_rank(S))

            if verbose and self.ols_beta_rank_ < B:
                print(
                    f"[{self.__class__.__name__}] "
                    f"warning: beta block rank deficient "
                    f"({self.ols_beta_rank_}/{B}); "
                    f"beta coefficients may not be unique."
                )

            # Step 2:
            # With beta fixed, recover alpha_e edge by edge
            for e in range(self.E):
                X_pinv = X_pinv_list[e]
                y_e = y[:, e]
                Z_e = Z_stack[:, e, :]

                alpha_hat[:, e] = X_pinv @ (y_e - Z_e @ beta_hat)
        else:
            # With no beta terms, this degenerates to an order-L OLS per edge
            self.ols_beta_rank_ = 0
            for e in range(self.E):
                X_pinv = X_pinv_list[e]
                y_e = y[:, e]
                alpha_hat[:, e] = X_pinv @ y_e

        # Write back to the torch parameters
        # This way existing interfaces such as predict / export / AIC-BIC need no changes
        with torch.no_grad():
            self.model.alpha.copy_(
                torch.as_tensor(
                    alpha_hat,
                    dtype=self.model.alpha.dtype,
                    device=self.model.alpha.device,
                )
            )
            self.model.beta.copy_(
                torch.as_tensor(
                    beta_hat,
                    dtype=self.model.beta.dtype,
                    device=self.model.beta.device,
                )
            )

        # Record the training error
        with torch.no_grad():
            y_pred = self.forward(X)
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        y_pred = np.asarray(y_pred, dtype=float)

        resid = y - y_pred
        mse = float(np.mean(resid ** 2))

        self.ols_method_ = "joint_profile_ols"
        self.ols_rcond_ = float(ols_rcond)
        self.ols_beta_pairs_ = list(beta_pairs)
        self.ols_residuals_ = resid
        self.ols_rss_ = float(np.sum(resid ** 2))
        self.ols_mse_ = mse
        self.ols_nobs_ = int(N * self.E)
        self.ols_k_params_ = int(self._num_mean_params())

        self.last_fit_best_loss = mse
        self.last_fit_best_epoch = 1
        self.last_fit_epochs = 1
        self.last_fit_stop_reason = "ols_closed_form"

        self._record_epoch(
            epoch=1,
            loss=mse,
            epoch_seconds=0.0,
            abs_improve=np.nan,
            rel_improve=np.nan,
            best_loss_so_far=mse,
            no_improve_rounds=0,
            low_gain_rounds=0,
            verbose=verbose,
            print_every=1,
        )

    def predict_train_set(self, train_set: Optional[EdgeTrainSet] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if train_set is None:
            train_set = self.train_set
        if train_set is None:
            raise RuntimeError("train_set is None. Fit the model first or pass a train_set explicitly.")

        y_pred = self.forward(train_set.X)
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        y_pred = np.asarray(y_pred, dtype=float)
        return y_pred.reshape(train_set.y.shape)

    def conditional_residuals(self, train_set: Optional[EdgeTrainSet] = None) -> np.ndarray:
        if train_set is None:
            train_set = self.train_set
        if train_set is None:
            raise RuntimeError("train_set is None. Fit the model first or pass a train_set explicitly.")

        y_true = np.asarray(train_set.y, dtype=float)
        y_pred = self.predict_train_set(train_set=train_set)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true shape={y_true.shape} != y_pred shape={y_pred.shape}.")

        return y_true - y_pred

    def _num_mean_params(self) -> int:
        return int(self.alpha.size) + int(len(self.beta))

    def calc_aic_bic(
        self,
        train_set: Optional[EdgeTrainSet] = None,
        covariance: str = "scalar",
        bic_nobs: str = "vector",
        eps: float = 1e-12,
    ) -> Dict[str, Union[float, int, np.ndarray]]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        R = self.conditional_residuals(train_set=train_set)
        n_samples, n_edges = R.shape
        p = self._num_mean_params()

        if covariance == "scalar":
            n_obs = n_samples * n_edges
            rss = float(np.sum(R ** 2))
            sigma2_hat = max(rss / n_obs, eps)
            loglik = -0.5 * n_obs * (np.log(2.0 * np.pi) + 1.0 + np.log(sigma2_hat))
            k = p + 1
            extra = {
                "rss": rss,
                "sigma2_hat": float(sigma2_hat),
            }
        elif covariance == "diag":
            sigma2_by_edge = np.mean(R ** 2, axis=0)
            sigma2_by_edge = np.maximum(sigma2_by_edge, eps)

            loglik = (
                -0.5 * n_samples * n_edges * np.log(2.0 * np.pi)
                -0.5 * n_samples * np.sum(np.log(sigma2_by_edge))
                -0.5 * np.sum((R ** 2) / sigma2_by_edge[None, :])
            )
            k = p + n_edges
            extra = {
                "sigma2_hat_by_edge": sigma2_by_edge,
            }
        else:
            raise ValueError("covariance must be one of {'scalar', 'diag'}")

        if bic_nobs == "vector":
            nobs_for_bic = n_samples
        elif bic_nobs == "flattened":
            nobs_for_bic = n_samples * n_edges
        else:
            raise ValueError("bic_nobs must be one of {'vector', 'flattened'}")

        AIC = 2.0 * k - 2.0 * loglik
        BIC = np.log(nobs_for_bic) * k - 2.0 * loglik

        out = {
            "loglik": float(loglik),
            "AIC": float(AIC),
            "BIC": float(BIC),
            "k_params": int(k),
            "mean_params": int(p),
            "nobs_for_bic": int(nobs_for_bic),
            "n_samples": int(n_samples),
            "n_edges": int(n_edges),
            "covariance": covariance,
            "bic_nobs_mode": bic_nobs,
        }
        out.update(extra)
        return out

    # ---------- compatibility with the old interface ----------
    def predict_next_edge_vec(self, periods: Optional[List[pd.Period]] = None) -> np.ndarray:
        return self.predict_next(periods=periods)

    def predict_next_matrix(self, periods: Optional[List[pd.Period]] = None) -> np.ndarray:
        pred_vec = self.predict_next(periods=periods)

        K = self.graph.n_nodes
        mat = np.zeros((K, K), dtype=float)

        edge_idx = np.asarray(self.edges, dtype=int)
        u_idx = edge_idx[:, 0]
        v_idx = edge_idx[:, 1]
        mat[u_idx, v_idx] = pred_vec
        return mat

    def get_parameters(self, include_psi: bool = False) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        alpha_np, beta_dict = self.model.export_alpha_beta()

        out = {
            "alpha": alpha_np.copy(),
            "beta": {(int(l), int(r)): float(v) for (l, r), v in beta_dict.items()},
        }

        if include_psi:
            out["Psi_list"] = [psi.copy() for psi in self.model.export_psi_list()]

        return out

    def _parameters_to_text(
        self,
        include_psi: bool = False,
        precision: int = 6,
    ) -> str:
        params = self.get_parameters(include_psi=include_psi)

        lines = [
            f"class = {self.__class__.__name__}",
            f"is_fitted = {self.is_fitted}",
            f"E = {self.E}",
            f"L = {self.L}",
            f"R_by_l = {self.R_by_l}",
            "alpha =",
            np.array2string(
                params["alpha"],
                precision=precision,
                suppress_small=False,
                threshold=10**9,
            ),
            "beta =",
        ]

        beta_dict = params["beta"]
        if beta_dict:
            for (l, r), v in sorted(beta_dict.items()):
                lines.append(f"  beta[{l},{r}] = {v:.{precision}f}")
        else:
            lines.append("  {}")

        if include_psi:
            psi_list = params["Psi_list"]
            for i, psi in enumerate(psi_list, start=1):
                lines.append(f"Psi[{i}] =")
                lines.append(
                    np.array2string(
                        psi,
                        precision=precision,
                        suppress_small=False,
                        threshold=10**9,
                    )
                )

        return "\n".join(lines)

    def print_parameters(
        self,
        include_psi: bool = False,
        precision: int = 6,
    ):
        print(self._parameters_to_text(
            include_psi=include_psi,
            precision=precision,
        ))

    def save_parameters_to_file(
        self,
        filepath: str,
        *,
        include_psi: bool = False,
        precision: int = 6,
        encoding: str = "utf-8",
    ) -> str:
        text = self._parameters_to_text(
            include_psi=include_psi,
            precision=precision,
        )
        with open(filepath, "w", encoding=encoding) as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        return filepath

    def _edge_labels(self) -> List[str]:
        codes = list(self.graph.node_labels)
        labels = []
        for u, v in self.edges:
            labels.append(f"{codes[u]}->{codes[v]}")
        return labels

    def export_parameters_to_excel(
            self,
            filepath: str,
            sheet_name: str,
            *args,
            **kwargs,
    ) -> str:
        """
        Export current learner parameters to one Excel sheet.

        Key points:
        - Each sheet calls DataFrame.to_excel(...) only once
        - Because when mode="a" and if_sheet_exists="replace",
          calling to_excel multiple times for the same sheet repeatedly replaces the sheet.
        - The original approach wrote meta first, then alpha, then beta;
          from step_001 onward the file already exists, so in the end only beta remained.
        """
        include_psi = bool(kwargs.get("include_psi", False))

        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        filepath = str(filepath)
        out_dir = os.path.dirname(os.path.abspath(filepath))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        sheet_name = str(sheet_name).strip()
        if len(sheet_name) == 0:
            raise ValueError("sheet_name must not be empty.")

        params = self.get_parameters(include_psi=include_psi)
        edge_labels = self._edge_labels()

        meta_df = pd.DataFrame({
            "field": [
                "class",
                "is_fitted",
                "E",
                "L",
                "R_by_l",
                "use_stl",
                "measure",
                "lookback",
                "horizon",
            ],
            "value": [
                self.__class__.__name__,
                bool(self.is_fitted),
                int(self.E),
                int(self.L),
                str(self.R_by_l),
                bool(self.use_stl),
                self.measure,
                int(self.lookback),
                int(self.horizon),
            ],
        })

        alpha_df = pd.DataFrame(
            np.asarray(params["alpha"], dtype=float),
            index=[f"lag_{l}" for l in range(1, self.L + 1)],
            columns=edge_labels,
        )
        alpha_df.index.name = "lag"

        beta_rows = [
            {
                "lag": int(l),
                "stage": int(r),
                "beta": float(v),
            }
            for (l, r), v in sorted(params["beta"].items())
        ]
        beta_df = pd.DataFrame(beta_rows, columns=["lag", "stage", "beta"])

        def _cell_value(x):
            if pd.isna(x):
                return None
            if isinstance(x, np.generic):
                return x.item()
            return x

        def _append_blank(rows: List[List[Any]], n: int = 2) -> None:
            for _ in range(int(n)):
                rows.append([])

        def _append_df_section(
                rows: List[List[Any]],
                title: str,
                df: pd.DataFrame,
                *,
                include_index: bool,
        ) -> None:
            rows.append([title])

            if include_index:
                rows.append([df.index.name or ""] + [str(c) for c in df.columns])
                for idx, row in df.iterrows():
                    rows.append([idx] + [_cell_value(v) for v in row.tolist()])
            else:
                rows.append([str(c) for c in df.columns])
                for row in df.itertuples(index=False, name=None):
                    rows.append([_cell_value(v) for v in row])

            _append_blank(rows, 2)

        rows: List[List[Any]] = []

        _append_df_section(
            rows,
            "meta",
            meta_df,
            include_index=False,
        )

        _append_df_section(
            rows,
            "alpha",
            alpha_df,
            include_index=True,
        )

        _append_df_section(
            rows,
            "beta",
            beta_df,
            include_index=False,
        )

        if include_psi:
            psi_list = params["Psi_list"]

            for i, psi in enumerate(psi_list, start=1):
                psi_df = pd.DataFrame(
                    np.asarray(psi, dtype=float),
                    index=edge_labels,
                    columns=edge_labels,
                )
                psi_df.index.name = "edge"

                _append_df_section(
                    rows,
                    f"Psi_{i}",
                    psi_df,
                    include_index=True,
                )

        max_cols = max((len(r) for r in rows), default=1)
        rows = [r + [None] * (max_cols - len(r)) for r in rows]
        out_df = pd.DataFrame(rows)

        file_exists = os.path.exists(filepath)

        writer_kwargs = {
            "engine": "openpyxl",
            "mode": "a" if file_exists else "w",
        }

        if file_exists:
            writer_kwargs["if_sheet_exists"] = "replace"

        with pd.ExcelWriter(filepath, **writer_kwargs) as writer:
            out_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                header=False,
            )

        return filepath

