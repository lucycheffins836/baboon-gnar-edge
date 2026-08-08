"""
BaseEdgeGNAR_edge_global.py  -- global-coefficient GNAR-edge (the "global GNAR-edge model")
===========================================================================================

The "global GNAR-edge model" from page 7 of the paper: each lag l has a single
network-wide shared alpha_l (no longer per-edge), while beta_{l,r} stays shared
as before. The parameter count shrinks from L*E + B to L + B.

Called exactly like GNAREdgeLearner, just a different class name:

    learner = GNAREdgeGlobalLearner(graph=g, L=3, stages_per_lag=[[1],[1,2],[1,2]],
                                    use_ols=True)
    learner.fit(use_ols=True)
    learner.predict_next()

Inheritance (a pragmatic choice for now; see the design-discussion chapter of the
docs for the long-term "coefficient spec" family base class):
    BaseEdgeLearner -> TorchEdgeLearner -> GNAREdgeLearner -> GNAREdgeGlobalLearner

Reused from the parent: training-set construction / SGD training / AIC-BIC /
residuals / rolling interface / beta feature construction.
Overridden: build_module (global module), _fit_impl_ols (pooled OLS with merged
design columns), export_parameters_to_excel (alpha shape changes from (L,E) to (L,)).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from BaseEdge import EdgeTrainSet
from BaseEdgeGNAR_edge import GNAREdgeLearner


# -------------------------------------------------
# 1) torch module: forward equation with global alpha
# -------------------------------------------------

class GNAREdgeGlobalModule(nn.Module):
    """
    Global-coefficient GNAR-edge forward equation.

    Input X: (N, L, E), with the convention X[:, -l, :] = lag l.
    Output y_pred: (N, E).

    Equation (same as GNAREdgeModule, except alpha goes from (L,E) to (L,)):
        X_t = sum_{l=1}^L [ alpha_l * X_{t-l} + sum_{r in R_l} beta_{l,r} * W^(r) X_{t-l} ]

    alpha_l is a scalar (network-wide, broadcast over all edges); beta_{l,r} is
    network-wide shared as in the original.
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

        # Trainable parameters: one scalar alpha per lag; one scalar beta per (l, r)
        self.alpha = nn.Parameter(torch.zeros(self.L, dtype=dtype))

        self.beta_pairs: List[Tuple[int, int]] = []
        for l in range(1, self.L + 1):
            for r in self.R_by_l[l - 1]:
                self.beta_pairs.append((l, r))

        self.beta_index = {pair: j for j, pair in enumerate(self.beta_pairs)}
        self.beta = nn.Parameter(torch.zeros(len(self.beta_pairs), dtype=dtype))

        # Fixed W^(r), registered as buffers (same as the original)
        needed_r = sorted({r for rs in self.R_by_l for r in rs})
        for r in needed_r:
            W = np.asarray(W_dict[r], dtype=np.float32)
            self.register_buffer(f"W_{r}", torch.as_tensor(W, dtype=dtype))

    def get_W(self, r: int) -> torch.Tensor:
        return getattr(self, f"W_{r}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        N, L_in, E_in = X.shape
        if L_in != self.L:
            raise ValueError(f"Expected lookback L={self.L}, got {L_in}")
        if E_in != self.E:
            raise ValueError(f"Expected E={self.E}, got {E_in}")

        pred = torch.zeros((N, self.E), dtype=X.dtype, device=X.device)

        for l in range(1, self.L + 1):
            x_lag = X[:, -l, :]                       # (N, E)

            # alpha part: scalar broadcast over all edges
            pred = pred + self.alpha[l - 1] * x_lag

            # beta part: same as the original
            for r in self.R_by_l[l - 1]:
                beta_lr = self.beta[self.beta_index[(l, r)]]
                pred = pred + beta_lr * (x_lag @ self.get_W(r).T)

        return pred

    @torch.no_grad()
    def export_alpha_beta(self) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
        alpha_np = self.alpha.detach().cpu().numpy().copy()          # (L,)
        beta_np = self.beta.detach().cpu().numpy().copy()
        beta_dict = {pair: float(beta_np[j]) for j, pair in enumerate(self.beta_pairs)}
        return alpha_np, beta_dict

    @torch.no_grad()
    def export_psi_list(self) -> List[np.ndarray]:
        """Psi_l = alpha_l * I_E + sum_r beta_{l,r} W^(r) (the global special case of eq. 6)."""
        alpha_np, beta_dict = self.export_alpha_beta()
        psi_list = []
        for l in range(1, self.L + 1):
            M = float(alpha_np[l - 1]) * np.eye(self.E)
            for r in self.R_by_l[l - 1]:
                M += beta_dict[(l, r)] * self.get_W(r).detach().cpu().numpy()
            psi_list.append(M)
        return psi_list


# -------------------------------------------------
# 2) learner: global alpha + global beta
# -------------------------------------------------

class GNAREdgeGlobalLearner(GNAREdgeLearner):
    """
    GNAR-edge learner with global alpha + global beta (the "global GNAR-edge model"
    used in the paper's experiments).

    The only difference from GNAREdgeLearner: alpha is no longer per-edge but a
    single network-wide scalar per lag. Parameter count = L + B (B = sum_l |R_l|).

    When to use it: with short samples / many edges (large E), the per-edge alpha
    has L*E parameters and overfits easily; the global version is its nested special
    case with lower variance and direct interpretation (use AIC/BIC to choose).
    """

    # ---------- model ----------
    def build_module(self) -> nn.Module:
        needed_r = sorted({r for rs in self.R_by_l for r in rs})
        W_dict = {r: self.graph.neighbor_matrix(r) for r in needed_r}

        return GNAREdgeGlobalModule(
            E=self.E,
            L=self.L,
            R_by_l=self.R_by_l,
            W_dict=W_dict,
        )

    # ---------- closed-form OLS ----------
    def _fit_impl_ols(self, train_set: EdgeTrainSet, **kwargs):
        """
        Closed-form OLS under global coefficients (pooled OLS).

        Key observation: shared parameters <=> merging design-matrix columns.
        After flattening (n, e) into N*E observations, the E sparse columns that
        belong to theta_{l,e} in the per-edge version get tied into a single
        theta_l in the global version -- i.e. those E columns are simply summed
        into one dense column x_l.flat. The design matrix therefore has only
        L+B columns:

            D = [ x_1.flat, ..., x_L.flat, z_(l,r).flat, ... ]   shape (N*E, L+B)
            theta_hat = pinv(D^T D) @ (D^T y.flat)

        No per-edge FWL partialling-out is needed any more; the normal equations
        are a small (L+B)x(L+B) system. The base features {x_l, z_(l,r)} come from
        exactly the same source as the parent / forward (reusing
        _build_ols_beta_features and the same W buffers), which guarantees the
        estimator and the predictor cannot drift apart.
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

        # Base feature columns for alpha: x_l[n,e] = X[n, -l, e], flattened to (N*E,)
        cols = [X[:, -l, :].reshape(-1) for l in range(1, self.L + 1)]

        # Base feature columns for beta: reuse the parent's builder (same W as forward)
        beta_pairs, Z_list = self._build_ols_beta_features(X)
        cols.extend(Z.reshape(-1) for Z in Z_list)

        D = np.stack(cols, axis=1)                 # (N*E, L+B)
        y_flat = y.reshape(-1)

        S = D.T @ D                                # (L+B, L+B)
        rhs = D.T @ y_flat
        theta = np.linalg.pinv(S, rcond=ols_rcond) @ rhs

        P = int(D.shape[1])
        self.ols_rank_ = int(np.linalg.matrix_rank(S))
        if verbose and self.ols_rank_ < P:
            print(
                f"[{self.__class__.__name__}] "
                f"warning: design rank deficient ({self.ols_rank_}/{P}); "
                f"coefficients may not be unique."
            )

        alpha_hat = theta[: self.L]                # (L,)
        beta_hat = theta[self.L:]                  # (B,)

        # Write back into the torch parameters; predict / export / AIC-BIC all work unchanged
        with torch.no_grad():
            self.model.alpha.copy_(
                torch.as_tensor(alpha_hat, dtype=self.model.alpha.dtype,
                                device=self.model.alpha.device)
            )
            self.model.beta.copy_(
                torch.as_tensor(beta_hat, dtype=self.model.beta.dtype,
                                device=self.model.beta.device)
            )

        # Record the training error (same convention as the parent)
        with torch.no_grad():
            y_pred = self.forward(X)
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        resid = y - np.asarray(y_pred, dtype=float)
        mse = float(np.mean(resid ** 2))

        self.ols_method_ = "pooled_global_ols"
        self.ols_rcond_ = float(ols_rcond)
        self.ols_beta_pairs_ = list(beta_pairs)
        self.ols_residuals_ = resid
        self.ols_rss_ = float(np.sum(resid ** 2))
        self.ols_mse_ = mse
        self.ols_nobs_ = int(N * self.E)
        self.ols_k_params_ = int(self._num_mean_params())   # = L + B

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

    # ---------- parameter export ----------
    def export_parameters_to_excel(
            self,
            filepath: str,
            sheet_name: str,
            *args,
            **kwargs,
    ) -> str:
        """
        Parameter export for the global version. The parent exports alpha as
        per-edge (L, E) columns; here alpha is (L,), so the shapes do not match.
        This compact version writes meta / alpha / beta (optionally Psi) into a
        single sheet.
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

        alpha_np, beta_dict = self.model.export_alpha_beta()

        rows: List[List[Any]] = []
        rows.append(["meta"])
        rows.append(["class", self.__class__.__name__])
        rows.append(["is_fitted", bool(self.is_fitted)])
        rows.append(["E", int(self.E)])
        rows.append(["L", int(self.L)])
        rows.append(["R_by_l", str(self.R_by_l)])
        rows.append([])

        rows.append(["alpha (global, one scalar per lag)"])
        rows.append(["lag", "alpha"])
        for l in range(1, self.L + 1):
            rows.append([f"lag_{l}", float(alpha_np[l - 1])])
        rows.append([])

        rows.append(["beta (global, one scalar per (lag, stage))"])
        rows.append(["lag", "stage", "beta"])
        for (l, r), v in sorted(beta_dict.items()):
            rows.append([int(l), int(r), float(v)])

        if include_psi:
            edge_labels = self._edge_labels()
            for i, psi in enumerate(self.model.export_psi_list(), start=1):
                rows.append([])
                rows.append([f"Psi_{i}"])
                rows.append(["edge"] + edge_labels)
                for lbl, rrow in zip(edge_labels, np.asarray(psi, dtype=float)):
                    rows.append([lbl] + [float(v) for v in rrow])

        max_cols = max(len(r) for r in rows)
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
            out_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        return filepath
