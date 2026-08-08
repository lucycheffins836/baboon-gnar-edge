"""
model.py

Defines GNAREdgeGlobalMultiCovModule and GNAREdgeGlobalMultiCovLearner, an extension
of the global GNAR-edge model (Mantziou et al., 2023) that additionally supports:
  - time-invariant edge-level covariates (e.g. age difference)
  - time-varying exogenous series common to all edges (e.g. temperature)
  - interaction terms between edge-level covariates and exogenous series

Estimation is performed via closed-form ordinary least squares, following the same
approach used by the reference GNAR-edge implementation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "external"))

from BaseEdgeGNAR_edge import GNAREdgeLearner
from BaseEdgeGNAR_edge_global import GNAREdgeGlobalLearner, GNAREdgeGlobalModule
from edge_graph import ArrayEdgeGraph
from RollingEdgePredict import RollingEdgePredict


class GNAREdgeGlobalMultiCovModule(GNAREdgeGlobalModule):
    """
    Extends the global GNAR-edge model with an arbitrary number of covariates:
      - "edge" covariates: fixed per-edge value (e.g. age_diff), constant over time.
      - "time" covariates: fixed per-timepoint value (e.g. temperature), constant across edges.
      - interaction_pairs: list of (edge_cov_name, time_cov_name) tuples specifying which
        edge-covariate x time-covariate interactions to include. Empty list = no interactions.

    Parameters
    ----------
    E : int
        Number of edges.
    L : int
        Number of autoregressive lags.
    R_by_l : list of list of int
        Neighbour stages included at each lag.
    W_dict : dict of {int: torch.Tensor}
        Mapping from neighbour stage r to the corresponding (E, E) neighbour-weight matrix.
    edge_cov_names : list of str
        Names of the time-invariant edge-level covariates to include.
    time_cov_names : list of str
        Names of the time-varying exogenous series to include.
    interaction_pairs : list of tuple of str, optional
        (edge_cov_name, time_cov_name) pairs specifying which interaction terms to include.
    dtype : torch.dtype, default torch.float32
        Data type used for model parameters and tensors.
    """
    def __init__(self, *, E, L, R_by_l, W_dict, edge_cov_names, time_cov_names,
                 interaction_pairs=None, dtype=torch.float32):
        super().__init__(E=E, L=L, R_by_l=R_by_l, W_dict=W_dict, dtype=dtype)
        self.edge_cov_names = list(edge_cov_names)
        self.time_cov_names = list(time_cov_names)
        self.interaction_pairs = list(interaction_pairs or [])

        #validate that every interaction pair references covariates that were actually supplied
        for edge_name, time_name in self.interaction_pairs:
            if edge_name not in self.edge_cov_names:
                raise ValueError(f"Interaction references unknown edge covariate '{edge_name}'.")
            if time_name not in self.time_cov_names:
                raise ValueError(f"Interaction references unknown time covariate '{time_name}'.")

        n_cov = len(self.edge_cov_names) + len(self.time_cov_names) + len(self.interaction_pairs)
        self.gamma = nn.Parameter(torch.zeros(n_cov, dtype=dtype))

    def forward(self, X: torch.Tensor, edge_covariates=None, time_covariates=None) -> torch.Tensor:
        """
        Computes the model's predicted edge weights.

        Parameters
        ----------
        X : torch.Tensor, shape (N, L, E)
            Lagged edge-weight history used to compute the autoregressive and
            network-effect components (via the parent class's forward method).
        edge_covariates : dict of {str: array-like}, optional
            Mapping from edge covariate name to its (E,) array of values.
        time_covariates : dict of {str: array-like}, optional
            Mapping from time covariate name to its (N,) array of values.

        Returns
        -------
        torch.Tensor, shape (N, E)
            Predicted edge weights.
        """
        pred = super().forward(X)   # (N, E)

        gamma_idx = 0
        #keep tensors around so interaction terms can reuse them without rebuilding
        edge_tensors = {}
        time_tensors = {}

        if edge_covariates is not None:
            for name in self.edge_cov_names:
                cov_tensor = torch.as_tensor(edge_covariates[name], dtype=pred.dtype, device=pred.device).view(1, -1)  # (1,E)
                edge_tensors[name] = cov_tensor
                pred = pred + self.gamma[gamma_idx] * cov_tensor
                gamma_idx += 1

        if time_covariates is not None:
            for name in self.time_cov_names:
                z_tensor = torch.as_tensor(time_covariates[name], dtype=pred.dtype, device=pred.device).view(-1, 1)  # (N,1)
                time_tensors[name] = z_tensor
                pred = pred + self.gamma[gamma_idx] * z_tensor
                gamma_idx += 1

        #any number of interaction terms, each edge_cov x time_cov, broadcasting to (N, E)
        for edge_name, time_name in self.interaction_pairs:
            interaction = time_tensors[time_name] * edge_tensors[edge_name]   # (N,1)*(1,E) -> (N,E)
            pred = pred + self.gamma[gamma_idx] * interaction
            gamma_idx += 1

        return pred


@dataclass
class GNAREdgeGlobalMultiCovLearner(GNAREdgeGlobalLearner):
    """
    Learner class for the extended global GNAR-edge model, supporting closed-form
    OLS estimation with edge-level covariates, time-varying exogenous series, and
    their interactions.

    Attributes
    ----------
    edge_covariates : dict of {str: np.ndarray}, optional
        Mapping from covariate name to its (E,) array of edge-level values.
    exog_series : dict of {str: pd.Series}, optional
        Mapping from series name to a pandas Series of exogenous values, indexed
        by time period.
    interaction_pairs : list of tuple of str, optional
        (edge_cov_name, time_cov_name) pairs specifying which interaction terms
        to include in the model.
    """
    edge_covariates: object = None      #dict: {name: np.array of shape (E,)}
    exog_series: object = None          #dict: {name: pd.Series indexed by time period}
    interaction_pairs: object = None    #list of (edge_cov_name, time_cov_name) tuples

    def __post_init__(self):
        self.edge_covariates = dict(self.edge_covariates or {})
        self.exog_series = dict(self.exog_series or {})
        self.interaction_pairs = list(self.interaction_pairs or [])
        self.edge_cov_names = list(self.edge_covariates.keys())
        self.time_cov_names = list(self.exog_series.keys())

        for edge_name, time_name in self.interaction_pairs:
            if edge_name not in self.edge_cov_names:
                raise ValueError(f"Interaction references unknown edge covariate '{edge_name}'.")
            if time_name not in self.time_cov_names:
                raise ValueError(f"Interaction references unknown time covariate '{time_name}'.")

        super().__post_init__()

    def build_module(self):
        """Constructs the underlying GNAREdgeGlobalMultiCovModule for this learner."""
        needed_r = sorted({r for rs in self.R_by_l for r in rs})
        W_dict = {r: self.graph.neighbor_matrix(r) for r in needed_r}
        return GNAREdgeGlobalMultiCovModule(
            E=self.E, L=self.L, R_by_l=self.R_by_l, W_dict=W_dict,
            edge_cov_names=self.edge_cov_names, time_cov_names=self.time_cov_names,
            interaction_pairs=self.interaction_pairs,
        )

    def _get_time_cov(self, name, periods):
        """
        Retrieves and validates a time-varying exogenous covariate's values for
        the given periods.

        Parameters
        ----------
        name : str
            Name of the exogenous series (must be a key in self.exog_series).
        periods : array-like
            Time periods for which to retrieve values.

        Returns
        -------
        np.ndarray
            The covariate's values at the requested periods.

        Raises
        ------
        ValueError
            If any requested period is missing from the exogenous series.
        """
        z = self.exog_series[name].reindex(periods)
        if z.isna().any():
            raise ValueError(f"Missing exogenous values for '{name}'.")
        return z.to_numpy(dtype=float)

    def _fit_impl_ols(self, train_set, **kwargs):
        """
        Fits the model via closed-form OLS, flattening the (time, edge) panel into
        a single linear regression with columns for lagged autoregressive terms,
        neighbour-weighted terms, edge covariates, time covariates, and interactions.

        Parameters
        ----------
        train_set : object
            Training data, exposing X (shape (N, L, E)), y (shape (N, E)), and
            sample_target_periods.
        verbose : bool, optional
            If True, prints a warning if the design matrix is rank deficient.
        ols_rcond : float, default 1e-10
            Cutoff for small singular values in the pseudoinverse solve.

        Returns
        -------
        float
            Mean squared error of the fitted model on the training data.
        """
        verbose = bool(kwargs.get("verbose", self.verbose_fit))
        ols_rcond = float(kwargs.get("ols_rcond", 1e-10))
        if ols_rcond < 0:
            raise ValueError("ols_rcond must be >= 0.")

        X = np.asarray(train_set.X, dtype=float)   # (N, L, E)
        y = np.asarray(train_set.y, dtype=float)   # (N, E)

        if X.ndim != 3 or y.ndim != 2:
            raise ValueError("Invalid matrix layout dimensions.")

        N, L_in, E_in = X.shape

        cols = [X[:, -l, :].reshape(-1) for l in range(1, self.L + 1)]

        beta_pairs, Z_list = self._build_ols_beta_features(X)
        cols.extend(Z.reshape(-1) for Z in Z_list)

        #keep flattened edge/time covariate arrays around for reuse in interaction terms
        edge_cov_flat = {}
        for name in self.edge_cov_names:
            cov = self.edge_covariates[name]
            if len(cov) != self.E:
                raise ValueError(f"edge_covariates['{name}'] length {len(cov)} != E {self.E}.")
            flat = np.tile(cov, N)          #length N*E
            edge_cov_flat[name] = flat
            cols.append(flat)

        time_cov_flat = {}
        for name in self.time_cov_names:
            z = self._get_time_cov(name, train_set.sample_target_periods)
            flat = np.repeat(z, self.E)     #length N*E
            time_cov_flat[name] = flat
            cols.append(flat)

        #general interaction terms: elementwise product of the two already-flattened arrays
        for edge_name, time_name in self.interaction_pairs:
            interaction_flat = edge_cov_flat[edge_name] * time_cov_flat[time_name]
            cols.append(interaction_flat)

        D = np.stack(cols, axis=1)
        y_flat = y.reshape(-1)

        #using OLS to solve for the estimated parameters, which are contained in theta
        S = D.T @ D
        rhs = D.T @ y_flat
        theta = np.linalg.pinv(S, rcond=ols_rcond) @ rhs

        P = int(D.shape[1])
        self.ols_rank_ = int(np.linalg.matrix_rank(S))
        if verbose and self.ols_rank_ < P:
            print(f"[{self.__class__.__name__}] warning: design rank deficient ({self.ols_rank_}/{P})")

        #storing estimated parameters
        alpha_hat = theta[: self.L]
        beta_hat = theta[self.L : self.L + len(beta_pairs)]
        gamma_hat = theta[self.L + len(beta_pairs):]

        #naming the exogenous terms
        gamma_names = (
            self.edge_cov_names
            + self.time_cov_names
            + [f"{e}_X_{t}" for e, t in self.interaction_pairs]
        )
        self.gamma_ = dict(zip(gamma_names, gamma_hat))

        with torch.no_grad():
            self.model.alpha.copy_(
                torch.as_tensor(alpha_hat, dtype=self.model.alpha.dtype, device=self.model.alpha.device)
            )
            self.model.beta.copy_(
                torch.as_tensor(beta_hat, dtype=self.model.beta.dtype, device=self.model.beta.device)
            )
            self.model.gamma.copy_(
                torch.as_tensor(gamma_hat, dtype=self.model.gamma.dtype, device=self.model.gamma.device)
            )
        #calculating mse
        with torch.no_grad():
            y_pred = self.forward(
                X,
                edge_covariates=self.edge_covariates,
                time_covariates={name: self._get_time_cov(name, train_set.sample_target_periods)
                                  for name in self.time_cov_names},
            )
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        resid = y - np.asarray(y_pred, dtype=float)
        mse = float(np.mean(resid ** 2))

        self.ols_method_ = "pooled_global_multicov_ols"
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
            epoch=1, loss=mse, epoch_seconds=0.0, abs_improve=np.nan, rel_improve=np.nan,
            best_loss_so_far=mse, no_improve_rounds=0, low_gain_rounds=0, verbose=verbose, print_every=1,
        )
        return mse

    def _num_mean_params(self):
        return int(self.alpha.size) + len(self.beta) + len(self.gamma_)

    @property
    def alpha(self):
        return self.model.export_alpha_beta()[0]

    @property
    def beta(self):
        return self.model.export_alpha_beta()[1]

    def forward(self, X, edge_covariates=None, time_covariates=None):
        X_tensor = torch.as_tensor(X, dtype=self.model.alpha.dtype, device=self.model.alpha.device)
        
        # fallback for edge covariates if not provided
        if edge_covariates is None:
            edge_covariates = self.edge_covariates
            
        # fallback for time covariates if not provided
        if time_covariates is None:
            # we assume the time series sequence length matches the target periods 
            # associated with the training sample frame size N
            N = X.shape[0]
            
            # Use the last N periods of train_periods to reconstruct historical alignments
            current_periods = self.train_periods[-N:] 
            time_covariates = {
                name: self._get_time_cov(name, current_periods)
                for name in self.time_cov_names
            }
            
        return self.model(X_tensor, edge_covariates=edge_covariates, time_covariates=time_covariates)
    def predict_next(self, periods=None, **kwargs):
        """
        Generates a one-step-ahead prediction using the fitted model.

        Parameters
        ----------
        periods : array-like, optional
            Time periods to use as history for the prediction; defaults to
            self.train_periods if not provided.

        Returns
        -------
        np.ndarray, shape (E,)
            Predicted edge weights for the period immediately following `periods`.
        """
        if periods is None:
            periods = self.train_periods
        X_in = self.build_history_input(periods)

        time_cov_next = {
            name: self._get_time_cov(name, [periods[-1]])
            for name in self.time_cov_names
        }

        self.model.eval()
        with torch.no_grad():
            out = self.forward(X_in, edge_covariates=self.edge_covariates, time_covariates=time_cov_next)

        return out.detach().cpu().numpy().reshape(-1)
    

def build_edge_covariate(edge_names, age_lookup, sex_lookup=None, kind="abs_diff"):
    """
    Construct an edge-level covariate from node-level demographic information.

    This function computes a covariate value for each edge in a network using
    the demographic attributes of the two incident nodes. Supported covariates
    include the absolute age difference, the sum of ages, the mean age, and an
    indicator of whether the two nodes have the same sex.

    Parameters
    ----------
    edge_names : list of str
        List of edge labels in the form ``"node1-node2"``, for example
        ``["ANGELE-FANA", "BOBO-HARLEM"]``.
    age_lookup : dict
        Dictionary mapping node names to ages.
    sex_lookup : dict, optional
        Dictionary mapping node names to sex labels. Required when
        ``kind="same_sex"``.
    kind : {"abs_diff", "sum", "mean", "same_sex"}, default="abs_diff"
        Edge covariate to construct:

        - ``"abs_diff"``: absolute difference in node ages.
        - ``"sum"``: sum of node ages.
        - ``"mean"``: mean of node ages.
        - ``"same_sex"``: indicator taking value 1.0 if the two nodes have the
        same sex and 0.0 otherwise.

    Returns
    -------
    numpy.ndarray
        One-dimensional array containing the covariate value for each edge in the
        same order as ``edge_names``.

    Raises
    ------
    ValueError
        If ``kind="same_sex"`` and ``sex_lookup`` is not provided, or if
        ``kind`` is not one of the supported covariate types.
    """
    covariate = []

    for edge in edge_names:
        n1, n2 = edge.split("-")

        if kind == "same_sex":
            if sex_lookup is None:
                raise ValueError("sex_lookup must be provided when kind='same_sex'")
            covariate.append(1.0 if sex_lookup[n1] == sex_lookup[n2] else 0.0)
            continue

        a1 = age_lookup[n1]
        a2 = age_lookup[n2]

        if kind == "abs_diff":
            covariate.append(abs(a1 - a2))
        elif kind == "sum":
            covariate.append(a1 + a2)
        elif kind == "mean":
            covariate.append((a1 + a2) / 2.0)
        else:
            raise ValueError("kind must be one of: 'abs_diff', 'sum', 'mean', 'same_sex'")

    return np.array(covariate, dtype=float)