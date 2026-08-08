#loading packages
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.multitest import multipletests
from collections import Counter
from scipy.stats import pearsonr
import sys
import os
import time
from pathlib import Path   

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "external"))

from BaseEdgeGNAR_edge import GNAREdgeLearner
from BaseEdgeGNAR_edge_global import GNAREdgeGlobalLearner
from edge_baselines import ARIMAEdgeBaseline, NaiveEdgeBaseline, ZeroEdgeBaseline, compare_baselines
from edge_graph import ArrayEdgeGraph
from RollingEdgePredict import RollingEdgePredict
from model import build_edge_covariate, GNAREdgeGlobalMultiCovLearner
from simulation import (
    # Network generation
    calibrate_rdp_radius,
    get_or_calibrate_rdp_radius,
    generate_network,
    simulate_node_covariates,
    build_synthetic_edge_covariates,

    # Time-series simulation
    simulate_ar1_exog,
    simulate_full_model,

    # Graph and design-matrix utilities
    build_W_matrices,
    build_design_matrix,
    compute_ols_standard_errors,

    # Monte Carlo simulation
    run_full_simulation_replication,
    run_large_network_replication_density,
    run_single_step_prediction_replication,
    run_single_step_prediction_replication_learner,

    # Regime construction
    expand_R_by_l,
    build_beta_dict,
    build_regime,

    # Results tables
    build_per_parameter_table,
    build_aggregated_table4_style,

    # Small helper
    format_tuple_list,
    format_as_multiindex
)

def check_gnar_stationarity_eigenvalues(learner):
    """
    Check whether a fitted GNAR-edge model satisfies the stationarity condition.

    This function constructs the companion matrix corresponding to the fitted
    GNAR-edge model using the estimated autoregressive coefficient matrices stored
    in ``learner.Psi_list``. Stationarity is assessed by computing the eigenvalues
    of the companion matrix and verifying that all eigenvalues lie strictly inside
    the unit circle.

    Parameters
    ----------
    learner : GNAREdgeGlobalMultiCovLearner or compatible learner
        Fitted GNAR-edge learner exposing a ``Psi_list`` attribute, where each
        element is the estimated coefficient matrix for one autoregressive lag.

    Returns
    -------
    tuple
        A pair ``(max_modulus, is_stationary)`` where

        - ``max_modulus`` is the largest modulus of the companion matrix
        eigenvalues.
        - ``is_stationary`` is ``True`` if all eigenvalues have modulus strictly
        less than one and ``False`` otherwise.

    Notes
    -----
    The stationarity criterion is the standard condition for vector
    autoregressive (VAR) models. A fitted GNAR-edge model is considered
    stationary when the spectral radius of its companion matrix is less than one.
    """
    Psi_list = learner.Psi_list
    L = len(Psi_list)
    K = Psi_list[0].shape[0]

    top_row = np.hstack(Psi_list)
    if L > 1:
        identity_block = np.eye(K * (L - 1))
        bottom = np.hstack([identity_block, np.zeros((K * (L - 1), K))])
        companion = np.vstack([top_row, bottom])
    else:
        companion = top_row

    eigenvalues = np.linalg.eigvals(companion)
    max_modulus = np.max(np.abs(eigenvalues))
    is_stationary = max_modulus < 1

    return max_modulus, is_stationary




PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "output"

FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

def save_figure(fig, filename, dpi=300):
    """Save a figure as both PDF and PNG in the project's figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{filename}.pdf",
                dpi=dpi,
                bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{filename}.png",
                dpi=dpi,
                bbox_inches="tight")


def save_table(df, filename):
    """Save a DataFrame as a CSV file in the project's tables directory."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(TABLES_DIR / f"{filename}.csv", index=False)


def run_residual_diagnostics_pooled(
    learner,
    train_set,
    label,
    exog_series,
    edges_to_plot=None,
):
    """
    Run pooled residual diagnostics for a fitted GNAR-edge model.

    This function conducts residual analysis on a fitted GNAR-edge model
    using pooled residuals from all edge-time observations. It computes summary diagnostic statistics, assesses model
    stationarity, produces diagnostic plots, identifies residual outliers, and
    optionally examines the relationship between residuals and an exogenous
    time-varying covariate.

    All figures are saved to the project figures directory
    and all tables are saved to the project tables directory
    with filenames prefixed by `label`.

    The following diagnostics are produced:

    - Summary statistics of pooled residuals.
    - Shapiro-Wilk test for residual normality.
    - Stationarity assessment based on the eigenvalues of the GNAR companion matrix.
    - Boxplots of residuals across edges at each time point.
    - Observed-versus-fitted plots for selected edges.
    - Residual time series for selected edges.
    - Histogram and Q-Q plot of pooled residuals.
    - Autocorrelation function (ACF) of the mean residual series.
    - Edge-wise correlations between residuals and an exogenous covariate (if supplied).

    Parameters
    ----------
    learner : GNAREdgeGlobalMultiCovLearner or compatible learner
        Fitted GNAR-edge learner.
    train_set : object
        Training dataset used to fit the learner. Must expose the response matrix
        ``y`` and the target periods ``sample_target_periods``.
    label : str
        Label used in diagnostic tables, plot titles, and saved filenames.
    exog_series : pandas.Series or None
        Time-varying exogenous covariate aligned to the training period. If
        provided, edge-wise Pearson correlations between residuals and the
        exogenous series are computed.
    edges_to_plot : list of str, optional
        List of edge labels to visualize individually. Labels should match the
        format ``"node1->node2"``. Any labels not found in the fitted graph are
        reported and ignored.

    Returns
    -------
    tuple
        A four-element tuple containing

        - ``diagnostics_table`` : pandas.DataFrame summarizing pooled residual
        statistics and the Shapiro-Wilk normality test.
        - ``stationarity_info`` : dict containing

        - ``"max_eigenvalue_modulus"``
        - ``"stationary"``

        - ``edge_corr_df`` : pandas.DataFrame containing edge-wise Pearson
        correlations with the exogenous covariate, or ``None`` if no exogenous
        series is supplied.
        - ``edge_corr_summary`` : pandas.DataFrame summarizing the distribution of
        edge-wise residual correlations, or ``None`` if no exogenous series is
        supplied.

    """
    safe_label = label.replace(" ", "_").replace("/", "_").lower()

    y_true = train_set.y
    y_pred = learner.predict_train_set(train_set)
    resid = learner.conditional_residuals(train_set)   #shape (n_periods, n_edges)
    periods = train_set.sample_target_periods

    #edge lookup for optional plots
    edge_labels = [
        f"{learner.graph.node_labels[u]}->{learner.graph.node_labels[v]}"
        for u, v in learner.edges
    ]

    edge_lookup = {name: i for i, name in enumerate(edge_labels)}

    edges_to_plot = edges_to_plot or []

    edge_indices = []
    missing = []

    for edge in edges_to_plot:
        if edge in edge_lookup:
            edge_indices.append(edge_lookup[edge])
        else:
            missing.append(edge)

    if missing:
        print("Edges not found:", missing)

    #pool all residuals (every edge, every time point) into one flat array
    all_residuals_flat = resid.flatten()

    #compute the shapiro wilk statistic
    try:
        shapiro_stat, shapiro_p = stats.shapiro(all_residuals_flat)
    except Exception:
        shapiro_stat, shapiro_p = np.nan, np.nan

    #saving a diagnostics table
    diagnostics_table = pd.DataFrame([{
        "label": label,
        "mean_residual": float(np.mean(all_residuals_flat)),
        "sd_residual": float(np.std(all_residuals_flat)),
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "n_periods": resid.shape[0],
        "n_edges": resid.shape[1],
        "n_observations": all_residuals_flat.size,
    }])
    save_table(diagnostics_table, f"{safe_label}_diagnostics_table")

    #computing stationarity
    max_mod, is_stat = check_gnar_stationarity_eigenvalues(learner)

    #boxplot of residuals across edges, per time stamp
    fig = plt.figure(figsize=(14, 6))
    plt.boxplot(resid.T, showfliers=True, widths=0.5)
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"{label}: residual distribution across edges, per time stamp", fontsize=14, fontweight="bold")
    save_figure(fig, f"{safe_label}_residual_boxplot")
    plt.show()

    #individual edge fitted vs observed
    for edge_idx in edge_indices:

        fig = plt.figure(figsize=(12, 4))

        plt.plot(
            periods,
            y_true[:, edge_idx],
            linewidth=2,
            label="Observed",
        )

        plt.plot(
            periods,
            y_pred[:, edge_idx],
            linewidth=2,
            label="Fitted",
        )

        plt.title(f"{label}: {edge_labels[edge_idx]}", fontsize=14, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Response", fontsize=12)
        plt.legend()
        plt.tight_layout()
        save_figure(fig, f"{safe_label}_fitted_vs_observed_{edge_labels[edge_idx].replace('->', '_')}")
        plt.show()

    #individual edge residuals
    for edge_idx in edge_indices:

        fig = plt.figure(figsize=(12, 4))

        plt.plot(
            periods,
            resid[:, edge_idx],
            linewidth=2,
        )

        plt.axhline(
            0,
            color="black",
            linestyle="--",
        )

        plt.title(f"{label}: residuals ({edge_labels[edge_idx]})", fontsize=14, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Residual", fontsize=12)
        plt.tight_layout()
        save_figure(fig, f"{safe_label}_residuals_{edge_labels[edge_idx].replace('->', '_')}")
        plt.show()

    #count actual outliers using 2 standard deviations per time step
    outlier_count = 0
    total_count = 0
    for t in range(resid.shape[0]):
        row = resid[t, :]
        mean = np.mean(row)
        std = np.std(row)

        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std

        outliers_this_step = np.sum((row < lower_bound) | (row > upper_bound))
        outlier_count += outliers_this_step
        total_count += len(row)

    print(f"Outliers (2 std dev): {outlier_count} / {total_count} ({outlier_count/total_count*100:.1f}%)")

    outlier_summary_df = pd.DataFrame([{
        "label": label,
        "outlier_count": outlier_count,
        "total_count": total_count,
        "outlier_pct": outlier_count / total_count * 100,
    }])
    save_table(outlier_summary_df, f"{safe_label}_outlier_summary")

    #histogram of pooled residuals
    fig = plt.figure(figsize=(7, 5))
    plt.hist(all_residuals_flat, bins=30, edgecolor="black", density=True)
    plt.axvline(0, color="red", linestyle="--")
    plt.title(f"{label}: histogram of pooled residuals (all edges x time)", fontsize=14, fontweight="bold")
    save_figure(fig, f"{safe_label}_pooled_residual_histogram")
    plt.show()

    #QQ plot of pooled residuals
    fig = plt.figure(figsize=(6, 6))
    stats.probplot(all_residuals_flat, dist="norm", plot=plt)
    plt.title(f"{label}: QQ plot of pooled residuals", fontsize=14, fontweight="bold")
    save_figure(fig, f"{safe_label}_pooled_residual_qq")
    plt.show()

    #ACF computed on the mean residual series
    mean_resid = resid.mean(axis=1)
    fig = plt.figure(figsize=(8, 4))
    plot_acf(mean_resid, lags=min(10, len(mean_resid)-1))
    plt.title(f"{label}: ACF of mean residuals", fontsize=14, fontweight="bold")
    save_figure(fig, f"{safe_label}_mean_residual_acf")
    plt.show()

    #edge wise residual - exogenous correlations
    edge_corr_df = None
    edge_corr_summary = None

    if exog_series is not None:

        #align exogenous series to training periods
        aligned_exog = (
            exog_series
            .reindex(periods)
            .to_numpy(dtype=float)
        )

        edge_corrs = []

        for e in range(resid.shape[1]):

            valid = (
                np.isfinite(aligned_exog)
                & np.isfinite(resid[:, e])
            )

            if valid.sum() < 3:
                r = np.nan
                p = np.nan
            #calculate correlation between each edge residual and the exog series
            else:
                r, p = pearsonr(
                    aligned_exog[valid],
                    resid[:, e]
                )

            edge_corrs.append({
                "edge": e,
                "correlation": r,
                "p_value": p,
            })

        edge_corr_df = pd.DataFrame(edge_corrs)
        save_table(edge_corr_df, f"{safe_label}_edge_corr_df")
        #summarising distribution of correlations with exog series across edges
        edge_corr_summary = (
            edge_corr_df["correlation"]
            .describe()
            .to_frame("correlation")
        )
        save_table(edge_corr_summary.reset_index(), f"{safe_label}_edge_corr_summary")

        print("\nEdge-wise residual/exogenous correlations")
        print(edge_corr_summary)

        fig = plt.figure(figsize=(7, 4))
        plt.hist(
            edge_corr_df["correlation"].dropna(),
            bins=20,
            edgecolor="black"
        )
        plt.xlabel("Pearson correlation", fontsize=12)
        plt.ylabel("Number of edges", fontsize=12)
        plt.title(f"{label}: edge-wise residual correlations", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_figure(fig, f"{safe_label}_edge_corr_histogram")
        plt.show()

    return (
        diagnostics_table,
        {
            "max_eigenvalue_modulus": max_mod,
            "stationary": is_stat,
        },
        edge_corr_df,
        edge_corr_summary,
    )
def time_call(experiment, fn, timing_results, **kwargs):
    """
    Measure the execution time of a function call.

    This helper executes a specified function, records its execution time,
    appends the runtime to a user-supplied list, and prints the elapsed time.
    The function's return value is passed through unchanged.

    Parameters
    ----------
    experiment : str
        Descriptive name of the experiment or function being timed.
    fn : callable
        Function to execute.
    timing_results : list
        List to which a dictionary containing the experiment name and runtime
        will be appended.
    **kwargs
        Keyword arguments passed directly to ``fn``.

    Returns
    -------
    object
        The return value of ``fn(**kwargs)``.

    Notes
    -----
    Execution time is measured in seconds using ``time.perf_counter()``, which
    provides a high-resolution timer suitable for benchmarking. Runtime
    information is appended to ``timing_results`` as a dictionary with the
    keys ``"experiment"`` and ``"runtime_seconds"``.
    """
    t0 = time.perf_counter()
    result = fn(**kwargs)
    runtime = time.perf_counter() - t0

    timing_results.append({
        "experiment": experiment,
        "runtime_seconds": runtime,
    })

    print(f"{experiment}: {runtime:.2f}s")

    return result