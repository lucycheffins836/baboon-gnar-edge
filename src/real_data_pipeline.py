#loading packages
import numpy as np
import pandas as pd
import sys  
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "external"))
DATA_DIR = PROJECT_ROOT / "data"

from BaseEdgeGNAR_edge import GNAREdgeLearner
from BaseEdgeGNAR_edge_global import GNAREdgeGlobalLearner
from edge_baselines import ARIMAEdgeBaseline, NaiveEdgeBaseline, ZeroEdgeBaseline, compare_baselines
from edge_graph import ArrayEdgeGraph
from RollingEdgePredict import RollingEdgePredict
from model import build_edge_covariate, GNAREdgeGlobalMultiCovLearner





def edge_weights_distribution(
    timestamp="1D",
    L=1,
    stages_per_lag=None,
    learner_type="global",
    transform="none",
    growth_epsilon=1.0,
    standardize=False,
    edge_threshold=1.0,
    n_rolling_test=5,
    custom_edge_pairs=None,
    verbose=True,
    weight_mode="count",
    path=None,
):
    """
    Loads interaction data, constructs a fixed-graph multivariate edge-weight
    time series, fits a GNAR-edge model via rolling-origin evaluation, and returns
    fitted model objects alongside predictive performance metrics.

    Parameters
    ----------
    timestamp : str, default "1D"
        Pandas frequency string used to floor raw interaction timestamps into
        discrete time windows (e.g. "1D" for daily windows, "12h" for 12-hourly).
    L : int, default 1
        Number of autoregressive lags included in the GNAR-edge model.
    stages_per_lag : list of list of int, optional
        Neighbour stages to include at each lag; the l-th entry gives the list of
        neighbour stages used for lag l (e.g. [[1], [1, 2]] for L=2, using 1-stage
        neighbours at lag 1 and 1- and 2-stage neighbours at lag 2). Defaults to
        [[]] (no neighbour information at any lag) if None.
    learner_type : {"global", "local"}, default "global"
        Whether to fit the global GNAR-edge model (autoregressive and neighbour
        coefficients shared across all edges) or the local model (edge-specific
        autoregressive coefficients).
    transform : {"none", "difference", "growth_rate"}, default "none"
        Transformation applied to the edge-weight level series before fitting:
        "none" fits on raw (normalised) levels, "difference" fits on first
        differences, and "growth_rate" fits on period-on-period growth rates.
    growth_epsilon : float, default 1.0
        Small constant added to the denominator when computing growth rates
        (transform="growth_rate"), to avoid division by zero.
    standardize : bool, default False
        If True, each edge's series (after any transform) is divided by its own
        training-period standard deviation before fitting, so that all edges are
        estimated on a comparable scale; if False, no per-edge standardisation
        is applied.
    edge_threshold : float, default 1.0
        Minimum proportion of the initial training windows in which a baboon pair
        must interact for an edge to be drawn between them (e.g. 1.0 requires
        interaction in every training window; 0.9 requires interaction in at
        least 90% of training windows). Ignored if custom_edge_pairs is provided.
    n_rolling_test : int, default 5
        Number of rolling-origin evaluation steps used to assess predictive
        performance; the model is trained on an expanding window and used to
        predict one step ahead at each step.
    custom_edge_pairs : list of tuple, optional
        If provided, overrides the persistence-based edge selection criterion and
        instead draws edges between exactly this custom set of baboon pairs.
    verbose : bool, default True
        If True, prints intermediate summaries (rolling-step results and overall
        model summary) during execution.
    weight_mode : {"count", "proportion_all", "proportion_edges"}, default "count"
        How raw interaction counts are normalised before fitting: "count" uses
        raw counts unmodified; "proportion_all" divides each pair's count by the
        total interactions among all baboon pairs that window; "proportion_edges"
        divides each pair's count by the total interactions among only the
        persistent (edge-drawn) pairs that window.
    path : str
        Path to the raw interaction data file (tab-separated, with columns
        including "i", "j", and "DateTime"). 

    Returns
    -------
    dict
        Dictionary containing the persistent edge list, node and window
        information, edge-weight panels at each stage of processing (raw, level,
        and model-fitting scale), the fitted graph and roller objects, rolling-
        window prediction results, per-edge error summaries, and an overall
        summary of model configuration and performance.
    """

    if stages_per_lag is None:
        stages_per_lag = [[]]

    if path is None:
        path = DATA_DIR / "baboons_proximity_data.txt"
    
    if weight_mode not in ("count", "proportion_all", "proportion_edges"):
        raise ValueError(
            "weight_mode must be one of 'count', 'proportion_all', 'proportion_edges'."
        )

    # helper: summary of a T x E edge-weight panel
    def network_summary(edge_weight_df):
        return pd.DataFrame({
            "DATE": edge_weight_df.index,
            "total_edge_weight": edge_weight_df.sum(axis=1).values,
            "mean_edge_weight": edge_weight_df.mean(axis=1).values,
            "sd_edge_weight": edge_weight_df.std(axis=1).values,
            "median_edge_weight": edge_weight_df.median(axis=1).values,
            "max_edge_weight": edge_weight_df.max(axis=1).values,
            "n_active_edges": (edge_weight_df > 0).sum(axis=1).values,
        })


    # read data
    df = pd.read_csv(path, sep="\t")

    # define nodes, set of potential edges, and windows
    df[["node1", "node2"]] = pd.DataFrame(
        df.apply(lambda row: sorted([row["i"], row["j"]]), axis=1).tolist(),
        index=df.index
    )

     # aggregate windows according to chosen window length
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%d/%m/%Y %H:%M")
    df["window"] = df["DateTime"].dt.floor(timestamp)

    #weights initially set equal to interaction counts
    edges = df.groupby(["window", "node1", "node2"]).size().reset_index(name="weight") 
    windows = sorted(edges["window"].unique())
    nodes = sorted(set(df["i"]).union(df["j"]))

    if n_rolling_test >= len(windows) - L - 1:
        raise ValueError("n_rolling_test is too large relative to the number of available windows.")

    #define which edges are drawn in the fixed graph
    if custom_edge_pairs is not None:
        persistent_pair_list = sorted(custom_edge_pairs)
    else:
        # define edges based on usable training periods for first rolling step
        initial_train_windows = windows[:-n_rolling_test] 
        edges_initial = edges[edges["window"].isin(initial_train_windows)].copy()
        edges_count_initial = (
            edges_initial.groupby(["node1", "node2"]).size().reset_index(name="n_windows")
        )
        min_windows_required = int(np.ceil(edge_threshold * len(initial_train_windows)))
        persistent_edges = edges_count_initial[
            edges_count_initial["n_windows"] >= min_windows_required
        ].copy()
        #list of pairs with edges drawn between them
        persistent_pair_list = list(
            persistent_edges[["node1", "node2"]].itertuples(index=False, name=None)
        )

    #raw count lookup table
    weight_lookup = {
        (row["window"], row["node1"], row["node2"]): row["weight"]
        for _, row in edges.iterrows()
    }

    #node identifiers
    node_to_idx = {name: i for i, name in enumerate(nodes)}
    idx_to_node = {i: name for name, i in node_to_idx.items()}
    K = len(nodes)

    #edge identifiers
    edge_idx_list = [(node_to_idx[a], node_to_idx[b]) for a, b in persistent_pair_list]
    edge_names = [f"{idx_to_node[a]}-{idx_to_node[b]}" for a, b in edge_idx_list]
    E = len(edge_idx_list)


    # raw panel: rows = windows, cols = edges
    X_level_raw = np.zeros((len(windows), E), dtype=float)
    for t, w in enumerate(windows):
        for e, (a, b) in enumerate(persistent_pair_list):
            X_level_raw[t, e] = weight_lookup.get((w, a, b), 0.0)

    daily_total_all = (
        edges.groupby("window")["weight"].sum().reindex(windows, fill_value=0.0)
    )


    # normalise by selected weight_mode
    if weight_mode == "count":
        X_level = X_level_raw.copy()

    elif weight_mode == "proportion_all":
        denom = daily_total_all.to_numpy(dtype=float)
        if np.any(denom == 0.0):
            raise ValueError(
                "At least one window has zero total interactions across all pairs; "
                "cannot compute proportion_all weights for that window."
            )
        X_level = X_level_raw / denom[:, None] #normalise by aggregate interactions between ALL pairs (not just those with edges between them)

    elif weight_mode == "proportion_edges":
        denom = X_level_raw.sum(axis=1)
        if np.any(denom == 0.0):
            raise ValueError(
                "At least one window has zero total interactions among persistent pairs; "
                "cannot compute proportion_edges weights for that window."
            )
        X_level = X_level_raw / denom[:, None] #normalise by aggregate interactions between pairs with edges between them

    # transform for model fitting
    if transform == "difference":
        X_model = np.diff(X_level, axis=0)
        model_windows = windows[1:]
    elif transform == "growth_rate":
        numerator = X_level[1:] - X_level[:-1]
        denominator = X_level[:-1] + growth_epsilon
        X_model = numerator / denominator
        model_windows = windows[1:]
    else:
        X_model = X_level
        model_windows = windows


    # standardisation
    n_train_for_std = len(model_windows) - n_rolling_test
    edge_std = X_model[:n_train_for_std, :].std(axis=0)
    edge_std[edge_std == 0] = 1.0

    if standardize:
        X_model_fit = X_model / edge_std
    else:
        X_model_fit = X_model
        edge_std = np.ones(E, dtype=float)

    # building the graph
    graph = ArrayEdgeGraph.from_edge_panel(
        X_model_fit,
        edge_idx_list,
        n_nodes=K,
        time_labels=model_windows,
        node_labels=nodes
    )

    # building the learners for each step of rolling edge prediction
    if learner_type == "global":
        learner_cls = GNAREdgeGlobalLearner
    elif learner_type == "local":
        learner_cls = GNAREdgeLearner
    else:
        raise ValueError("learner_type must be 'global' or 'local'")

    train_periods = model_windows[:-n_rolling_test]
    val_periods = model_windows[-n_rolling_test:]

    roller = RollingEdgePredict(
        graph=graph,
        learner_cls=learner_cls,
        train_periods=train_periods,
        val_periods=val_periods,
        learner_kwargs=dict(L=L, stages_per_lag=stages_per_lag, use_ols=True),
        fit_kwargs=dict(use_ols=True),
    ).run()

    summary_df = roller.summary()

    # evaluation in raw count scale (ie converted back after any differencing or normalisation)
    rolling_rows = []
    per_edge_errors = []
    window_to_idx = {w: i for i, w in enumerate(windows)}

    for s in roller.steps:
        target_period = s.target_period
        pred_value_fit = s.pred_value
        true_value_fit = s.true_value

        if true_value_fit is None:
            continue

        pred_value = pred_value_fit * edge_std
        true_value = true_value_fit * edge_std

        test_pos_level = window_to_idx[target_period]

        # converting to level scale if differenced
        if transform == "difference":
            last_level = X_level[test_pos_level - 1, :]
            y_pred_level = last_level + pred_value
        elif transform == "growth_rate":
            last_level = X_level[test_pos_level - 1, :]
            y_pred_level = pred_value * (last_level + growth_epsilon) + last_level
        else:
            y_pred_level = pred_value

        # converting to raw count scale if normalised
        if weight_mode == "count":
            y_pred_eval = y_pred_level
            y_true_eval = X_level_raw[test_pos_level, :]

        elif weight_mode == "proportion_all":
            total_all = float(daily_total_all.iloc[test_pos_level])
            y_pred_eval = y_pred_level * total_all
            y_true_eval = X_level_raw[test_pos_level, :]

        elif weight_mode == "proportion_edges":
            total_edges = float(X_level_raw[test_pos_level, :].sum())
            y_pred_eval = y_pred_level * total_edges
            y_true_eval = X_level_raw[test_pos_level, :]

        # computing errors
        abs_error = np.abs(y_true_eval - y_pred_eval)
        per_edge_errors.append(abs_error)

        #summary for each step
        rolling_rows.append({
            "step": s.step,
            "target_period": target_period,
            "MAE": abs_error.mean(),
            "RMSE": np.sqrt((abs_error ** 2).mean()),
            "n_train_samples": s.n_train_samples,
            "AIC": None if s.aic_bic is None else s.aic_bic.get("AIC"),
            "BIC": None if s.aic_bic is None else s.aic_bic.get("BIC"),
        })

    rolling_results = pd.DataFrame(rolling_rows)

    # errors for each edge
    per_edge_errors = np.array(per_edge_errors)
    per_edge_df = pd.DataFrame({
        "edge": edge_names,
        "edge_mae": per_edge_errors.mean(axis=0),
        "edge_rmse": np.sqrt((per_edge_errors ** 2).mean(axis=0)),
        "training_std": edge_std
    }).sort_values("edge_mae", ascending=False).reset_index(drop=True)

    # data frames of edge weights
    edge_weight_raw_df = pd.DataFrame(
        X_level_raw,
        index=pd.Index(windows, name="DATE"),
        columns=edge_names
    )

    edge_weight_level_df = pd.DataFrame(
        X_level,
        index=pd.Index(windows, name="DATE"),
        columns=edge_names
    )

    edge_weight_model_df = pd.DataFrame(
        X_model,
        index=pd.Index(model_windows, name="DATE"),
        columns=edge_names
    )

    summary = {
        "model": f"GNAR-edge ({learner_type}, L={L})",
        "timestamp": timestamp,
        "L": L,
        "transform": transform,
        "standardize": standardize,
        "weight_mode": weight_mode,
        "evaluation_scale": "raw_counts",
        "learner_type": learner_type,
        "edge_threshold": edge_threshold if custom_edge_pairs is None else "custom_leiden",
        "n_edges": E,
        "n_rolling_test": n_rolling_test,
        "mean_MAE": rolling_results["MAE"].mean(),
        "mean_RMSE": rolling_results["RMSE"].mean(),
        "std_MAE": rolling_results["MAE"].std(),
        "std_RMSE": rolling_results["RMSE"].std()
    }

    if verbose:
        print(summary_df)
        print(rolling_results)
        print(summary)

    #defining a function to return a summary of the distribution of edge weights
    def network_summary(edge_weight_df):
        return pd.DataFrame({
            "DATE": edge_weight_df.index,
            "total_edge_weight": edge_weight_df.sum(axis=1).values,
            "mean_edge_weight": edge_weight_df.mean(axis=1).values,
            "sd_edge_weight": edge_weight_df.std(axis=1).values,
            "median_edge_weight": edge_weight_df.median(axis=1).values,
            "max_edge_weight": edge_weight_df.max(axis=1).values,
            "n_active_edges": (edge_weight_df > 0).sum(axis=1).values,
        })

    network_summary_level = network_summary(edge_weight_level_df)

    return {
        "persistent_pair_list": persistent_pair_list,
        "edge_names": edge_names,
        "nodes": nodes,
        "windows": windows,
        "model_windows": model_windows,
        "X_level_raw": X_level_raw,
        "X_level": X_level,
        "X_model": X_model,
        "edge_weight_raw_df": edge_weight_raw_df,
        "edge_weight_level_df": edge_weight_level_df,
        "edge_weight_model_df": edge_weight_model_df,
        "network_summary_level": network_summary_level,
        "graph": graph,
        "roller": roller,
        "summary_df": summary_df,
        "rolling_results": rolling_results,
        "per_edge_df": per_edge_df,
        "summary": summary,
    }


def build_baboon_edge_graph(edge_threshold=1.0, timestamp="1D", n_rolling_test=5, path=None):
    """
    Builds an ArrayEdgeGraph on raw-count scale, using the persistence criterion
    applied to the first (T - n_rolling_test) training windows only, consistent
    with the edge-selection convention used throughout this project.
    """
    if path is None:
        path = DATA_DIR / "baboons_proximity_data.txt"

    #defining nodes, edges and aggregating into discrete time windows
    df = pd.read_csv(path, sep="\t")
    df[["node1", "node2"]] = pd.DataFrame(
        df.apply(lambda row: sorted([row["i"], row["j"]]), axis=1).tolist(), index=df.index
    )
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%d/%m/%Y %H:%M")
    df["window"] = df["DateTime"].dt.floor(timestamp)
    edges = df.groupby(["window", "node1", "node2"]).size().reset_index(name="weight")
    windows = sorted(edges["window"].unique())
    nodes = sorted(set(df["i"]).union(df["j"]))

    #defining edge set using persistence criterion
    initial_train_windows = windows[:-n_rolling_test]
    edges_initial = edges[edges["window"].isin(initial_train_windows)].copy()
    edges_count_initial = edges_initial.groupby(["node1", "node2"]).size().reset_index(name="n_windows")
    min_windows_required = int(np.ceil(edge_threshold * len(initial_train_windows)))
    persistent_edges = edges_count_initial[edges_count_initial["n_windows"] >= min_windows_required].copy()
    persistent_pair_list = list(persistent_edges[["node1", "node2"]].itertuples(index=False, name=None))

    node_to_idx = {name: i for i, name in enumerate(nodes)}
    K = len(nodes)
    edge_idx_list = [(node_to_idx[a], node_to_idx[b]) for a, b in persistent_pair_list]
    E = len(edge_idx_list)

    #raw interaction count lookup
    weight_lookup = {(r["window"], r["node1"], r["node2"]): r["weight"] for _, r in edges.iterrows()}
    X_level_raw = np.zeros((len(windows), E), dtype=float)
    for t, w in enumerate(windows):
        for e, (a, b) in enumerate(persistent_pair_list):
            X_level_raw[t, e] = weight_lookup.get((w, a, b), 0.0)

    graph = ArrayEdgeGraph.from_edge_panel(
        X_level_raw, edge_idx_list, n_nodes=K,
        time_labels=windows, node_labels=nodes,
    )
    return graph, windows, persistent_pair_list, X_level_raw


def run_baseline_comparison(graph, windows, n_rolling_test, gnar_L, gnar_stages_no_neighbours,
                             gnar_stages_with_neighbours, arima_orders=("1,0,0", "auto")):
    """
    Runs naive, zero, ARIMA (fixed + auto order), and two GNAR-edge (no covariates)
    specifications through the compare_baselines rolling harness.
    """
    val_frac_equivalent = n_rolling_test / len(windows)

    models = {
        "naive": (NaiveEdgeBaseline, {}),
        "zero": (ZeroEdgeBaseline, {}),
        "arima(1,0,0)": (ARIMAEdgeBaseline, dict(order=(1, 0, 0))),
        "arima(auto)": (ARIMAEdgeBaseline, dict(order="auto", auto_ic="aic")),
        "gnar_no_neighbours": (GNAREdgeGlobalLearner, dict(L=gnar_L, stages_per_lag=gnar_stages_no_neighbours,
                                                            use_ols=True, verbose_fit=False)),
        "gnar_with_neighbours": (GNAREdgeGlobalLearner, dict(L=gnar_L, stages_per_lag=gnar_stages_with_neighbours,
                                                              use_ols=True, verbose_fit=False)),
    }
    fit_kwargs_by_model = {
        "gnar_no_neighbours": dict(use_ols=True, verbose=False),
        "gnar_with_neighbours": dict(use_ols=True, verbose=False),
    }

    table = compare_baselines(
        graph, models=models, fit_kwargs_by_model=fit_kwargs_by_model,
        val_frac=val_frac_equivalent, reference="naive",
    )
    return table

def make_bidirectional_pairs(persistent_pair_list):
    """
    Given a list of canonical (alphabetically ordered) undirected pairs,
    returns a bidirectional edge list containing BOTH directions for each
    pair.

    Parameters
    ----------
    persistent_pair_list : list of tuple
        Canonically labelled undirected edge pairs, with each pair stored
        once as ``(i, j)``.

    Returns
    -------
    list of tuple
        Edge list containing both ``(i, j)`` and ``(j, i)`` for every
        persistent pair.
    """
    bidirectional_edges = []

    for (i, j) in persistent_pair_list:
        bidirectional_edges.append((i, j))
        bidirectional_edges.append((j, i))

    return bidirectional_edges


def run_gnar_edge_rolling_multicov(
    timestamp="1D", L=1, stages_per_lag=None, transform="difference", growth_epsilon=1.0,
    seasonal_period=7, standardize=False, edge_threshold=1.0, n_rolling_test=5, custom_edge_pairs=None,
    verbose=True, weight_mode="count", path=None,
    age_lookup=None, sex_lookup=None, edge_cov_kinds=None,
    exog_series_dict=None,
    interaction_pairs=None, use_ols=True, bidirectional=False
):
    """
    Run rolling-origin GNAR-edge forecasting with edge-level covariates,
    optional exogenous series, and interaction terms.

    This function reads baboon interaction data from a tab-separated file,
    aggregates interactions into time windows, constructs a fixed edge panel
    over persistent baboon pairs, optionally transforms and standardizes the
    edge-weight series, and fits a multivariate GNAR-edge model using rolling
    one-step-ahead evaluation. It supports edge covariates, exogenous time
    series, covariate-interaction terms, and an optional bidirectional
    representation of the persistent network.

    ## Parameters

    timestamp : str, default="1D"
    Pandas frequency string used to floor timestamps into discrete aggregation
    windows.

    L : int, default=1
    Number of autoregressive lags in the GNAR-edge model.

    stages_per_lag : list of list of int, optional
    Neighbour stages to include at each lag. If None, defaults to [[1]] * L.

    transform : {"difference", "seasonal_difference", "growth_rate", "none"}, default="difference"
    Transformation applied to the edge-weight series before fitting.
    "seasonal_difference" computes X_t - X_{t-seasonal_period}, e.g. comparing
    each day to the same day in the previous week when seasonal_period=7,
    to remove weekly-cyclical patterns rather than a smooth local trend.

    growth_epsilon : float, default=1.0
    Small constant added to the denominator when computing growth rates.

    seasonal_period : int, default=7
    Lag used when transform="seasonal_difference" (e.g. 7 for weekly
    differencing of daily data).

    standardize : bool, default=False
    If True, standardize each edge series by its training-period standard
    deviation before fitting.

    edge_threshold : float, default=1.0
    Minimum proportion of initial training windows in which an edge must appear
    to be retained.

    n_rolling_test : int, default=5
    Number of rolling-origin validation windows.

    custom_edge_pairs : list of tuple, optional
    Explicit edge list to use instead of threshold-based edge selection.

    verbose : bool, default=True
    If True, print the model summary, rolling results, and final summary
    dictionary.

    weight_mode : {"count", "proportion_all", "proportion_edges"}, default="count"
    Normalization used for raw edge weights before fitting.

    path : str, default="/Users/admin/Downloads/baboons_proximity_data.txt"
    Path to the tab-separated interaction data file.

    age_lookup : dict, optional
    Mapping from node name to age, used when building edge covariates.

    sex_lookup : dict, optional
    Mapping from node name to sex, used when building edge covariates.

    edge_cov_kinds : list, optional
    Edge-covariate types to include, such as "abs_diff", "mean", or
    "same_sex".

    exog_series_dict : dict, optional
    Dictionary of external time series to pass into the learner.

    interaction_pairs : list of tuple, optional
    Pairs of covariate names to include as interaction terms.

    use_ols : bool, default=True
    Whether to use OLS fitting inside the learner.

    bidirectional : bool, default=False
    If True, includes BOTH directions ``(i, j)`` and ``(j, i)`` for every
    persistent edge, rather than only the canonically labelled direction.
    Both directions are assigned the SAME interaction-count time series,
    since the underlying sensor data carries no directional information.
    This is used as a robustness check on the canonical labelling convention.

    ## Returns

    tuple
        A 6-tuple containing:
        ``(roller, summary_df, rolling_results, summary, graph, per_edge_df)``

    ## Notes

    The function assumes that the helper functions/classes it calls are
    available in the runtime environment, including:

    * build_edge_covariate
    * GNAREdgeGlobalMultiCovLearner
    * ArrayEdgeGraph
    * RollingEdgePredict

    The code also assumes that the input data contains columns named "i",
    "j", and "DateTime", and that "DateTime" matches the format
    "%d/%m/%Y %H:%M".

    When ``bidirectional=True``, the underlying interaction counts are copied
    to both orientations of each persistent pair. This does not create new
    directional information; it duplicates the same undirected interaction
    series under the two possible edge orientations.

    When ``transform="seasonal_difference"``, the first ``seasonal_period``
    windows are consumed by the transform (since ``X_t - X_{t-seasonal_period}``
    requires ``seasonal_period`` prior observations to exist), reducing the
    number of usable time points more than ordinary first differencing does.
    """

    if path is None:
        path = DATA_DIR / "baboons_proximity_data.txt"

    if stages_per_lag is None:
        stages_per_lag = [[1]] * L

    if weight_mode not in ("count", "proportion_all", "proportion_edges"):
        raise ValueError(
            "weight_mode must be one of "
            "'count', 'proportion_all', 'proportion_edges'."
        )

    if transform not in ("difference", "seasonal_difference", "growth_rate", "none"):
        raise ValueError(
            "transform must be one of "
            "'difference', 'seasonal_difference', 'growth_rate', 'none'."
        )

    if edge_cov_kinds is None:
        edge_cov_kinds = []

    if exog_series_dict is None:
        exog_series_dict = {}

    if interaction_pairs is None:
        interaction_pairs = []

    #defining nodes and edges, and aggregating into discrete time stamps
    df = pd.read_csv(path, sep="\t")

    df[["node1", "node2"]] = pd.DataFrame(
        df.apply(
            lambda row: sorted([row["i"], row["j"]]),
            axis=1
        ).tolist(),
        index=df.index
    )

    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        format="%d/%m/%Y %H:%M"
    )

    df["window"] = df["DateTime"].dt.floor(timestamp)

    edges = (
        df.groupby(["window", "node1", "node2"])
        .size()
        .reset_index(name="weight")
    )

    windows = sorted(edges["window"].unique())
    nodes = sorted(set(df["i"]).union(df["j"]))

    if n_rolling_test >= len(windows) - L - 1:
        raise ValueError(
            "n_rolling_test is too large relative to the number of "
            "available windows."
        )

    #checking there is enough data for seasonal differencing, if requested
    if transform == "seasonal_difference" and len(windows) <= seasonal_period:
        raise ValueError(
            f"Not enough windows ({len(windows)}) for "
            f"seasonal_period={seasonal_period}."
        )

    if custom_edge_pairs is not None:
        persistent_pair_list = sorted(custom_edge_pairs)
    else:
        #defiining edge set according to persistence criterion
        #persistence criterion applied to training windows for first rolling step
        initial_train_windows = windows[:-n_rolling_test]

        edges_initial = edges[
            edges["window"].isin(initial_train_windows)
        ].copy()

        edges_count_initial = (
            edges_initial
            .groupby(["node1", "node2"])
            .size()
            .reset_index(name="n_windows")
        )

        min_windows_required = int(
            np.ceil(edge_threshold * len(initial_train_windows))
        )

        persistent_edges = edges_count_initial[
            edges_count_initial["n_windows"] >= min_windows_required
        ].copy()

        persistent_pair_list = list(
            persistent_edges[
                ["node1", "node2"]
            ].itertuples(index=False, name=None)
        )

    #expand to both directions, if requested
    if bidirectional:
        persistent_pair_list = make_bidirectional_pairs(
            persistent_pair_list
        )

    #raw interaction count lookup
    weight_lookup = {
        (r["window"], r["node1"], r["node2"]): r["weight"]
        for _, r in edges.iterrows()
    }

    #node and edge lookups
    node_to_idx = {
        name: i
        for i, name in enumerate(nodes)
    }

    idx_to_node = {
        i: name
        for name, i in node_to_idx.items()
    }

    K = len(nodes)

    edge_idx_list = [
        (node_to_idx[a], node_to_idx[b])
        for a, b in persistent_pair_list
    ]

    edge_names = [
        f"{idx_to_node[a]}-{idx_to_node[b]}"
        for a, b in edge_idx_list
    ]

    E = len(edge_idx_list)

    #building edge covariates
    edge_covariates = {}

    for kind in edge_cov_kinds:
        cov_name = {
            "abs_diff": "age_diff",
            "mean": "mean_age",
            "same_sex": "same_sex"
        }.get(kind, kind)

        edge_covariates[cov_name] = build_edge_covariate(
            edge_names,
            age_lookup=age_lookup,
            sex_lookup=sex_lookup,
            kind=kind
        )

    #building raw interaction count edge panel
    X_level_raw = np.zeros(
        (len(windows), E),
        dtype=float
    )

    #symmetric lookup so both (i,j) and (j,i) retrieve the same
    #underlying interaction count
    def get_weight(window, a, b):
        key1 = (window, a, b)
        key2 = (window, b, a)

        return weight_lookup.get(
            key1,
            weight_lookup.get(key2, 0.0)
        )

    for t, w in enumerate(windows):
        for e, (a, b) in enumerate(persistent_pair_list):
            X_level_raw[t, e] = get_weight(w, a, b)

    daily_total_all = (
        edges.groupby("window")["weight"]
        .sum()
        .reindex(windows, fill_value=0.0)
    )

    #normalising weights
    if weight_mode == "count":
        X_level = X_level_raw.copy()

    elif weight_mode == "proportion_all":
        denom = daily_total_all.to_numpy(dtype=float)

        if np.any(denom == 0.0):
            raise ValueError(
                "At least one window has zero total interactions "
                "across all pairs."
            )

        X_level = X_level_raw / denom[:, None]

    elif weight_mode == "proportion_edges":
        denom = X_level_raw.sum(axis=1)

        if np.any(denom == 0.0):
            raise ValueError(
                "At least one window has zero total interactions "
                "among persistent pairs."
            )

        X_level = X_level_raw / denom[:, None]

    #applying difference / seasonal difference / growth rate transform if requested
    if transform == "difference":
        X_model = np.diff(
            X_level,
            axis=0
        )
        model_windows = windows[1:]

    elif transform == "seasonal_difference":
        #comparing each period to the corresponding period seasonal_period steps earlier
        #(e.g. each day to the same day in the previous week, when seasonal_period=7)
        X_model = X_level[seasonal_period:, :] - X_level[:-seasonal_period, :]
        model_windows = windows[seasonal_period:]

    elif transform == "growth_rate":
        numerator = X_level[1:] - X_level[:-1]
        denominator = X_level[:-1] + growth_epsilon

        X_model = numerator / denominator
        model_windows = windows[1:]

    else:
        X_model = X_level
        model_windows = windows

    n_train_for_std = len(model_windows) - n_rolling_test

    edge_std = X_model[
        :n_train_for_std,
        :
    ].std(axis=0)

    edge_std[edge_std == 0] = 1.0

    #standardising, if requested
    if standardize:
        X_model_fit = X_model / edge_std
    else:
        X_model_fit = X_model
        edge_std = np.ones(
            E,
            dtype=float
        )

    graph = ArrayEdgeGraph.from_edge_panel(
        X_model_fit,
        edge_idx_list,
        n_nodes=K,
        time_labels=model_windows,
        node_labels=nodes
    )

    train_periods = model_windows[:-n_rolling_test]
    val_periods = model_windows[-n_rolling_test:]

    #fitting model and performing rolling edge prediction
    roller = RollingEdgePredict(
        graph=graph,
        learner_cls=GNAREdgeGlobalMultiCovLearner,
        train_periods=train_periods,
        val_periods=val_periods,
        learner_kwargs=dict(
            L=L,
            stages_per_lag=stages_per_lag,
            edge_covariates=edge_covariates,
            exog_series=exog_series_dict,
            interaction_pairs=interaction_pairs
        ),
        fit_kwargs=dict(use_ols=use_ols),
        store_learners=True,
    ).run()

    summary_df = roller.summary()

    rolling_rows = []
    per_edge_errors = []

    window_to_idx = {
        w: i
        for i, w in enumerate(windows)
    }

    #converting each step's predicted values back to raw count form
    for s in roller.steps:
        target_period = s.target_period
        pred_value_fit = s.pred_value
        true_value_fit = s.true_value

        if true_value_fit is None:
            continue

        pred_value = pred_value_fit * edge_std

        test_pos_level = window_to_idx[target_period]

        #adding differenced/seasonal/growth predictions to previous level
        if transform == "difference":
            last_level = X_level[
                test_pos_level - 1,
                :
            ]

            y_pred_level = (
                last_level
                + pred_value
            )

        elif transform == "seasonal_difference":
            #comparing back to the level seasonal_period periods before the target
            last_level_seasonal = X_level[
                test_pos_level - seasonal_period,
                :
            ]

            y_pred_level = (
                last_level_seasonal
                + pred_value
            )

        elif transform == "growth_rate":
            last_level = X_level[
                test_pos_level - 1,
                :
            ]

            y_pred_level = (
                pred_value
                * (last_level + growth_epsilon)
                + last_level
            )

        else:
            y_pred_level = pred_value

        #scaling by the day's aggregate interaction count, if data was normalised
        if weight_mode == "count":
            y_pred_eval = y_pred_level
            y_true_eval = X_level_raw[
                test_pos_level,
                :
            ]

        elif weight_mode == "proportion_all":
            total_all = float(
                daily_total_all.iloc[test_pos_level]
            )

            y_pred_eval = y_pred_level * total_all

            y_true_eval = X_level_raw[
                test_pos_level,
                :
            ]

        elif weight_mode == "proportion_edges":
            total_edges = float(
                X_level_raw[
                    test_pos_level,
                    :
                ].sum()
            )

            y_pred_eval = y_pred_level * total_edges

            y_true_eval = X_level_raw[
                test_pos_level,
                :
            ]

        abs_error = np.abs(
            y_true_eval - y_pred_eval
        )

        per_edge_errors.append(abs_error)

        #summary for each rolling step
        rolling_rows.append({
            "step": s.step,
            "target_period": target_period,
            "MAE": abs_error.mean(),
            "RMSE": np.sqrt((abs_error ** 2).mean()),
            "y_pred_eval": y_pred_eval.copy(),    #ADD: full per-edge array
            "y_true_eval": y_true_eval.copy(),    #ADD: full per-edge array
            "n_train_samples": s.n_train_samples,
            "AIC": None if s.aic_bic is None else s.aic_bic.get("AIC"),
            "BIC": None if s.aic_bic is None else s.aic_bic.get("BIC"),
        })

    rolling_results = pd.DataFrame(
        rolling_rows
    )

    #computing errors for each edge
    per_edge_errors = np.array(
        per_edge_errors
    )

    per_edge_df = pd.DataFrame({
        "edge": edge_names,
        "edge_mae": per_edge_errors.mean(axis=0),
        "edge_rmse": np.sqrt(
            (per_edge_errors ** 2).mean(axis=0)
        ),
        "training_std": edge_std
    }).sort_values(
        "edge_mae",
        ascending=False
    ).reset_index(drop=True)

    last_gammas = (
        roller.steps[-1].learner.gamma_
        if roller.steps[-1].learner is not None
        else None
    )

    all_covs = (
        edge_cov_kinds
        + list(exog_series_dict.keys())
        + [
            f"{e}_X_{t}"
            for e, t in interaction_pairs
        ]
    )

    #summary over the rolling steps
    summary = {
        "model": (
            f"GNAR-edge (multicov: {all_covs}, "
            f"L={L}, transform={transform}, bidirectional={bidirectional})"
        ),
        "timestamp": timestamp,
        "L": L,
        "transform": transform,
        "seasonal_period": seasonal_period if transform == "seasonal_difference" else None,
        "standardize": standardize,
        "weight_mode": weight_mode,
        "n_edges": E,
        "n_rolling_test": n_rolling_test,
        "n_covariates": (
            len(edge_cov_kinds)
            + len(exog_series_dict)
            + len(interaction_pairs)
        ),
        "mean_MAE": rolling_results["MAE"].mean(),
        "mean_RMSE": rolling_results["RMSE"].mean(),
        "std_MAE": rolling_results["MAE"].std(),
        "std_RMSE": rolling_results["RMSE"].std(),
        "fitted_gammas_last_step": last_gammas,
    }

    if verbose:
        print(summary_df)
        print(rolling_results)
        print(summary)

    return (
        roller,
        summary_df,
        rolling_results,
        summary,
        graph,
        per_edge_df,
    )

def build_stages_per_lag(L, r):
    """
    r=0 means no neighbour structure at any lag (equivalent to independent AR).
    r>=1 means stage-1-through-r neighbours included at every lag.
    """
    if r == 0:
        return [[] for _ in range(L)]
    else:
        return [list(range(1, r + 1)) for _ in range(L)]



def run_full_sweep(covariate_configs, L_values, r_values, base_kwargs_fixed):
    """
    Evaluate GNAR-edge models over a grid of lag orders, neighbour stages, and
    covariate specifications.

    This function performs a systematic parameter sweep over the specified
    autoregressive lag orders, neighbour-stage configurations, and covariate
    models. For each combination, a rolling-origin GNAR-edge model is fitted and
    its forecasting performance is summarised using the mean and standard
    deviation of the mean absolute error (MAE) and root mean squared error (RMSE).

    Parameters
    ----------
    covariate_configs : list of dict
        List of covariate model specifications. Each dictionary should contain a
        descriptive ``"label"`` together with any of
        ``"edge_cov_kinds"``, ``"exog_series_dict"``, and
        ``"interaction_pairs"``.
    L_values : iterable of int
        Autoregressive lag orders to evaluate.
    r_values : iterable of int
        Maximum neighbour stages to evaluate. The corresponding
        ``stages_per_lag`` specification is generated automatically using
        ``build_stages_per_lag``.
    base_kwargs_fixed : dict
        Keyword arguments passed unchanged to
        ``run_gnar_edge_rolling_multicov`` for every model fit. These typically
        include parameters such as the aggregation window, transformation,
        standardisation, edge threshold, rolling-test size, weight mode, data
        path, and node covariates.

    Returns
    -------
    pandas.DataFrame
        Data frame containing one row for every model specification with the
        columns

        - ``covariate_config`` : covariate model label.
        - ``L`` : autoregressive lag order.
        - ``r`` : maximum neighbour stage.
        - ``mean_MAE`` : mean rolling-origin MAE.
        - ``mean_RMSE`` : mean rolling-origin RMSE.
        - ``std_MAE`` : standard deviation of rolling-origin MAE.
        - ``std_RMSE`` : standard deviation of rolling-origin RMSE.
        - ``n_covariates`` : total number of edge, time-varying, and interaction
        covariates included in the fitted model.

    Notes
    -----
    Model-fitting failures are caught and reported, allowing the parameter sweep
    to continue uninterrupted. For failed fits, the corresponding performance
    metrics are recorded as missing values.
    """
    all_rows = []

    #looping over lags and neighbour stages
    for L in L_values:
        for r in r_values:
            stages_per_lag = build_stages_per_lag(L, r)

            for cfg in covariate_configs:
                try:
                    _, _, rolling_results, summary, _, _ = run_gnar_edge_rolling_multicov(
                        **base_kwargs_fixed,
                        L=L,
                        stages_per_lag=stages_per_lag,
                        edge_cov_kinds=cfg.get("edge_cov_kinds", []),
                        exog_series_dict=cfg.get("exog_series_dict", {}),
                        interaction_pairs=cfg.get("interaction_pairs", []),
                        use_ols=True,
                        verbose=False,
                    )
                    all_rows.append({
                        "covariate_config": cfg["label"],
                        "L": L,
                        "r": r,
                        "mean_MAE": summary["mean_MAE"],
                        "mean_RMSE": summary["mean_RMSE"],
                        "std_MAE": summary["std_MAE"],
                        "std_RMSE": summary["std_RMSE"],
                        "n_covariates": summary["n_covariates"],
                    })
                except Exception as e:
                    print(f"FAILED: L={L}, r={r}, config={cfg['label']}: {type(e).__name__}: {e}")
                    all_rows.append({
                        "covariate_config": cfg["label"], "L": L, "r": r,
                        "mean_MAE": np.nan, "mean_RMSE": np.nan,
                        "std_MAE": np.nan, "std_RMSE": np.nan,
                        "n_covariates": len(cfg.get("edge_cov_kinds", [])) + len(cfg.get("exog_series_dict", {})) + len(cfg.get("interaction_pairs", [])),
                    })

    return pd.DataFrame(all_rows)

def get_winner_and_loser(df):
    """
    Identify the best- and worst-performing covariate configurations for each
    combination of autoregressive lag and neighbour stage.

    This function groups a summary table by ``(L, r)``, ranks the candidate
    covariate configurations according to their mean RMSE, and records the
    best-performing configuration, the worst-performing configuration, and the
    difference in RMSE between them.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame containing model comparison results. It must include the
        columns ``"L"``, ``"r"``, ``"covariate_config"``, and ``"mean_RMSE"``.

    Returns
    -------
    pandas.DataFrame
        Data frame with one row for each ``(L, r)`` combination containing

        - ``L`` : autoregressive lag order.
        - ``r`` : neighbour stage.
        - ``Best_config`` : covariate configuration with the lowest mean RMSE.
        - ``Best_RMSE`` : lowest mean RMSE.
        - ``Worst_config`` : covariate configuration with the highest mean RMSE.
        - ``Worst_RMSE`` : highest mean RMSE.
        - ``RMSE_gap`` : difference between the worst and best mean RMSE.

    Notes
    -----
    The function assumes that lower RMSE indicates better predictive performance.
    If multiple configurations have identical RMSE values, the first occurring
    configuration after sorting is selected.
    """
    results = []
    
    # Group by each (L, r) combination
    for (L, r), group in df.groupby(["L", "r"]):
        # sort configurations by mean_RMSE ascending
        sorted_group_best = group.sort_values("mean_RMSE").reset_index(drop=True)
        sorted_group_worst = group.sort_values("mean_RMSE", ascending=False).reset_index(drop=True)

        # 1st place (winner)
        winner_cfg = sorted_group_best.loc[0, "covariate_config"]
        winner_rmse = sorted_group_best.loc[0, "mean_RMSE"]

        #last place (loser)
        loser_cfg = sorted_group_worst.loc[0, "covariate_config"]
        loser_rmse = sorted_group_worst.loc[0, "mean_RMSE"]
        
        rmse_diff = loser_rmse - winner_rmse
            
        results.append({
            "L": L,
            "r": r,
            "Best_config": winner_cfg,
            "Best_RMSE": round(winner_rmse, 4),
            "Worst_config": loser_cfg,
            "Worst_RMSE": round(loser_rmse, 4),
            "RMSE_gap": round(rmse_diff, 5)
        })
        
    return pd.DataFrame(results).sort_values(["L", "r"]).reset_index(drop=True)

def compute_naive_baseline_mae(edge_threshold, timestamp="1D", n_rolling_test=5,
                                weight_mode="proportion_edges",
                                path=None):
    """
    Compute the naive one-step-ahead baseline mean absolute error (MAE).

    This function constructs the persistent-edge network using the specified edge
    selection threshold, builds the corresponding edge-weight panel on the raw count
    scale, and evaluates the naive forecasting baseline over the rolling test
    period. The naive forecast predicts each edge weight using its value from the
    previous time window.

    Parameters
    ----------
    edge_threshold : float
        Minimum proportion of training windows in which an edge must appear to be
        retained in the persistent network.
    timestamp : str, default="1D"
        Pandas frequency string used to aggregate interaction timestamps into
        discrete time windows.
    n_rolling_test : int, default=5
        Number of rolling-origin test windows.
    weight_mode : {"count", "proportion_all", "proportion_edges"}, \
    default="proportion_edges"
        Included for consistency with the main modelling pipeline. The naive
        baseline is always evaluated on the raw count scale regardless of this
        argument.
    path : str
        Path to the tab-separated interaction dataset.

    Returns
    -------
    dict
        Dictionary containing

        - ``edge_threshold`` : edge persistence threshold.
        - ``n_edges`` : number of persistent edges.
        - ``baseline_MAE`` : mean absolute error of the naive forecasting
        baseline over the rolling test period.

    Notes
    -----
    The baseline predicts each edge weight at time ``t`` using its observed value
    at time ``t - 1``. Errors are always computed on the raw interaction count
    scale to ensure comparability with the GNAR-edge model evaluation.
    """
    #building edges and nodes, and aggregating into discrete time windows
    df = pd.read_csv(path, sep="\t")
    df[["node1", "node2"]] = pd.DataFrame(
        df.apply(lambda row: sorted([row["i"], row["j"]]), axis=1).tolist(), index=df.index
    )
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%d/%m/%Y %H:%M")
    df["window"] = df["DateTime"].dt.floor(timestamp)
    edges = df.groupby(["window", "node1", "node2"]).size().reset_index(name="weight")
    windows = sorted(edges["window"].unique())
    nodes = sorted(set(df["i"]).union(df["j"]))

    #defining edge set based on persistent edge criterion, applied to first rolling time step training windows
    initial_train_windows = windows[:-n_rolling_test]
    edges_initial = edges[edges["window"].isin(initial_train_windows)].copy()
    edges_count_initial = edges_initial.groupby(["node1", "node2"]).size().reset_index(name="n_windows")
    min_windows_required = int(np.ceil(edge_threshold * len(initial_train_windows)))
    persistent_edges = edges_count_initial[edges_count_initial["n_windows"] >= min_windows_required].copy()
    persistent_pair_list = list(persistent_edges[["node1", "node2"]].itertuples(index=False, name=None))

    #raw interaction count lookup
    weight_lookup = {(r["window"], r["node1"], r["node2"]): r["weight"] for _, r in edges.iterrows()}
    E = len(persistent_pair_list)

    #always build the raw count panel -- this is the evaluation scale, regardless of weight_mode
    X_level_raw = np.zeros((len(windows), E), dtype=float)
    for t, w in enumerate(windows):
        for e, (a, b) in enumerate(persistent_pair_list):
            X_level_raw[t, e] = weight_lookup.get((w, a, b), 0.0)

    #naive baseline evaluated in raw count scale
    test_start_idx = len(windows) - n_rolling_test
    naive_errors = []
    for t in range(test_start_idx, len(windows)):
        y_true = X_level_raw[t, :]
        y_naive_pred = X_level_raw[t - 1, :]
        naive_errors.append(np.abs(y_true - y_naive_pred).mean())

    baseline_mae = np.mean(naive_errors)

    return {
        "edge_threshold": edge_threshold,
        "n_edges": E,
        "baseline_MAE": baseline_mae,
    }


def full_mase_sweep(base_kwargs_fixed, L_values, r_values, covariate_configs, thresholds=(1.0, 0.9, 0.8)):
    """
    Evaluate GNAR-edge models over multiple specifications using the mean absolute
    scaled error (MASE).

    This function performs a grid search over autoregressive lag orders, neighbour
    structures, edge-selection thresholds, and covariate configurations. For each
    combination, it fits the GNAR-edge model, computes the mean absolute error
    (MAE), and compares it with the corresponding naive forecasting baseline to
    calculate the mean absolute scaled error (MASE).

    Parameters
    ----------
    base_kwargs_fixed : dict
        Dictionary of arguments passed unchanged to
        ``run_gnar_edge_rolling_multicov`` for every model fit.
    L_values : iterable of int
        Autoregressive lag orders to evaluate.
    r_values : iterable of int
        Maximum neighbour stages to evaluate.
    covariate_configs : list of dict
        Covariate model specifications. Each dictionary should contain a
        descriptive ``"label"`` together with any of
        ``"edge_cov_kinds"``, ``"exog_series_dict"``, and
        ``"interaction_pairs"``.
    thresholds : iterable of float, default=(1.0, 0.9, 0.8)
        Edge persistence thresholds defining the fixed network used for each
        model fit.

    Returns
    -------
    pandas.DataFrame
        Data frame containing one row for every evaluated model specification with
        the columns

        - ``covariate_config``
        - ``L``
        - ``r``
        - ``edge_threshold``
        - ``n_edges``
        - ``model_MAE``
        - ``baseline_MAE``
        - ``MASE``

    Notes
    -----
    The naive baseline MAE is computed only once for each edge threshold and reused
    for all model specifications sharing that threshold. Model-fitting failures are
    caught and recorded as missing values so that the full parameter sweep can
    continue uninterrupted.
    """
    all_rows = []

    # compute baseline MAE once per threshold
    baseline_cache = {}
    for threshold in thresholds:
        try:
            baseline_info = compute_naive_baseline_mae(
                edge_threshold=threshold,
                timestamp=base_kwargs_fixed.get("timestamp", "1D"),
                n_rolling_test=base_kwargs_fixed.get("n_rolling_test", 5),
                weight_mode=base_kwargs_fixed.get("weight_mode", "proportion_edges"),
                path=base_kwargs_fixed.get("path"),
            )
            baseline_cache[threshold] = baseline_info
        except Exception as e:
            print(f"FAILED baseline: threshold={threshold}: {type(e).__name__}: {e}")
            baseline_cache[threshold] = {
                "baseline_MAE": np.nan,
                "n_edges": np.nan,
            }

    #looping over lags and neighbour stages
    for L in L_values:
        for r in r_values:
            stages_per_lag = build_stages_per_lag(L, r)
            #looping over covariate configurations
            for cfg in covariate_configs:
                #looping over edge thresholds
                for threshold in thresholds:
                    try:
                        # clean kwargs to avoid collisions
                        run_kwargs = base_kwargs_fixed.copy()
                        run_kwargs.pop("edge_threshold", None)

                        # run model
                        _, _, _, summary, _, _ = run_gnar_edge_rolling_multicov(
                            **run_kwargs,
                            edge_threshold=threshold,
                            L=L,
                            stages_per_lag=stages_per_lag,
                            edge_cov_kinds=cfg.get("edge_cov_kinds", []),
                            exog_series_dict=cfg.get("exog_series_dict", {}),
                            interaction_pairs=cfg.get("interaction_pairs", []),
                            use_ols=True,
                            verbose=False,
                        )

                        model_mae = summary["mean_MAE"]
                        baseline_mae = baseline_cache[threshold]["baseline_MAE"]
                        n_edges = baseline_cache[threshold]["n_edges"]

                        #computing MASE
                        if pd.notna(baseline_mae) and baseline_mae != 0:
                            mase = model_mae / baseline_mae
                        else:
                            mase = np.nan

                        all_rows.append({
                            "covariate_config": cfg["label"],
                            "L": L,
                            "r": r,
                            "edge_threshold": threshold,
                            "n_edges": n_edges,
                            "model_MAE": model_mae,
                            "baseline_MAE": baseline_mae,
                            "MASE": mase,
                        })

                    except Exception as e:
                        print(
                            f"FAILED: L={L}, r={r}, threshold={threshold}, "
                            f"config={cfg['label']}: {type(e).__name__}: {e}"
                        )
                        all_rows.append({
                            "covariate_config": cfg["label"],
                            "L": L,
                            "r": r,
                            "edge_threshold": threshold,
                            "n_edges": baseline_cache.get(threshold, {}).get("n_edges", np.nan),
                            "model_MAE": np.nan,
                            "baseline_MAE": baseline_cache.get(threshold, {}).get("baseline_MAE", np.nan),
                            "MASE": np.nan,
                        })

    return pd.DataFrame(all_rows)

