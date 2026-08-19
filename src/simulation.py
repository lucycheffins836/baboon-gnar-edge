#loading packages
import numpy as np
import pandas as pd
import igraph as ig
from scipy.stats import pearsonr
import sys    
from pathlib import Path                

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "external"))

from BaseEdgeGNAR_edge import GNAREdgeLearner
from BaseEdgeGNAR_edge_global import GNAREdgeGlobalLearner
from edge_baselines import ARIMAEdgeBaseline, NaiveEdgeBaseline, ZeroEdgeBaseline, compare_baselines,  DFMEdgeBaseline, dfm_pca_one_step
from edge_graph import ArrayEdgeGraph
from RollingEdgePredict import RollingEdgePredict
from model import build_edge_covariate, GNAREdgeGlobalMultiCovLearner




def calibrate_rdp_radius(K, target_density, n_trials=30, radius_range=(0.1, 3.0), tol=0.01):
    """Empirically calibrate the radius parameter for rdp graph generator.

    This function searches for a radius value that yields an average edge density
    close to ``target_density`` under repeated Monte Carlo simulation. For each trial,
    node positions are sampled from a standard normal distribution in two dimensions,
    normalized to lie on a circle of the proposed radius, and then an undirected edge
    between nodes ``i`` and ``j`` is added with probability ``clip(dot(pos_i, pos_j), 0, 1)``.
    A binary search is used to adjust the radius until the simulated mean density is
    within ``tol`` of the target, or until the search iterations are exhausted.

    Parameters
    ----------
    K : int
        Number of nodes in the generated network.
    target_density : float
        Desired edge density, expressed as a proportion of all possible undirected edges.
    n_trials : int, default=30
        Number of Monte Carlo simulations used to estimate the density for a given radius.
    radius_range : tuple of float, default=(0.1, 3.0)
        Lower and upper bounds for the binary search over radius.
    tol : float, default=0.01
        Absolute tolerance between the achieved and target density at which the search
        may stop early.

    Returns
    -------
    tuple of float
        A pair ``(radius, final_density)`` where ``radius`` is the calibrated radius
        and ``final_density`` is the average density achieved at that radius across
        ``n_trials`` simulations.
    """
    #calculating average density that results from a given radius
    def mean_density_for_radius(radius, n_trials=n_trials):
        densities = []
        for seed in range(n_trials):
            #simulates an rdp graph with chosen radius
            rng = np.random.default_rng(seed)
            positions = rng.normal(0, 1, size=(K, 2))
            positions = positions / np.linalg.norm(positions, axis=1, keepdims=True) * radius
            n_edges = 0
            for i in range(K):
                for j in range(i+1, K):
                    p_edge = np.clip(np.dot(positions[i], positions[j]), 0, 1)
                    if rng.random() < p_edge:
                        n_edges += 1
             #stores density for this rdp graph           
            densities.append(n_edges / (K*(K-1)/2))
        #calculates average density over many trials
        return np.mean(densities)

    lo, hi = radius_range
    for _ in range(40):   
        #binary search
        mid = (lo + hi) / 2
        d = mean_density_for_radius(mid, n_trials=15)   
        if abs(d - target_density) < tol:
            break
        if d < target_density:
            lo = mid
        else:
            hi = mid

    final_density = mean_density_for_radius(mid, n_trials=n_trials)
    #returns the calibrated radius and the resulting average density
    return mid, final_density

def get_or_calibrate_rdp_radius(K,rdp_radius_cache, density, n_trials=30):
    """Retrieve a cached RDP radius for a given graph size and density, or calibrate it.

    This function uses a global cache keyed by ``(K, round(density, 4))``. If the
    requested combination is not already cached, it calls ``calibrate_rdp_radius``,
    stores the resulting radius, and prints a short calibration message. The cached
    radius is then returned.

    Parameters
    ----------
    K : int
        Number of nodes in the network.
    density : float
        Target edge density for the RDP generator.
    n_trials : int, default=30
        Number of Monte Carlo simulations to use if calibration is required.
    rdp_radius_cache: dict
        Dictionary of cached calibrated radii

    Returns
    -------
    float
        The cached or newly calibrated radius associated with the requested
        ``(K, density)`` combination.

    """
    key = (K, round(density, 4))
    if key not in rdp_radius_cache:
        radius, achieved = calibrate_rdp_radius(K=K, target_density=density, n_trials=n_trials)
        rdp_radius_cache[key] = radius
        print(f"Calibrated RDP radius for K={K}, density={density:.4f}: radius={radius:.3f}, achieved={achieved:.3f}")
    return rdp_radius_cache[key]

def generate_network(K, density=0.3, structure="ER", rdp_radius_cache = None, seed=None):
    """Generate a synthetic undirected network under one of several random graph models.

    The function supports three structures:
    ``"ER"``
        Erdős-Rényi graph with edge probability equal to ``density``.
    ``"SBM"``
        Two-block stochastic block model with stronger within-block than between-block
        connection probabilities.
    ``"RDP"``
        Radius-based graph generator using node positions on a circle of calibrated
        radius and probabilistic edge formation based on dot products of positions.

    The returned graph is converted into a plain edge list.

    Parameters
    ----------
    K : int
        Number of nodes in the network.
    density : float, default=0.3
        Target density or connection probability used by the selected graph model.
    structure : {"ER", "SBM", "RDP"}, default="ER"
        Random network structure to generate.
    rdp_radius_cache: dict
        Dictionary of cached calibrated radii
    seed : int, optional
        Random seed used for reproducibility.

    Returns
    -------
    tuple
        A tuple ``(edge_list, K)`` where ``edge_list`` is a list of undirected edges
        represented as ``(u, v)`` node-index pairs and ``K`` is the number of nodes.

    Raises
    ------
    ValueError
        If ``structure`` is not one of ``"ER"``, ``"SBM"``, or ``"RDP"``.

    Notes
    -----
    This function requires ``igraph`` to be available as ``ig`` in the runtime
    environment.
    """
    rng = np.random.default_rng(seed)

    #generate ER graph
    if structure == "ER":
        G = ig.Graph.Erdos_Renyi(n=K, p=density, directed=False)

    #generate SBM network
    elif structure == "SBM":
        n_per_block = K // 2
        block_sizes = [n_per_block, K - n_per_block]
        pref_matrix = [[density*1.75, density*0.25], [density*0.25, density*1.75]]
        G = ig.Graph.SBM(pref_matrix=pref_matrix, block_sizes=block_sizes, directed=False)

    #generate RDP network using calibrated radius
    elif structure == "RDP":
        radius = get_or_calibrate_rdp_radius(K=K, rdp_radius_cache=rdp_radius_cache, density=density)
        positions = rng.normal(0, 1, size=(K, 2))
        positions = positions / np.linalg.norm(positions, axis=1, keepdims=True) * radius
        edges = []
        for i in range(K):
            for j in range(i+1, K):
                p_edge = np.clip(np.dot(positions[i], positions[j]), 0, 1)
                if rng.random() < p_edge:
                    edges.append((i, j))
        G = ig.Graph(n=K, edges=edges, directed=False)

    else:
        raise ValueError("structure must be 'ER', 'SBM', or 'RDP'")

    edge_list = [(e.source, e.target) for e in G.es]
    return edge_list, K

def simulate_node_covariates(K, seed = None):
    """Simulate node-level covariates for a synthetic baboon network.

    This helper generates two node attributes:
    an age-like continuous variable drawn uniformly from 5 to 25, and a binary sex
    label drawn from ``{"M", "F"}`` with approximately equal probability.

    Parameters
    ----------
    K : int
        Number of nodes.
    seed : int, optional
        Random seed used for reproducibility.

    Returns
    -------
    tuple of numpy.ndarray
        A pair ``(ages, sexes)`` where ``ages`` is a length-``K`` array of floats
        and ``sexes`` is a length-``K`` array of string labels.
    """
    #creating random number generator
    rng = np.random.default_rng(seed)
    #simulating ages between 5-25 years
    ages = rng.uniform(5, 25, size = K)
    #simulate sex, roughly balanced
    sexes = rng.choice(["M", "F"], size = K)
    return ages, sexes

def build_synthetic_edge_covariates(edge_list, ages, sexes):
    """Construct edge-level covariates from node-level ages and sexes.

    For each edge ``(u, v)``, the function computes:
    ``age_diff``
        Absolute age difference ``|age_u - age_v|``.
    ``mean_age``
        Mean age ``(age_u + age_v) / 2``.
    ``same_sex``
        Indicator equal to 1.0 when the two nodes have the same sex label and 0.0
        otherwise.

    Parameters
    ----------
    edge_list : list of tuple
        List of undirected edges encoded as node-index pairs ``(u, v)``.
    ages : array-like
        Node ages indexed by node id.
    sexes : array-like
        Node sex labels indexed by node id.

    Returns
    -------
    dict of numpy.ndarray
        Dictionary with keys ``"age_diff"``, ``"mean_age"``, and ``"same_sex"``,
        each mapped to a length-``len(edge_list)`` array of edge covariate values.
    """
    age_diff, mean_age, same_sex = [], [], []
    for u, v in edge_list:
        age_diff.append(abs(ages[u] - ages[v]))
        mean_age.append((ages[u] + ages[v]) / 2)
        same_sex.append(1.0 if sexes[u] == sexes[v] else 0)

    return {
        "age_diff": np.array(age_diff),
        "mean_age": np.array(mean_age),
        "same_sex": np.array(same_sex)
    }

def simulate_ar1_exog(T, phi = 0.7, mean = 30, sd = 3.0, seed = None):
    """
    Simulate an AR(1) exogenous time series.

    The generated series is centered around ``mean`` and follows the recursion

        z[t] = mean + phi * (z[t-1] - mean) + epsilon[t]

    where ``epsilon[t]`` is Gaussian noise with variance chosen so that the
    stationary standard deviation is approximately ``sd``. The first value is
    initialized from a normal distribution with the requested mean and standard
    deviation.

    Parameters
    ----------
    T : int
        Length of the time series to generate.
    phi : float, default=0.7
        AR(1) persistence parameter.
    mean : float, default=30
        Long-run mean of the process.
    sd : float, default=3.0
        Target stationary standard deviation of the process.
    seed : int, optional
        Random seed used for reproducibility.

    Returns
    -------
    numpy.ndarray
        A length-``T`` array containing the simulated AR(1) series.
    """
    #creating random number generator
    rng = np.random.default_rng(seed)
    #definining variance of the innovations so that variance of
    #x_t is stationary at sd^2
    innovation_sd = sd * np.sqrt(1 - phi**2)

    #z = simulated exogenous series
    z = np.zeros(T)
    z[0] = rng.normal(mean, sd)
    for t in range(1, T):
        z[t] = mean + phi * (z[t-1] - mean) + rng.normal(0, innovation_sd)

    return z


def build_W_matrices(edge_list, K, needed_r):
    """
    Construct neighbour-selection matrices from a graph edge list.

    This function builds a temporary ``ArrayEdgeGraph`` from a dummy edge panel
    containing only zeros, solely so that the graph's neighbour-matrix machinery can
    be reused. It then returns the neighbour matrix for each requested stage ``r``.

    Parameters
    ----------
    edge_list : list of tuple
        Undirected edge list encoded as node-index pairs ``(u, v)``.
    K : int
        Number of nodes in the network.
    needed_r : iterable of int
        Neighbour stages to extract from the graph.

    Returns
    -------
    tuple
        A pair ``(W_dict, graph)`` where ``W_dict`` maps each requested stage ``r``
        to its corresponding neighbour matrix and ``graph`` is the constructed
        ``ArrayEdgeGraph`` instance.
    """
    #creating a placeholder edge panel, where only the structure of X matters
    dummy_X = np.zeros((2, len(edge_list)))
    graph = ArrayEdgeGraph.from_edge_panel(dummy_X, edge_list, n_nodes = K)
    #returns the graph and its neighbour matrices
    return {r: graph.neighbor_matrix(r) for r in needed_r}, graph


def build_design_matrix(learner, train_set):
    """
    Reconstruct the OLS design matrix used by a GNAR-edge learner.

    This function mirrors the learner's internal feature construction so that the
    resulting matrix can be used for diagnostics such as rank checks, condition
    numbers, and standard-error calculations. The columns include:
    lagged edge weights, neighbour-edge features from ``_build_ols_beta_features``,
    edge-level covariates, time-varying covariates, and interaction terms between
    edge-level and time-varying covariates.

    Parameters
    ----------
    learner : object
        Fitted or partially fitted GNAR-edge learner exposing the attributes and
        helper methods used below, including ``L``, ``E``, ``edge_cov_names``,
        ``time_cov_names``, ``interaction_pairs``, ``edge_covariates``,
        ``_build_ols_beta_features``, and ``_get_time_cov``.
    train_set : object
        Training set object exposing ``X`` and ``sample_target_periods``.

    Returns
    -------
    tuple
        A pair ``(D, feature_names)`` where ``D`` is the reconstructed design matrix
        of shape ``(N * E, n_features)`` and ``feature_names`` is a list of column
        labels in the same order as the columns of ``D``.
    """

    X = np.asarray(train_set.X,
                   dtype = float)
    N, _, E = X.shape

    cols = []
    feature_names = []

    #adding lagged edge weights to design matrix for each edge
    for l in range(1, learner.L + 1):
        cols.append(X[:, -l, :].reshape(-1))
        feature_names.extend([f"lag{l}_edge{j}" for j in range(E)])

    #adding neighbour edge weights
    beta_pairs, Z_list = learner._build_ols_beta_features(X)
    for j, Z in enumerate(Z_list, start = 1):
        cols.append(Z.reshape(-1))
        feature_names.append(f"Z{j}")

    #adding per-edge exogenous covariate values
    for name in learner.edge_cov_names:
        cols.append(np.tile(learner.edge_covariates[name], N))
        feature_names.append(f"edge_cov::{name}")
    
    #adding time varying exogenous covariate
    for name in learner.time_cov_names:
        z = learner._get_time_cov(name, train_set.sample_target_periods)
        cols.append(np.repeat(z, learner.E))
        feature_names.append(f"time_cov::{name}")
    
    #adding interaction terms
    for edge_name, time_name in learner.interaction_pairs:
        edge_part = np.tile(learner.edge_covariates[edge_name], N)
        time_part = np.repeat(
            learner._get_time_cov(time_name, train_set.sample_target_periods),
            learner.E,
        )
        cols.append(edge_part * time_part)
        feature_names.append(f"interaction::{edge_name}x{time_name}")

    if len(cols) == 0:
        D = np.empty((N*E, 0))
    else:
        D = np.column_stack(cols)

    return D, feature_names

def simulate_full_model(edge_list, K, L, stages_per_lag,
                        true_alpha, true_beta_dict, edge_covariates, true_gamma_edge_cov,
                        true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
                        T, noise_std=1.0, time_cov_name="sim_temp", seed=None):
    """
        Simulate a full multivariate GNAR-edge process with covariate effects.
    
        The function generates an edge-weight panel according to a user-specified
        data-generating process with autoregressive terms, neighbour-stage effects,
        edge-level covariate effects, a time-varying exogenous covariate, and optional
        interaction effects between edge and time covariates. It also constructs the
        corresponding ``ArrayEdgeGraph`` used by the model-fitting code.
    
        Parameters
        ----------
        edge_list : list of tuple
            Undirected edge list encoded as node-index pairs ``(u, v)``.
        K : int
            Number of nodes in the network.
        L : int
            Number of autoregressive lags.
        stages_per_lag : list of list of int
            Neighbour stages included at each lag. The ``l``-th entry corresponds to
            lag ``l + 1``.
        true_alpha : array-like
            True autoregressive coefficients, one per lag.
        true_beta_dict : dict
            Mapping ``(lag, stage)`` to the true neighbour-effect coefficient.
        edge_covariates : dict
            Dictionary of edge-level covariate arrays keyed by covariate name.
        true_gamma_edge_cov : dict
            Mapping from edge-covariate name to its true coefficient.
        true_gamma_time_cov : float
            True coefficient on the time-varying exogenous covariate.
        interaction_pairs : list of tuple
            Pairs ``(edge_name, time_name)`` identifying interaction terms.
        true_gamma_interaction : dict
            Mapping from interaction pair to its true coefficient.
        T : int
            Total length of the simulated time series.
        noise_std : float, default=1.0
            Standard deviation of the Gaussian noise added at each time step.
        seed : int, optional
            Random seed used for reproducibility.
    
        Returns
        -------
        tuple
            A triple ``(X, time_cov, graph)`` where ``X`` is the simulated edge panel
            of shape ``(T, E)``, ``time_cov`` is the simulated exogenous series, and
            ``graph`` is the ``ArrayEdgeGraph`` containing the network structure.
        """
    rng = np.random.default_rng(seed)
    E = len(edge_list)

    needed_r = sorted({r for rs in stages_per_lag for r in rs})
    W_dict, graph = build_W_matrices(edge_list, K, needed_r)

    time_cov = simulate_ar1_exog(T, phi=0.7, mean=30, sd=3, seed=seed)

    X = np.zeros((T, E))
    X[:L, :] = rng.normal(0, noise_std, size=(L, E))

    for t in range(L, T):
        pred = np.zeros(E)
        for l in range(1, L + 1):
            pred += true_alpha[l-1] * X[t-l, :]
            for r in stages_per_lag[l-1]:
                pred += true_beta_dict[(l, r)] * (W_dict[r] @ X[t-l, :])

        for name, gamma_val in true_gamma_edge_cov.items():
            pred += gamma_val * edge_covariates[name]

        pred += true_gamma_time_cov * time_cov[t]

        for (edge_name, cov_name) in interaction_pairs:
            gamma_val = true_gamma_interaction[(edge_name, cov_name)]
            pred += gamma_val * edge_covariates[edge_name] * time_cov[t]

        X[t, :] = pred + rng.normal(0, noise_std, size=E)

    return X, time_cov, graph

def run_fixed_network_replication(
    edge_list, K, L, stages_per_lag,
    true_alpha, true_beta_dict, true_gamma_edge_cov,
    true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
    T, seed, ages=None, sexes=None
):
    """
    Run one simulation replication on a fixed, pre-specified network topology.

    This function simulates an extended GNAR-edge process on a fixed network,
    fits the extended GNAR-edge model to the simulated data, and constructs a
    per-parameter results table containing point estimates, 95% confidence
    intervals, and indicators of whether the confidence intervals cover the
    supplied true parameter values.

    Parameters
    ----------
    edge_list : list of tuple
        Fixed edge list defining the network topology.
    K : int
        Number of nodes.
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag.
    true_alpha : array-like
        True autoregressive coefficients used in the data-generating process.
    true_beta_dict : dict
        Dictionary mapping ``(lag, stage)`` pairs to true neighbour-effect
        coefficients.
    true_gamma_edge_cov : dict
        Dictionary mapping edge-covariate names to their true coefficients.
    true_gamma_time_cov : float
        True coefficient for the time-varying exogenous covariate.
    interaction_pairs : list of tuple
        Interaction terms included in the fitted model.
    true_gamma_interaction : dict
        Dictionary mapping interaction terms to their true coefficients.
    T : int
        Number of time points.
    seed : int
        Random seed used for reproducibility.
    ages : array-like, optional
        Node-level ages used to construct edge covariates. If ``None``,
        synthetic ages are simulated.
    sexes : array-like, optional
        Node-level sexes used to construct edge covariates. If ``None``,
        synthetic sexes are simulated.

    Returns
    -------
    tuple
        A tuple containing:

        - ``result_table`` : pandas.DataFrame
            Per-parameter true values, estimates, 95% confidence intervals,
            and coverage indicators.
        - ``learner`` : GNAREdgeGlobalMultiCovLearner
            Fitted extended GNAR-edge model.
        - ``graph`` : ArrayEdgeGraph
            Simulated network graph.
    """

    time_cov_name = (
    interaction_pairs[0][1]
    if len(interaction_pairs) > 0
    else "sim_temp"
    )

    # Simulating node-level covariates if they were not supplied
    if ages is None or sexes is None:
        ages, sexes = simulate_node_covariates(K=K, seed=seed)

    # Building edge covariates
    edge_covariates_full = build_synthetic_edge_covariates(
        edge_list, ages, sexes
    )
    edge_covariates_sim = {
        name: edge_covariates_full[name]
        for name in true_gamma_edge_cov.keys()
    }

    # Simulating the model on the fixed network
    X_sim, time_cov_sim, graph = simulate_full_model(
        edge_list,
        K,
        L,
        stages_per_lag,
        true_alpha,
        true_beta_dict,
        edge_covariates_sim,
        true_gamma_edge_cov,
        true_gamma_time_cov,
        interaction_pairs,
        true_gamma_interaction,
        T=T,
        time_cov_name=time_cov_name,
        seed=seed,
    )

    # Setting time index
    time_index = pd.date_range(
        "2020-01-01",
        periods=T,
        freq="D"
    )

    exog_series_sim = {}

    if true_gamma_time_cov != 0.0 or len(interaction_pairs) > 0:
        exog_series_sim = {
            time_cov_name: pd.Series(time_cov_sim, index=time_index)
        }

    # Building graph object
    sim_graph = ArrayEdgeGraph.from_edge_panel(
        X_sim,
        edge_list,
        n_nodes=K,
        time_labels=time_index
    )

    # Fitting the extended GNAR-edge model
    learner = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph,
        L=L,
        stages_per_lag=stages_per_lag,
        train_periods=time_index,
        use_ols=True,
        edge_covariates=edge_covariates_sim,
        exog_series=exog_series_sim,
        interaction_pairs=interaction_pairs,
    )

    learner.fit(
        use_ols=True,
        verbose=False
    )

    # Building design matrix and computing standard errors
    D, feature_names = build_design_matrix(
        learner,
        learner.train_set
    )
    se_theta = compute_ols_standard_errors(
        learner,
        D
    )

    # Extracting estimated beta coefficients in the fitted order
    beta_pairs_actual = learner.ols_beta_pairs_

    beta_hat_array = np.array([
        learner.beta[pair]
        for pair in beta_pairs_actual
    ])

    beta_true_array = np.array([
        true_beta_dict.get(pair, np.nan)
        for pair in beta_pairs_actual
    ])

    # Extracting time-covariate truth only when it is included
    gamma_time_true = (
        [true_gamma_time_cov]
        if time_cov_name in learner.time_cov_names
        else []
    )

    # Combining all estimated parameters
    theta_hat = np.concatenate([
        learner.alpha.flatten(),
        beta_hat_array,
        np.array([
            learner.gamma_[name]
            for name in learner.edge_cov_names
        ]),
        np.array([
            learner.gamma_[name]
            for name in learner.time_cov_names
        ]),
        np.array([
            learner.gamma_[f"{e}_X_{t}"]
            for e, t in learner.interaction_pairs
        ]),
    ])

    # Combining all true parameter values in the same order
    true_theta = np.concatenate([
        np.array(true_alpha),
        beta_true_array,
        np.array(list(true_gamma_edge_cov.values())),
        np.array(gamma_time_true),
        np.array(list(true_gamma_interaction.values())),
    ])

    # Computing 95% confidence intervals
    ci_lower = theta_hat - 1.96 * se_theta
    ci_upper = theta_hat + 1.96 * se_theta

    # Determining whether the true value lies inside the confidence interval
    covers_true = (
        (true_theta >= ci_lower)
        & (true_theta <= ci_upper)
    )

    # Parameter names in the same order as theta_hat
    param_names = (
        [f"alpha{l}" for l in range(1, L + 1)]
        + [
            f"beta{l},{r}"
            for l, r in beta_pairs_actual
        ]
        + list(true_gamma_edge_cov.keys())
        + (
            [time_cov_name]
            if (
                time_cov_name in learner.time_cov_names
                and true_gamma_time_cov != 0.0
            )
            else []
        )
        + [
            f"{e}_X_{t}"
            for e, t in interaction_pairs
        ]
    )

    # Building final per-parameter results table
    result_table = pd.DataFrame({
        "parameter": param_names,
        "true_value": true_theta,
        "estimated": theta_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "covers_true": covers_true,
    })

    return result_table, learner, graph

def compute_ols_standard_errors(learner, D):
    """
    Compute OLS standard errors for the parameters estimated by a GNAR-edge learner.

    This function uses the standard linear-model variance formula

        Var(theta_hat) = sigma^2 (D' D)^-1

    where ``sigma^2`` is estimated from the learner's OLS residuals and ``D`` is the
    design matrix supplied as input. The diagonal of the Moore-Penrose pseudoinverse
    of ``D' D`` is used so that the calculation remains numerically stable when the
    matrix is ill-conditioned or rank-deficient.

    Parameters
    ----------
    learner : object
        Fitted learner exposing an ``ols_residuals_`` attribute containing the OLS
        residuals used in the fit.
    D : numpy.ndarray
        Design matrix with shape ``(n_obs, n_params)``.

    Returns
    -------
    numpy.ndarray
        A length-``n_params`` array of standard errors. If the degrees of freedom
        are not positive, the function returns an array of ``np.nan`` values.

    Notes
    -----
    This routine assumes the learner was fitted by OLS on the same design matrix
    passed here.
    """
    #computing the residuals
    resid = learner.ols_residuals_.reshape(-1)

    #computing degrees of freedom
    n_obs, n_params = D.shape
    dof = n_obs - n_params
    if dof <= 0:
        return np.full(n_params, np.nan)
    
    #computing std errors
    sigma2_hat = np.sum(resid**2) / dof
    DtD_inv = np.linalg.pinv(D.T @ D)
    var_theta = sigma2_hat * np.diag(DtD_inv)
    se_theta = np.sqrt(np.maximum(var_theta, 0))

    return se_theta

def run_full_simulation_replication(K, density, structure, rdp_radius_cache, L, stages_per_lag,
                                    true_alpha, true_beta_dict, true_gamma_edge_cov,
                                    true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
                                    T, seed):
    """
    Run a single replication of the full GNAR-edge simulation study.

    This function generates a synthetic network, simulates node- and edge-level
    covariates, simulates edge-weight time series from the specified data-generating
    process, fits a GNAR-edge model with matching covariates, and evaluates
    estimation and inference performance. Performance measures include parameter
    squared errors, design-matrix diagnostics, residual diagnostics, and empirical
    95% confidence-interval coverage.

    Parameters
    ----------
    K : int
        Number of nodes in the simulated network.
    density : float
        Target network density.
    structure : {"ER", "SBM", "RDP"}
        Network generation mechanism.
    rdp_radius_cache: dict
        Dictionary of cached calibrated radii.
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag.
    true_alpha : array-like
        True autoregressive coefficients.
    true_beta_dict : dict
        Dictionary mapping ``(lag, stage)`` pairs to true neighbour coefficients.
    true_gamma_edge_cov : dict
        Dictionary mapping edge-covariate names to true coefficients.
    true_gamma_time_cov : float
        True coefficient for the time-varying exogenous covariate.
    interaction_pairs : list of tuple
        Edge/time covariate interaction terms.
    true_gamma_interaction : dict
        Dictionary of true interaction coefficients.
    T : int
        Length of the simulated time series.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary containing simulation metadata, parameter squared errors,
        design-matrix diagnostics, residual diagnostics, confidence-interval
        coverage indicators, and the fitted learner object.

    Notes
    -----
    This function forms the core simulation routine used throughout the Monte Carlo
    study. One call corresponds to one complete simulated dataset and model fit.
    """ 
    #generating network
    edge_list, _ = generate_network(rdp_radius_cache=rdp_radius_cache, K = K, density = density, structure= structure,  seed = seed)
    E = len(edge_list)

    #simulating edge covariates, only keeping the covariates
    #that are actually part of the true model being simulated
    ages, sexes = simulate_node_covariates(K = K, seed = seed)
    edge_covariates_full = build_synthetic_edge_covariates(edge_list, ages, sexes)
    edge_covariates_sim = {
        name: edge_covariates_full[name] for name in true_gamma_edge_cov.keys()
    }

    #simulating the full model
    X_sim, time_cov_sim, graph = simulate_full_model(
        edge_list, K, L, stages_per_lag,
        true_alpha, true_beta_dict,
        edge_covariates_sim, true_gamma_edge_cov,
        true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
        T=T, seed = seed,
    )

    #simulating exogenous series
    time_index = pd.date_range("2020-01-01", periods=T, freq="D")
    exog_series_sim = {}
    if true_gamma_time_cov != 0.0 or len(interaction_pairs) > 0:
        exog_series_sim = {"sim_temp": pd.Series(time_cov_sim, index=time_index)}

    #simulating the graph and fitting the model to the simulated graph
    sim_graph = ArrayEdgeGraph.from_edge_panel(
        X_sim, edge_list, n_nodes = K, time_labels = time_index
    )

    learner = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph, L=L, stages_per_lag=stages_per_lag,
        train_periods=time_index, use_ols=True,
        edge_covariates=edge_covariates_sim,
        exog_series=exog_series_sim,   #empty dict for regimes 1-5, matching the paper exactly
        interaction_pairs=interaction_pairs,
    )

    learner.fit(use_ols = True, verbose = False)

    #calculating squared errors per parameter type
    alpha_sq_errors = (learner.alpha.flatten() - np.array(true_alpha))**2

    beta_sq_errors = []
    for (l,r), true_val in true_beta_dict.items():
        est_val = learner.beta.get((l,r), np.nan)
        beta_sq_errors.append((est_val - true_val)**2)
    beta_sq_errors = np.array(beta_sq_errors)

    gamma_sq_errors = []
    for name, true_val in true_gamma_edge_cov.items():
        est_val = learner.gamma_.get(name, np.nan)
        gamma_sq_errors.append((est_val - true_val)**2)

    if "sim_temp" in learner.time_cov_names:
        est_temp_gamma = learner.gamma_.get("sim_temp", np.nan)
        gamma_sq_errors.append((est_temp_gamma - true_gamma_time_cov)**2)

    for (edge_name, time_name), true_val in true_gamma_interaction.items():
        est_val = learner.gamma_.get(f"{edge_name}_X_{time_name}", np.nan)
        gamma_sq_errors.append((est_val - true_val)**2)

    #design matrix diagnostics
    D, feature_names = build_design_matrix(learner, learner.train_set)
    rank_deficient = np.linalg.matrix_rank(D) < D.shape[1]
    cond_number = np.linalg.cond(D)

    #calculating independence of mean residuals with exogenous series
    resid_mean = learner.conditional_residuals().mean(axis = 1)
    r_corr, p_corr = pearsonr(time_cov_sim[L:], resid_mean)

    #computing standard errors
    se_theta = compute_ols_standard_errors(learner, D)

    beta_pairs_actual = learner.ols_beta_pairs_
    beta_hat_array = np.array([learner.beta[pair] for pair in beta_pairs_actual])
    beta_true_array = np.array([true_beta_dict.get(pair, np.nan) for pair in beta_pairs_actual])

    
    gamma_time_true = [true_gamma_time_cov] if "sim_temp" in learner.time_cov_names else []

    #concatenating all estimated parameters into theta_hat
    theta_hat = np.concatenate([
        learner.alpha.flatten(),
        np.array(list(learner.beta.values())),
        np.array([learner.gamma_[name] for name in learner.edge_cov_names]),
        np.array([learner.gamma_[name] for name in learner.time_cov_names]),
        np.array([learner.gamma_[f"{e}_X_{t}"] for e, t in learner.interaction_pairs]),
    ])

    #concatenating all true parameters into true_theta
    true_theta = np.concatenate([
        np.array(true_alpha),
        np.array(list(true_beta_dict.values())),
        np.array(list(true_gamma_edge_cov.values())),
        np.array(gamma_time_true),
        np.array(list(true_gamma_interaction.values())),
    ])

    #computing 95% CI coverage
    if np.any(np.isnan(se_theta)):
        covered = np.full(len(theta_hat), np.nan)
    else:
        ci_lower = theta_hat - 1.96 * se_theta
        ci_upper = theta_hat + 1.96 * se_theta
        covered = ((true_theta >= ci_lower) & (true_theta <= ci_upper)).astype(float)

    return {
        "T": T, "E": E, "seed": seed,
        "alpha_sq_errors": alpha_sq_errors,
        "beta_sq_errors": beta_sq_errors,
        "gamma_sq_errors": gamma_sq_errors,
        "rank_deficient": rank_deficient,
        "condition_number": cond_number,
        "resid_exog_correlation": r_corr,
        "resid_exog_pvalue": p_corr,
        "covered": covered,
        "learner": learner
    }

def expand_R_by_l(R_max_list):
    """
    Expand lag-wise maximum neighbour stages into explicit stage lists.

    This helper converts a compact representation such as ``[2, 0, 1]`` into the
    format expected by the GNAR simulation and fitting code, namely
    ``[[1, 2], [], [1]]``. Each entry in the input list corresponds to one lag and
    specifies the maximum neighbour stage to include at that lag.

    Parameters
    ----------
    R_max_list : list of int
        List of maximum neighbour stages by lag.

    Returns
    -------
    list of list of int
        Expanded ``stages_per_lag`` representation, where each lag is represented
        by the list ``[1, ..., r_max]`` if ``r_max > 0`` and by ``[]`` otherwise.
    """
    return [list(range(1, r_max + 1)) if r_max > 0 else [] for r_max in R_max_list]


def build_beta_dict(R_by_l, beta_values):
    """
    Build a dictionary of lag-and-stage neighbour coefficients.

    This helper takes an expanded ``stages_per_lag`` specification and a matching
    collection of coefficient values, then returns a dictionary keyed by
    ``(lag, stage)`` pairs. It is intended for preparing simulation inputs for the
    GNAR-edge process.

    Parameters
    ----------
    R_by_l : list of list of int
        Expanded stage structure by lag, such as ``[[1, 2], [], [1]]``.
    beta_values : sequence
        Coefficient values corresponding to the stages in ``R_by_l``. Each lag may
        provide a scalar or a sequence of values. If a scalar is provided, it is
        reused for all stages in that lag.

    Returns
    -------
    dict
        Dictionary mapping ``(lag, stage)`` to the corresponding beta coefficient.

    Notes
    -----
    The function zips each lag's stage list with the supplied values, so any excess
    values or stages beyond the shorter of the two are ignored.
    """
    beta_dict = {}
    for l, (stages, vals) in enumerate(zip(R_by_l, beta_values), start = 1):
        vals = [vals] if not isinstance(vals, (list,tuple)) else vals
        for r, v in zip(stages, vals):
            beta_dict[(l,r)] = v
    return beta_dict


def build_regime(regime_spec):
    """
    Construct the simulation inputs for a single parameter regime.

    This helper converts a compact regime specification into the collection of
    objects required by ``run_full_simulation_replication``. In particular, it
    expands neighbour-stage specifications and constructs the corresponding beta
    coefficient dictionary.

    Parameters
    ----------
    regime_spec : dict
        Dictionary describing one simulation regime. Expected keys include
        ``"R_max"``, ``"alpha"``, ``"beta_vals"``,
        ``"gamma_edge_cov"``, ``"gamma_time_cov"``,
        ``"interaction_pairs"``, and ``"gamma_interaction"``.

    Returns
    -------
    tuple
        Tuple containing

        - ``R_by_l``
        - ``alpha``
        - ``beta_dict``
        - ``gamma_edge_cov``
        - ``gamma_time_cov``
        - ``interaction_pairs``
        - ``gamma_interaction``

    ready for use by the simulation functions.
    """
    R_by_l = expand_R_by_l(regime_spec["R_max"])
    beta_dict = build_beta_dict(R_by_l, regime_spec["beta_vals"])

    return(
        R_by_l,
        regime_spec["alpha"],
        beta_dict,
        regime_spec["gamma_edge_cov"],
        regime_spec["gamma_time_cov"],
        regime_spec["interaction_pairs"],
        regime_spec["gamma_interaction"],
    )

def format_as_multiindex(table):
    table = table.set_index("parameter")
    cols = pd.MultiIndex.from_tuples([
        (col.split("_")[0], "Coverage" if "coverage" in col else "RMSE")
        for col in table.columns
    ])
    table.columns = cols
    return table[["ER", "SBM", "RDP"]] 


def format_tuple_list(x):
    """
    Convert a list into a parenthesized string representation.

    Lists are formatted as comma-separated values enclosed in parentheses.
    Objects of any other type are simply converted using ``str()``.

    Parameters
    ----------
    x : object
        Object to format.

    Returns
    -------
    str
        Formatted string representation of the input.
    """   
    if isinstance(x, list):
        return "(" + ", ".join(str(v) for v in x) + ")"
    return str(x)

def build_per_parameter_table(all_results, regime_id, paper_regimes, T, structures=("ER", "SBM", "RDP"),
                                exclude_extreme_condition=None):
    """
    Summarize parameter estimation performance across simulation replications.

    This function aggregates simulation results for a specified parameter regime
    and sample size, producing one row per model parameter. For each network
    structure, it reports the empirical root mean squared error (RMSE) and
    95% confidence-interval coverage.

    Parameters
    ----------
    all_results : list of dict
        Collection of simulation outputs from multiple replications.
    regime_id : int or str
        Identifier of the parameter regime to summarize.
    paper_regimes : dict
        Dictionary containing all simulation regime specifications.
    T : int
        Time-series length to summarize.
    structures : tuple of str, default=("ER", "SBM", "RDP")
        Network structures to include in the summary table.
    exclude_extreme_condition : float, optional
        If supplied, replications with condition numbers greater than this value
        are excluded before summary statistics are computed.

    Returns
    -------
    pandas.DataFrame
        Data frame containing one row per parameter and RMSE and coverage columns
        for each requested network structure.

    Raises
    ------
    ValueError
        If the constructed parameter labels do not match the number of estimated
        parameter error arrays.
    """
    subset = [r for r in all_results if r["regime"] == regime_id and r["T"] == T]

    # including option to exclude results with very high condition numbers so they dont skew results
    if exclude_extreme_condition is not None:
        n_before = len(subset)
        subset = [r for r in subset if r["condition_number"] < exclude_extreme_condition]
        n_after = len(subset)
        print(f"Excluded {n_before - n_after} replications with condition_number >= {exclude_extreme_condition}")

    #calculating the number of each parameter type
    n_alpha = len(subset[0]["alpha_sq_errors"])
    n_beta = len(subset[0]["beta_sq_errors"])
    n_gamma = len(subset[0]["gamma_sq_errors"])

    #choosing the parameter specification from the list of parameter regimes
    regime_spec = paper_regimes[regime_id]

    #converting parameter specification into format expected by simulation functions
    R_by_l = expand_R_by_l(regime_spec["R_max"])
    beta_labels = [(l, r) for l, stages in enumerate(R_by_l, start=1) for r in stages]

    #gamma labels, in the same order run_full_simulation_replication builds gamma_sq_errors:
    #edge covariates first, then time covariate, then interaction terms
    gamma_labels = (
        list(regime_spec["gamma_edge_cov"].keys())
        + (["sim_temp"] if regime_spec.get("gamma_time_cov", 0.0) != 0.0 or len(regime_spec.get("interaction_pairs", [])) > 0 else [])
        + [f"{e}_X_{t}" for e, t in regime_spec.get("interaction_pairs", [])]
    )

    param_names = (
        [f"alpha{l}" for l in range(1, n_alpha + 1)]
        + [f"beta{l},{r}" for l, r in beta_labels]
        + [f"gamma_{name}" for name in gamma_labels]
    )

    # ensuring that param_names length matches total error array lengths
    if len(param_names) != n_alpha + n_beta + n_gamma:
        raise ValueError(
            f"param_names length ({len(param_names)}) doesn't match "
            f"n_alpha+n_beta+n_gamma ({n_alpha+n_beta+n_gamma}) -- check gamma_labels construction "
            f"for regime {regime_id}."
        )

    
    rows = []
    #looping over each parameter
    for i, name in enumerate(param_names):
        row = {"parameter": name}
        #looping over each network structure
        for structure in structures:
            struct_subset = [r for r in subset if r["structure"] == structure]

            if i < n_alpha:
                sq_errors = [r["alpha_sq_errors"][i] for r in struct_subset]
            elif i < n_alpha + n_beta:
                beta_idx = i - n_alpha
                sq_errors = [r["beta_sq_errors"][beta_idx] for r in struct_subset]
            else:
                gamma_idx = i - n_alpha - n_beta
                sq_errors = [r["gamma_sq_errors"][gamma_idx] for r in struct_subset]

            covered = [r["covered"][i] for r in struct_subset]

            rmse = np.sqrt(np.nanmean(sq_errors))
            coverage = np.nanmean(covered)

            row[f"{structure}_coverage"] = coverage
            row[f"{structure}_rmse"] = rmse

        rows.append(row)

    return pd.DataFrame(rows)

def run_large_network_replication_density(K, density, structure, rdp_radius_cache, L, stages_per_lag,
                                            true_alpha, true_beta_dict, true_gamma_edge_cov,
                                            true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
                                            T, seed):
    """
    Run a single simulation replication for the density-based interaction-network study.

    This function follows the same workflow as
    ``run_full_simulation_replication`` but is intended for the interaction-network
    simulation experiments in which network sparsity is controlled through the
    target density supplied to ``generate_network``. The fitted model is used to
    compute parameter estimates, confidence intervals, residual diagnostics, and
    design-matrix diagnostics.

    Parameters
    ----------
    K : int
        Number of nodes.
    density : float
        Target network density.
    structure : {"ER", "SBM", "RDP"}
        Network generation mechanism.
    rdp_radius_cache: dict
        Dictionary of cached calibrated radii
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag.
    true_alpha : array-like
        True autoregressive coefficients.
    true_beta_dict : dict
        Dictionary of true neighbour coefficients.
    true_gamma_edge_cov : dict
        Dictionary of true edge-covariate coefficients.
    true_gamma_time_cov : float
        True coefficient for the time-varying exogenous covariate.
    interaction_pairs : list of tuple
        Interaction terms included in the fitted model.
    true_gamma_interaction : dict
        Dictionary of true interaction coefficients.
    T : int
        Number of time points.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        A tuple containing

        - ``result_table`` : parameter estimates and confidence intervals.
        - ``learner`` : fitted GNAR-edge learner.
        - ``graph`` : simulated graph.
        - ``diagnostics`` : dictionary containing summary diagnostics such as the
        number of edges, design-matrix condition number, and residual correlation
        p-value.
    """
    #generating network and producing edge list
    edge_list, _ = generate_network(K=K, density=density, structure=structure, rdp_radius_cache= rdp_radius_cache, seed=seed)
    E = len(edge_list)

    #simulating edge covariates
    ages, sexes = simulate_node_covariates(K=K, seed=seed)
    edge_covariates_full = build_synthetic_edge_covariates(edge_list, ages, sexes)
    edge_covariates_sim = {
        name: edge_covariates_full[name] for name in true_gamma_edge_cov.keys()
    }

    #simulating model based on generated edge list and simulated edge covariates
    X_sim, time_cov_sim, graph = simulate_full_model(
        edge_list, K, L, stages_per_lag,
        true_alpha, true_beta_dict,
        edge_covariates_sim, true_gamma_edge_cov,
        true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
        T=T, seed=seed,
    )
    #setting time index
    time_index = pd.date_range("2020-01-01", periods=T, freq="D")
    #storing simulated exogenous series
    exog_series_sim = {}
    if true_gamma_time_cov != 0.0 or len(interaction_pairs) > 0:
        exog_series_sim = {"sim_temp": pd.Series(time_cov_sim, index=time_index)}

    sim_graph = ArrayEdgeGraph.from_edge_panel(
        X_sim, edge_list, n_nodes=K, time_labels=time_index
    )
    #fitting GNAR-edge model to simulated data
    learner = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph, L=L, stages_per_lag=stages_per_lag,
        train_periods=time_index, use_ols=True,
        edge_covariates=edge_covariates_sim,
        exog_series=exog_series_sim,
        interaction_pairs=interaction_pairs,
    )
    learner.fit(use_ols=True, verbose=False)

    #building design matrix and computing standard errors
    D, feature_names = build_design_matrix(learner, learner.train_set)
    se_theta = compute_ols_standard_errors(learner, D)

    beta_pairs_actual = learner.ols_beta_pairs_
    beta_hat_array = np.array([learner.beta[pair] for pair in beta_pairs_actual])
    beta_true_array = np.array([true_beta_dict.get(pair, np.nan) for pair in beta_pairs_actual])

    gamma_time_true = [true_gamma_time_cov] if "sim_temp" in learner.time_cov_names else []

    #concatenating all estimated parameters into theta_hat
    theta_hat = np.concatenate([
        learner.alpha.flatten(),
        beta_hat_array,
        np.array([learner.gamma_[name] for name in learner.edge_cov_names]),
        np.array([learner.gamma_[name] for name in learner.time_cov_names]),
        np.array([learner.gamma_[f"{e}_X_{t}"] for e, t in learner.interaction_pairs]),
    ])

    #concatenating all true parameters into true_theta
    true_theta = np.concatenate([
        np.array(true_alpha),
        beta_true_array,
        np.array(list(true_gamma_edge_cov.values())),
        np.array(gamma_time_true),
        np.array(list(true_gamma_interaction.values())),
    ])

    #determining whether true parameter is inside conf int
    ci_lower = theta_hat - 1.96 * se_theta
    ci_upper = theta_hat + 1.96 * se_theta
    covers_true = (true_theta >= ci_lower) & (true_theta <= ci_upper)

    param_names = (
        [f"alpha{l}" for l in range(1, L+1)]
        + [f"beta{l},{r}" for l, r in beta_pairs_actual]
        + list(true_gamma_edge_cov.keys())
        + (["sim_temp"] if "sim_temp" in learner.time_cov_names and (true_gamma_time_cov != 0.0) else [])
        + [f"{e}_X_{t}" for e, t in interaction_pairs]
    )

    result_table = pd.DataFrame({
        "parameter": param_names,
        "true_value": true_theta,
        "estimated": theta_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "covers_true": covers_true,
    })

    #calculating condition number of design matrix
    cond_number = np.linalg.cond(D)
    #calculating correlation between mean residual and exogenous series
    resid_mean = learner.conditional_residuals().mean(axis=1)
    if "sim_temp" in learner.time_cov_names:
        r_corr, p_corr = pearsonr(time_cov_sim[L:], resid_mean)
    else:
        r_corr, p_corr = np.nan, np.nan

    return result_table, learner, graph, {"E": E, "condition_number": cond_number, "resid_corr_p": p_corr}

def build_aggregated_table4_style(replication_tables, n_replications):
    """
    Aggregate simulation results into a summary similar to Table 4 in the GNAR-edge paper.

    This function combines parameter estimates across multiple simulation
    replications and computes average parameter estimates, average confidence
    interval endpoints, and empirical confidence-interval coverage rates. The
    resulting table summarizes Monte Carlo performance rather than a single fitted model.

    Parameters
    ----------
    replication_tables : list of pandas.DataFrame
        List of parameter-summary tables produced by individual simulation
        replications.
    n_replications : int
        Number of replications included in the aggregation.

    Returns
    -------
    pandas.DataFrame
        Summary table containing the average parameter estimate, averaged
        95% confidence interval, and empirical coverage rate for each model
        parameter.
    """
    stacked = pd.concat(replication_tables, keys=range(n_replications), names=["replication", "row"])
    stacked = stacked.reset_index(level="row", drop=True).reset_index()

    agg = (
        stacked.groupby("parameter")
        .agg(
            Estimated=("estimated", "mean"),
            ci_lower=("ci_lower", "mean"),
            ci_upper=("ci_upper", "mean"),
            coverage_rate=("covers_true", "mean"),
        )
        .reset_index()
    )
    agg["95% CI"] = agg.apply(lambda row: f"({row['ci_lower']:.3f}, {row['ci_upper']:.3f})", axis=1)
    agg["Coverage"] = agg["coverage_rate"].apply(lambda x: f"{x*100:.0f}%")

    return agg[["parameter", "Estimated", "95% CI", "Coverage"]]

def run_single_step_prediction_replication(K, density, structure, rdp_radius_cache, L, stages_per_lag,
                                             true_alpha, true_beta_dict, true_gamma_edge_cov,
                                             true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
                                             T, seed, arima_order="auto"):
    """
    Run one replication of the one-step-ahead prediction simulation study.

    This function simulates a synthetic GNAR-edge process, fits three competing
    forecasting models using the first ``T - 1`` observations, and evaluates their
    one-step-ahead predictive performance on the final observation. The three models
    compared are:

    1. A GNAR-edge model with the specified neighbour structure and covariates.
    2. A GNAR-edge model with no neighbour effects but the same covariates.
    3. Independent ARIMA models fitted to each edge using the
    ``ARIMAEdgeBaseline`` class.

    Prediction accuracy is measured using the root mean squared error (RMSE)
    computed across all edges for the held-out final time point.

    Parameters
    ----------
    K : int
        Number of nodes in the simulated network.
    density : float
        Target network density.
    structure : {"ER", "SBM", "RDP"}
        Network generation mechanism.
    rdp_radius_cache: dict
        Dictionary of cached calibrated radii
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag for the GNAR-edge model.
    true_alpha : array-like
        True autoregressive coefficients used to generate the data.
    true_beta_dict : dict
        Dictionary mapping ``(lag, stage)`` pairs to the true neighbour-effect
        coefficients.
    true_gamma_edge_cov : dict
        Dictionary mapping edge-covariate names to their true coefficients.
    true_gamma_time_cov : float
        True coefficient for the time-varying exogenous covariate.
    interaction_pairs : list of tuple
        Edge/time covariate interaction terms included in the simulated model.
    true_gamma_interaction : dict
        Dictionary of true interaction coefficients.
    T : int
        Length of the simulated time series.
    seed : int
        Random seed used for reproducibility.
    arima_order : tuple or {"auto"}, default="auto"
        Order specification for the ARIMA benchmark. If ``"auto"``, the
        ``ARIMAEdgeBaseline`` class automatically selects the ARIMA order using
        the Akaike Information Criterion (AIC). Otherwise, the supplied
        ``(p, d, q)`` order is used for every edge.

    Returns
    -------
    dict
        Dictionary containing one-step-ahead RMSE values for each competing model,
        with keys:

        - ``"gnar_with_neighbours"``
        - ``"gnar_no_neighbours"``
        - ``"arima"``

    Notes
    -----
    The ARIMA benchmark is implemented using the ``ARIMAEdgeBaseline`` class and
    fits a separate ARIMA model to each edge independently. Unlike the GNAR-edge
    models, the ARIMA benchmark does not use network structure or exogenous
    covariates.
    """
    #generating network and simulating edge covariates
    edge_list, _ = generate_network(K=K, density=density, structure=structure, rdp_radius_cache=rdp_radius_cache, seed=seed)
    E = len(edge_list)

    ages, sexes = simulate_node_covariates(K=K, seed=seed)
    edge_covariates_full = build_synthetic_edge_covariates(edge_list, ages, sexes)
    edge_covariates_sim = {name: edge_covariates_full[name] for name in true_gamma_edge_cov.keys()}

    #simulating full model based on generated network and simulated edge covariates
    X_sim, time_cov_sim, graph = simulate_full_model(
        edge_list, K, L, stages_per_lag,
        true_alpha, true_beta_dict,
        edge_covariates_sim, true_gamma_edge_cov,
        true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
        T=T, seed=seed,
    )

    time_index = pd.date_range("2020-01-01", periods=T, freq="D")
    train_index = time_index[:-1]
    test_target = X_sim[-1, :]

    results = {}

    #model 1: GNAR-edge with neighbours, as specified (may include covariates)
    exog_series_sim = {}
    if true_gamma_time_cov != 0.0 or len(interaction_pairs) > 0:
        exog_series_sim = {"sim_temp": pd.Series(time_cov_sim[:-1], index=train_index)}

    sim_graph = ArrayEdgeGraph.from_edge_panel(X_sim[:-1, :], edge_list, n_nodes=K, time_labels=train_index)
    learner_full = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph, L=L, stages_per_lag=stages_per_lag,
        train_periods=train_index, use_ols=True,
        edge_covariates=edge_covariates_sim, exog_series=exog_series_sim,
        interaction_pairs=interaction_pairs,
    )
    learner_full.fit(use_ols=True, verbose=False)

    if exog_series_sim:
        next_temp = pd.Series([time_cov_sim[-1]], index=[time_index[-1]])
        learner_full.exog_series["sim_temp"] = pd.concat([exog_series_sim["sim_temp"], next_temp])
    pred_full = learner_full.predict_next(periods=train_index)
    results["gnar_with_neighbours"] = np.sqrt(np.mean((pred_full - test_target)**2))

    #model 2: GNAR-edge no neighbours (stages_per_lag all empty), same covariates
    stages_no_neighbour = [[] for _ in range(L)]
    learner_noneigh = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph, L=L, stages_per_lag=stages_no_neighbour,
        train_periods=train_index, use_ols=True,
        edge_covariates=edge_covariates_sim, exog_series=exog_series_sim,
        interaction_pairs=interaction_pairs,
    )
    learner_noneigh.fit(use_ols=True, verbose=False)
    if exog_series_sim:
        learner_noneigh.exog_series["sim_temp"] = pd.concat([exog_series_sim["sim_temp"], next_temp])
    pred_noneigh = learner_noneigh.predict_next(periods=train_index)
    results["gnar_no_neighbours"] = np.sqrt(np.mean((pred_noneigh - test_target)**2))

    #model 3: ARIMA per edge (no network, no covariates), via the toolkit
    arima_learner = ARIMAEdgeBaseline(
        graph=sim_graph, train_periods=train_index,
        order=arima_order, auto_ic="aic",
    )
    arima_learner.fit()
    arima_preds = arima_learner.predict_next()   #shape (E,)
    results["arima"] = np.sqrt(np.mean((arima_preds - test_target)**2))

    return results

def run_single_step_prediction_replication_with_dfm(K, density, structure, L, stages_per_lag,
                                                       true_alpha, true_beta_dict, true_gamma_edge_cov,
                                                       true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
                                                       T, seed, rdp_radius_cache, n_factors=1):
    """
    Run one replication of the one-step-ahead prediction study using a dynamic
    factor model (DFM).

    This function generates a synthetic GNAR-edge time series under the specified
    data-generating process, fits a dynamic factor model to the first ``T - 1``
    observations using principal component analysis (PCA), and evaluates
    one-step-ahead predictive performance on the held-out final observation. The
    prediction accuracy is measured using the root mean squared error (RMSE)
    computed across all edges.

    Parameters
    ----------
    K : int
        Number of nodes in the simulated network.
    density : float
        Target network density.
    structure : {"ER", "SBM", "RDP"}
        Network generation mechanism.
    L : int
        Number of autoregressive lags in the data-generating process.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag.
    true_alpha : array-like
        True autoregressive coefficients used in the data-generating process.
    true_beta_dict : dict
        Dictionary mapping ``(lag, stage)`` pairs to the true neighbour-effect
        coefficients.
    true_gamma_edge_cov : dict
        Dictionary mapping edge-covariate names to their true coefficients.
    true_gamma_time_cov : float
        True coefficient for the time-varying exogenous covariate.
    interaction_pairs : list of tuple
        Edge/time covariate interaction terms included in the simulated model.
    true_gamma_interaction : dict
        Dictionary of true interaction coefficients.
    T : int
        Length of the simulated time series.
    seed : int
        Random seed used for reproducibility.
    rdp_radius_cache : dict
        Cache of calibrated RDP radii used when generating radius-dependent random
        graphs.
    n_factors : int, default=1
        Number of latent factors retained in the dynamic factor model.

    Returns
    -------
    dict
        Dictionary containing the one-step-ahead RMSE of the dynamic factor model,
        with key:

        - ``"dfm"`` : one-step-ahead root mean squared error.

    Notes
    -----
    The function regenerates the synthetic dataset using the supplied random seed,
    ensuring that the DFM is evaluated on exactly the same simulated data as the
    other forecasting models. The dynamic factor model is estimated using PCA with
    AR(1) factor dynamics and AR(1) idiosyncratic components.
    """
    #generating network and simulating edge covariates
    edge_list, _ = generate_network(K=K, density=density, structure=structure, seed=seed,
                                     rdp_radius_cache=rdp_radius_cache)
    E = len(edge_list)

    ages, sexes = simulate_node_covariates(K=K, seed=seed)
    edge_covariates_full = build_synthetic_edge_covariates(edge_list, ages, sexes)
    edge_covariates_sim = {name: edge_covariates_full[name] for name in true_gamma_edge_cov.keys()}

    X_sim, time_cov_sim, graph = simulate_full_model(
        edge_list, K, L, stages_per_lag, true_alpha, true_beta_dict,
        edge_covariates_sim, true_gamma_edge_cov, true_gamma_time_cov,
        interaction_pairs, true_gamma_interaction, T=T, seed=seed,
    )
    #defining training and test periods
    train_data = X_sim[:-1, :]
    test_target = X_sim[-1, :]
    #predicting using dfm and calculating error
    pred_dfm = dfm_pca_one_step(train_data, n_factors=n_factors, factor_order=1,
                                 idiosyncratic="ar1", standardize=True)
    rmse_dfm = np.sqrt(np.mean((pred_dfm - test_target)**2))

    return {"dfm": rmse_dfm}

def run_dgp_fixed_model_comparison_replication(K, density, structure, rdp_radius_cache, L, stages_per_lag,
                                                 true_alpha, true_beta_dict, true_gamma_edge_cov,
                                                 true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
                                                 T, seed):
    """
    Run one replication comparing correctly and incorrectly specified fitted models.

    This function simulates a synthetic dataset from the full GNAR-edge
    data-generating process, including neighbour effects, edge-level covariates,
    time-varying covariates, and interaction effects. Three competing GNAR-edge
    models are then fitted to the same training data and compared using
    one-step-ahead prediction of the held-out final observation:

    1. A correctly specified model including all covariates and interaction terms.
    2. A model including the covariates but omitting the interaction term.
    3. A model omitting all covariates and interaction terms.

    Prediction accuracy is evaluated using the root mean squared error (RMSE)
    computed across all edges for the final time point.

    Parameters
    ----------
    K : int
        Number of nodes in the simulated network.
    density : float
        Target network density.
    structure : {"ER", "SBM", "RDP"}
        Network generation mechanism.
    rdp_radius_cache : dict
        Cache of calibrated RDP radii used when generating radius-dependent
        random graphs.
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag.
    true_alpha : array-like
        True autoregressive coefficients used in the data-generating process.
    true_beta_dict : dict
        Dictionary mapping ``(lag, stage)`` pairs to the true neighbour-effect
        coefficients.
    true_gamma_edge_cov : dict
        Dictionary mapping edge-covariate names to their true coefficients.
    true_gamma_time_cov : float
        True coefficient for the time-varying exogenous covariate.
    interaction_pairs : list of tuple
        Edge/time covariate interaction terms included in the true model.
    true_gamma_interaction : dict
        Dictionary of true interaction coefficients.
    T : int
        Length of the simulated time series.
    seed : int
        Random seed used for reproducibility.

    Returns
    -------
    dict
        Dictionary containing one-step-ahead RMSE values for the three fitted
        models, with keys:

        - ``"correctly_specified"``
        - ``"covariates_no_interaction"``
        - ``"no_covariates_fitted"``

    Notes
    -----
    All three models are fitted to exactly the same simulated dataset. The only
    difference between them is the covariate specification, allowing predictive
    performance to be compared under correct specification, omitted interaction
    effects, and complete omission of covariate information.
    """
    edge_list, _ = generate_network(K=K, density=density, structure=structure, rdp_radius_cache = rdp_radius_cache, seed=seed)
    E = len(edge_list)

    ages, sexes = simulate_node_covariates(K=K, seed=seed)
    edge_covariates_full = build_synthetic_edge_covariates(edge_list, ages, sexes)
    edge_covariates_sim = {name: edge_covariates_full[name] for name in true_gamma_edge_cov.keys()}

    #simulate from the full true dgp, with genuine covariate and interaction effects
    X_sim, time_cov_sim, graph = simulate_full_model(
        edge_list, K, L, stages_per_lag,
        true_alpha, true_beta_dict,
        edge_covariates_sim, true_gamma_edge_cov,
        true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
        T=T, seed=seed,
    )

    time_index = pd.date_range("2020-01-01", periods=T, freq="D")
    train_index = time_index[:-1]
    test_target = X_sim[-1, :]

    results = {}

    #model A: correctly specified (covariates and interaction)
    exog_series_full = {"sim_temp": pd.Series(time_cov_sim[:-1], index=train_index)}
    sim_graph = ArrayEdgeGraph.from_edge_panel(X_sim[:-1, :], edge_list, n_nodes=K, time_labels=train_index)

    learner_A = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph, L=L, stages_per_lag=stages_per_lag,
        train_periods=train_index, use_ols=True,
        edge_covariates=edge_covariates_sim, exog_series=exog_series_full,
        interaction_pairs=interaction_pairs,
    )
    learner_A.fit(use_ols=True, verbose=False)
    next_temp = pd.Series([time_cov_sim[-1]], index=[time_index[-1]])
    learner_A.exog_series["sim_temp"] = pd.concat([exog_series_full["sim_temp"], next_temp])
    pred_A = learner_A.predict_next(periods=train_index)
    results["correctly_specified"] = np.sqrt(np.mean((pred_A - test_target)**2))

    #model B: covariates, but without the interaction 
    learner_B = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph, L=L, stages_per_lag=stages_per_lag,
        train_periods=train_index, use_ols=True,
        edge_covariates=edge_covariates_sim, exog_series=exog_series_full,
        interaction_pairs=[],   #omits the true interaction effect
    )
    learner_B.fit(use_ols=True, verbose=False)
    learner_B.exog_series["sim_temp"] = pd.concat([exog_series_full["sim_temp"], next_temp])
    pred_B = learner_B.predict_next(periods=train_index)
    results["covariates_no_interaction"] = np.sqrt(np.mean((pred_B - test_target)**2))

    #model C: no covariates at all 
    learner_C = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph, L=L, stages_per_lag=stages_per_lag,
        train_periods=train_index, use_ols=True,
        edge_covariates={}, exog_series={},
        interaction_pairs=[],
    )
    learner_C.fit(use_ols=True, verbose=False)
    pred_C = learner_C.predict_next(periods=train_index)
    results["no_covariates_fitted"] = np.sqrt(np.mean((pred_C - test_target)**2))

    return results

def run_single_step_prediction_replication_learner(K, density, structure, rdp_radius_cache, L, stages_per_lag,
                                             true_alpha, true_beta_dict, true_gamma_edge_cov,
                                             true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
                                             T, seed):
    """
    Run one replication of the one-step-ahead prediction simulation study and
    return the fitted GNAR-edge learner.

    This function simulates a synthetic GNAR-edge process, fits three competing
    forecasting models using the first ``T - 1`` observations, and evaluates their
    one-step-ahead predictive performance on the held-out final observation. The
    same models are fitted as in ``run_single_step_prediction_replication``;
    however, instead of returning the prediction RMSEs, this function returns the
    fitted GNAR-edge learner corresponding to the correctly specified model along
    with its exogenous time series. This is useful for inspecting estimated
    parameters, diagnostics, or generating additional predictions after fitting.

    Parameters
    ----------
    K : int
        Number of nodes in the simulated network.
    density : float
        Target network density.
    structure : {"ER", "SBM", "RDP"}
        Network generation mechanism.
    rdp_radius_cache : dict
        Cache of calibrated RDP radii used when generating radius-dependent
        random graphs.
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag for the GNAR-edge model.
    true_alpha : array-like
        True autoregressive coefficients used to generate the data.
    true_beta_dict : dict
        Dictionary mapping ``(lag, stage)`` pairs to the true neighbour-effect
        coefficients.
    true_gamma_edge_cov : dict
        Dictionary mapping edge-covariate names to their true coefficients.
    true_gamma_time_cov : float
        True coefficient for the time-varying exogenous covariate.
    interaction_pairs : list of tuple
        Edge/time covariate interaction terms included in the simulated model.
    true_gamma_interaction : dict
        Dictionary of true interaction coefficients.
    T : int
        Length of the simulated time series.
    seed : int
        Random seed used for reproducibility.

    Returns
    -------
    tuple
        A pair ``(learner, exog_series)`` where

        - ``learner`` is the fitted ``GNAREdgeGlobalMultiCovLearner`` corresponding
        to the correctly specified model.
        - ``exog_series`` is the exogenous time series (including the held-out
        prediction period) used by the learner.

    Notes
    -----
    Although the no-neighbour GNAR-edge model and the autoregressive benchmark are
    also fitted internally, they are used only to mirror the fitting procedure of
    ``run_single_step_prediction_replication``. Their prediction errors are not
    returned.
    """
    #generating network
    edge_list, _ = generate_network(K=K, density=density, structure=structure, rdp_radius_cache = rdp_radius_cache, seed=seed)
    E = len(edge_list)

    #simulating edge covariates
    ages, sexes = simulate_node_covariates(K=K, seed=seed)
    edge_covariates_full = build_synthetic_edge_covariates(edge_list, ages, sexes)
    edge_covariates_sim = {name: edge_covariates_full[name] for name in true_gamma_edge_cov.keys()}

    X_sim, time_cov_sim, graph = simulate_full_model(
        edge_list, K, L, stages_per_lag,
        true_alpha, true_beta_dict,
        edge_covariates_sim, true_gamma_edge_cov,
        true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
        T=T, seed=seed,
    )

    time_index = pd.date_range("2020-01-01", periods=T, freq="D")
    train_index = time_index[:-1]
    test_target = X_sim[-1, :]

    results = {}

    #model 1: GNAR-edge with neighbours, as specified (may include covariates)
    exog_series_sim = {}
    if true_gamma_time_cov != 0.0 or len(interaction_pairs) > 0:
        exog_series_sim = {"sim_temp": pd.Series(time_cov_sim[:-1], index=train_index)}

    sim_graph = ArrayEdgeGraph.from_edge_panel(X_sim[:-1, :], edge_list, n_nodes=K, time_labels=train_index)
    learner_full = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph, L=L, stages_per_lag=stages_per_lag,
        train_periods=train_index, use_ols=True,
        edge_covariates=edge_covariates_sim, exog_series=exog_series_sim,
        interaction_pairs=interaction_pairs,
    )
    learner_full.fit(use_ols=True, verbose=False)

    #for prediction, need next-period covariate value if time-varying covariates are used
    if exog_series_sim:
        next_temp = pd.Series([time_cov_sim[-1]], index=[time_index[-1]])
        learner_full.exog_series["sim_temp"] = pd.concat([exog_series_sim["sim_temp"], next_temp])
    pred_full = learner_full.predict_next(periods=train_index)
    results["gnar_with_neighbours"] = np.sqrt(np.mean((pred_full - test_target)**2))

    #model 2: GNAR-edge no neighbours (stages_per_lag all empty), same covariates
    stages_no_neighbour = [[] for _ in range(L)]
    learner_noneigh = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph, L=L, stages_per_lag=stages_no_neighbour,
        train_periods=train_index, use_ols=True,
        edge_covariates=edge_covariates_sim, exog_series=exog_series_sim,
        interaction_pairs=interaction_pairs,
    )
    learner_noneigh.fit(use_ols=True, verbose=False)
    if exog_series_sim:
        learner_noneigh.exog_series["sim_temp"] = pd.concat([exog_series_sim["sim_temp"], next_temp])
    pred_noneigh = learner_noneigh.predict_next(periods=train_index)
    results["gnar_no_neighbours"] = np.sqrt(np.mean((pred_noneigh - test_target)**2))

    #model 3: AR per edge (no network, no covariates)
    ar_preds = np.zeros(E)
    for e in range(E):
        series = X_sim[:-1, e]
        try:
            fitted = AutoReg(series, lags=L, old_names=False).fit()
            ar_preds[e] = fitted.predict(start=len(series), end=len(series))[0]
        except Exception:
            ar_preds[e] = series[-1]
    results["ar"] = np.sqrt(np.mean((ar_preds - test_target)**2))

    return learner_full, learner_full.exog_series["sim_temp"]


def simulate_full_model_misspec(
    edge_list, K, L, stages_per_lag,
    true_alpha, true_beta_dict, edge_covariates, true_gamma_edge_cov,
    true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
    T, noise_std=1.0, noise_dist="normal", t_df=3, correlation=0.0,
    time_cov_name="sim_temp", seed=None
):
    """
    Simulate edge-weight time series under alternative innovation distributions.

    This function extends ``simulate_full_model`` for robustness and model
    misspecification experiments. The autoregressive, network, edge-covariate,
    time-varying covariate, and interaction components of the data-generating
    process are unchanged, while the innovation distribution can depart from
    the standard independent Gaussian specification.

    Innovations may be independent Gaussian, heavy-tailed Student's t, or
    correlated Gaussian. For correlated innovations, edges are randomly
    partitioned into disjoint pairs and the specified correlation is imposed
    between the innovations of the two edges within each pair. Innovations
    belonging to different pairs are uncorrelated.

    Parameters
    ----------
    edge_list : list of tuple of int
        Edges of the network, represented as ``(i, j)`` node-index pairs.
    K : int
        Number of nodes in the network.
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag.
    true_alpha : list of float
        True autoregressive coefficients, one for each lag.
    true_beta_dict : dict of {(int, int): float}
        True neighbour-effect coefficients, indexed by ``(lag, stage)``.
    edge_covariates : dict of {str: np.ndarray}
        Mapping from edge-covariate names to arrays of length ``E``.
    true_gamma_edge_cov : dict of {str: float}
        True coefficients associated with the edge-level covariates.
    true_gamma_time_cov : float
        True coefficient associated with the time-varying exogenous covariate.
    interaction_pairs : list of tuple of str
        Pairs specifying interactions between edge-level and time-varying
        covariates.
    true_gamma_interaction : dict of {(str, str): float}
        True coefficients associated with the interaction terms.
    T : int
        Length of the simulated time series.
    noise_std : float, default=1.0
        Marginal standard deviation of the innovations. Student's t
        innovations are rescaled to have variance ``noise_std**2``.
    noise_dist : {"normal", "t", "correlated"}, default="normal"
        Innovation distribution. ``"normal"`` produces independent Gaussian
        innovations, ``"t"`` produces independent variance-matched Student's
        t innovations, and ``"correlated"`` produces Gaussian innovations
        with the specified within-pair correlation.
    t_df : int, default=3
        Degrees of freedom of the Student's t innovations. Used only when
        ``noise_dist="t"``.
    correlation : float, default=0.0
        Correlation imposed between innovations of edges belonging to the
        same randomly assigned pair when ``noise_dist="correlated"``.
        Correlation between different pairs is zero.
    time_cov_name : str, default="sim_temp"
        Name assigned to the simulated time-varying exogenous series.
    seed : int, optional
        Random seed controlling simulation and the random assignment of edges
        to correlation pairs.

    Returns
    -------
    X : np.ndarray, shape (T, E)
        Simulated edge-weight time-series panel.
    time_cov : np.ndarray, shape (T,)
        Simulated time-varying exogenous series.
    graph : object
        Underlying network object returned by ``build_W_matrices``.

    Notes
    -----
    For ``noise_dist="t"``, the Student's t innovations are rescaled so that
    their marginal variance matches that of the Gaussian baseline. For
    ``noise_dist="correlated"``, the marginal variance of each innovation also
    remains ``noise_std**2``; only the contemporaneous dependence structure
    across paired edges is changed.
    """
    rng = np.random.default_rng(seed)
    E = len(edge_list)

    needed_r = sorted({r for rs in stages_per_lag for r in rs})
    W_dict, graph = build_W_matrices(edge_list, K, needed_r)

    time_cov = simulate_ar1_exog(T, phi=0.7, mean=30, sd=3, seed=seed)

    #precompute correlated-noise covariance once, if needed
    if noise_dist == "correlated":
        cov_matrix = build_block_correlated_covariance(E, correlation, noise_std, block_size=2, seed=seed)

    X = np.zeros((T, E))
    X[:L, :] = rng.normal(0, noise_std, size=(L, E))

    #simulating remaining parts of the model (autoregressive, network and exogenous terms)
    for t in range(L, T):
        pred = np.zeros(E)
        for l in range(1, L + 1):
            pred += true_alpha[l-1] * X[t-l, :]
            for r in stages_per_lag[l-1]:
                pred += true_beta_dict[(l, r)] * (W_dict[r] @ X[t-l, :])

        for name, gamma_val in true_gamma_edge_cov.items():
            pred += gamma_val * edge_covariates[name]

        pred += true_gamma_time_cov * time_cov[t]

        for (edge_name, cov_name) in interaction_pairs:
            gamma_val = true_gamma_interaction[(edge_name, cov_name)]
            pred += gamma_val * edge_covariates[edge_name] * time_cov[t]

        #noise generation, by distribution
        if noise_dist == "normal":
            noise = rng.normal(0, noise_std, size=E)
        elif noise_dist == "t":
            scale = noise_std / np.sqrt(t_df / (t_df - 2))
            noise = rng.standard_t(t_df, size=E) * scale
        elif noise_dist == "correlated":
            noise = rng.multivariate_normal(np.zeros(E), cov_matrix)
        else:
            raise ValueError("noise_dist must be 'normal', 't', or 'correlated'")

        X[t, :] = pred + noise

    return X, time_cov, graph

def build_block_correlated_covariance(E, correlation, noise_std=1.0, block_size=2, seed=None):
    """
    Construct a block-correlated covariance matrix for edge innovations.

    The edges are randomly partitioned into disjoint blocks of size
    ``block_size``. Innovations for edges within the same block have the
    specified pairwise correlation, while innovations belonging to different
    blocks are uncorrelated. This construction permits both positive and
    negative within-block correlations without imposing a common correlation
    across the full edge system.

    Parameters
    ----------
    E : int
        Number of edge-level innovation series.
    correlation : float
        Pairwise correlation imposed between innovations within each block.
        For the default ``block_size=2``, values between -1 and 1 produce a
        valid within-block correlation matrix.
    noise_std : float, default=1.0
        Marginal standard deviation of each edge innovation.
    block_size : int, default=2
        Number of edges assigned to each correlation block.
    seed : int, optional
        Random seed controlling the permutation used to assign edges to
        blocks.

    Returns
    -------
    np.ndarray
        An ``(E, E)`` covariance matrix with marginal variance
        ``noise_std**2``, the specified correlation within blocks, and zero
        covariance between blocks.

    Notes
    -----
    If ``E`` is not divisible by ``block_size``, any remaining edges are
    independent of all other edges. With the default ``block_size=2``, the
    construction pairs edges randomly and allows relatively strong positive
    or negative correlations within each pair.
    """
    rng = np.random.default_rng(seed)
    cov_matrix = np.eye(E) * noise_std**2

    edge_order = rng.permutation(E)
    for i in range(0, E - block_size + 1, block_size):
        block_edges = edge_order[i:i+block_size]
        for a in block_edges:
            for b in block_edges:
                if a != b:
                    cov_matrix[a, b] = correlation * noise_std**2

    return cov_matrix

def run_misspecification_replication(K, density, structure, rdp_radius_cache, L, stages_per_lag,
                                      true_alpha, true_beta_dict, true_gamma_edge_cov,
                                      true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
                                      T, seed, noise_dist="normal", t_df=3, correlation=0.0,
                                      rewiring_prob=0.0):
    """
    Run one replication of the model misspecification simulation study.

    This function simulates a GNAR-edge process under the specified data-generating
    process (DGP), optionally introducing heavy-tailed innovations, correlated
    errors, and network misspecification through graph rewiring. A GNAR-edge model
    is then fitted to the simulated data, and parameter recovery is assessed using
    squared estimation errors for the autoregressive, neighbour, and covariate
    coefficients.

    Three sources of model misspecification can be investigated:

    - Non-Gaussian innovations via Student's t-distributed noise.
    - Correlated innovations across edges.
    - Structural misspecification by fitting the model on a rewired network rather
    than the true network used for simulation.

    Parameters
    ----------
    K : int
        Number of nodes in the simulated network.
    density : float
        Target network density.
    structure : {"ER", "SBM", "RDP"}
        Network generation mechanism.
    rdp_radius_cache : dict
        Cache of calibrated RDP radii used when generating radius-dependent random
        graphs.
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag.
    true_alpha : array-like
        True autoregressive coefficients used in the data-generating process.
    true_beta_dict : dict
        Dictionary mapping ``(lag, stage)`` pairs to the true neighbour-effect
        coefficients.
    true_gamma_edge_cov : dict
        Dictionary mapping edge-covariate names to their true coefficients.
    true_gamma_time_cov : float
        True coefficient for the time-varying exogenous covariate.
    interaction_pairs : list of tuple
        Edge/time covariate interaction terms included in the simulated model.
    true_gamma_interaction : dict
        Dictionary of true interaction coefficients.
    T : int
        Length of the simulated time series.
    seed : int
        Random seed used for reproducibility.
    noise_dist : {"normal", "t", "correlated"}, default="normal"
        Distribution used to generate the innovation process. Correlated
        innovations are Gaussian with dependence imposed between randomly
        paired edges.
    t_df : int, default=3
        Degrees of freedom for the Student's t innovations when
        ``noise_dist="t"``.
    correlation : float, default=0.0
        Within-pair correlation of the edge innovations when
        ``noise_dist="correlated"``. Edges are randomly assigned to disjoint
        pairs, with zero innovation correlation between different pairs.
    rewiring_prob : float, default=0.0
        Probability of rewiring each edge before fitting the model. A value of
        zero fits the model using the true network structure.

    Returns
    -------
    dict
        Dictionary containing

        - ``learner`` : fitted GNAR-edge learner.
        - ``T`` : simulated time-series length.
        - ``seed`` : random seed.
        - ``noise_dist`` : innovation distribution.
        - ``t_df`` : degrees of freedom for t-distributed innovations.
        - ``correlation`` : innovation correlation.
        - ``rewiring_prob`` : network rewiring probability.
        - ``hamming_distance`` : Hamming distance between the true and fitted
        network structures.
        - ``alpha_sq_errors`` : squared estimation errors for the autoregressive
        coefficients.
        - ``beta_sq_errors`` : squared estimation errors for the neighbour
        coefficients.
        - ``beta_sq_by_pair`` : squared neighbour-coefficient errors indexed by
        ``(lag, stage)``.
        - ``gamma_sq_errors`` : squared estimation errors for the covariate and
        interaction coefficients.

    """
    #generating network and simulating edge covariates
    true_edge_list, _ = generate_network(K=K, density=density, structure=structure, rdp_radius_cache = rdp_radius_cache, seed=seed)
    ages, sexes = simulate_node_covariates(K=K, seed=seed)
    edge_covariates_full = build_synthetic_edge_covariates(true_edge_list, ages, sexes)
    edge_covariates_sim = {name: edge_covariates_full[name] for name in true_gamma_edge_cov.keys()}

    #simulating full model (misspecified)
    X_sim, time_cov_sim, _ = simulate_full_model_misspec(
        true_edge_list, K, L, stages_per_lag, true_alpha, true_beta_dict,
        edge_covariates_sim, true_gamma_edge_cov, true_gamma_time_cov,
        interaction_pairs, true_gamma_interaction, T=T, seed=seed,
        noise_dist=noise_dist, t_df=t_df, correlation=correlation,
    )

    def compute_hamming_distance(true_edge_list, rewired_edge_list, K):
        """
        Standard graph Hamming distance: proportion of mismatched edge states
        among all possible undirected node pairs.
        """
        true_set = set(true_edge_list)
        rewired_set = set(rewired_edge_list)

        total_possible_pairs = K * (K - 1) / 2

        mismatches = len(true_set.symmetric_difference(rewired_set))

        return mismatches / total_possible_pairs

    #rewiring (if requested): fit on a perturbed version of the network
    if rewiring_prob > 0:
        fit_edge_list = rewire_edges(true_edge_list, K, rewiring_prob, seed=seed + 100000)
        edge_covariates_fit_full = build_synthetic_edge_covariates(fit_edge_list, ages, sexes)
        edge_covariates_fit = {name: edge_covariates_fit_full[name] for name in true_gamma_edge_cov.keys()}
        hamming_distance = compute_hamming_distance(true_edge_list, fit_edge_list, K)   #UPDATED
    else:
        fit_edge_list = true_edge_list
        edge_covariates_fit = edge_covariates_sim
        hamming_distance = 0.0

    time_index = pd.date_range("2020-01-01", periods=T, freq="D")
    exog_series_sim = {}
    if true_gamma_time_cov != 0.0 or len(interaction_pairs) > 0:
        exog_series_sim = {"sim_temp": pd.Series(time_cov_sim, index=time_index)}

    sim_graph = ArrayEdgeGraph.from_edge_panel(X_sim, fit_edge_list, n_nodes=K, time_labels=time_index)

    #fitting a GNAR-edge learner to the simulated data
    learner = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph, L=L, stages_per_lag=stages_per_lag,
        train_periods=time_index, use_ols=True,
        edge_covariates=edge_covariates_fit, exog_series=exog_series_sim,
        interaction_pairs=interaction_pairs,
    )
    learner.fit(use_ols=True, verbose=False)

    #parameter recovery: alpha, beta, gamma RMSE
    alpha_sq_errors = (learner.alpha.flatten() - np.array(true_alpha))**2

    beta_pairs_actual = learner.ols_beta_pairs_
    beta_sq_errors = np.array([
        (learner.beta[pair] - true_beta_dict.get(pair, np.nan))**2 for pair in beta_pairs_actual
    ])
    beta_sq_by_pair = {
        pair: (learner.beta[pair] - true_beta_dict.get(pair, np.nan))**2 for pair in beta_pairs_actual
    }

    gamma_sq_errors = []
    for name, true_val in true_gamma_edge_cov.items():
        est_val = learner.gamma_.get(name, np.nan)
        gamma_sq_errors.append((est_val - true_val)**2)
    if "sim_temp" in learner.time_cov_names:
        est_temp_gamma = learner.gamma_.get("sim_temp", np.nan)
        gamma_sq_errors.append((est_temp_gamma - true_gamma_time_cov)**2)
    for (edge_name, time_name), true_val in true_gamma_interaction.items():
        est_val = learner.gamma_.get(f"{edge_name}_X_{time_name}", np.nan)
        gamma_sq_errors.append((est_val - true_val)**2)
    gamma_sq_errors = np.array(gamma_sq_errors)

    return {"learner": learner,
        "T": T, "seed": seed,
        "noise_dist": noise_dist, "t_df": t_df, "correlation": correlation,
        "rewiring_prob": rewiring_prob, "hamming_distance": hamming_distance,
        "alpha_sq_errors": alpha_sq_errors,
        "beta_sq_errors": beta_sq_errors,
        "beta_sq_by_pair": beta_sq_by_pair,
        "gamma_sq_errors": gamma_sq_errors,
        "learner": learner,
    }

def rewire_edges(edge_list, K, rewiring_prob, seed=None):
    """
    Randomly rewire edges in an undirected network.

    This function perturbs an existing edge list by independently considering each
    edge for rewiring with probability ``rewiring_prob``. When an edge is selected,
    one of its two endpoints is chosen uniformly at random and replaced with a
    randomly selected node, preserving the other endpoint. The resulting edge is
    stored in canonical order ``(min(u, v), max(u, v))`` so that the graph remains
    undirected.

    Parameters
    ----------
    edge_list : list of tuple
        List of undirected edges represented as node-index pairs ``(u, v)``.
    K : int
        Number of nodes in the network.
    rewiring_prob : float
        Probability that an individual edge is rewired. A value of 0 leaves the
        network unchanged, while a value of 1 attempts to rewire every edge.
    seed : int, optional
        Random seed used for reproducibility.

    Returns
    -------
    list of tuple
        Rewired edge list containing the same number of edges as the input.

    Notes
    -----
    This rewiring procedure preserves the number of edges but does not guarantee
    that duplicate edges or self-loops cannot arise if multiple rewired edges map
    to the same node pair. In the current implementation, self-loops are avoided by
    resampling the replaced endpoint until it differs from the fixed endpoint.
    """
    rng = np.random.default_rng(seed)
    rewired = []
    #looping over edges
    for (i, j) in edge_list:
        #rewire edge with probability rewiring_prob
        if rng.random() < rewiring_prob:
            if rng.random() < 0.5:
                new_i = rng.integers(0, K)
                #avoiding self loops
                while new_i == j:
                    new_i = rng.integers(0, K)
                rewired.append((min(new_i, j), max(new_i, j)))
            else:
                new_j = rng.integers(0, K)
                #avoiding self loops
                while new_j == i:
                    new_j = rng.integers(0, K)
                rewired.append((min(i, new_j), max(i, new_j)))
        else:
            rewired.append((i, j))
    return rewired

def simulate_full_model_exog_correlated(edge_list, K, L, stages_per_lag,
                                          true_alpha, true_beta_dict, edge_covariates, true_gamma_edge_cov,
                                          true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
                                          T, noise_std=1.0, exog_u_correlation=0.0, time_cov_name="sim_temp",
                                          seed=None):
    """
        Simulate a GNAR-edge process with endogenous exogenous covariates.
    
        This function generates a multivariate GNAR-edge time series in which the
        time-varying exogenous covariate and the innovation process share a common
        random component. Consequently, the exogenous variable is correlated with
        the model innovations, violating the exogeneity assumption required for
        unbiased ordinary least squares estimation.
    
        The simulated process includes autoregressive effects, neighbour effects,
        edge-level covariate effects, a time-varying exogenous covariate, and
        optional interaction terms.
    
        Parameters
        ----------
        edge_list : list of tuple
            Undirected edge list represented as node-index pairs ``(u, v)``.
        K : int
            Number of nodes in the network.
        L : int
            Number of autoregressive lags.
        stages_per_lag : list of list of int
            Neighbour stages included at each lag.
        true_alpha : array-like
            True autoregressive coefficients.
        true_beta_dict : dict
            Dictionary mapping ``(lag, stage)`` pairs to the true neighbour-effect
            coefficients.
        edge_covariates : dict
            Dictionary of edge-level covariate arrays.
        true_gamma_edge_cov : dict
            Dictionary mapping edge-covariate names to their true coefficients.
        true_gamma_time_cov : float
            True coefficient for the time-varying exogenous covariate.
        interaction_pairs : list of tuple
            Edge/time covariate interaction terms included in the simulated model.
        true_gamma_interaction : dict
            Dictionary of true interaction coefficients.
        T : int
            Length of the simulated time series.
        noise_std : float, default=1.0
            Standard deviation of the innovation process.
        exog_u_correlation : float, default 0.0
            Target correlation between z_t's innovations and the shared component of
            u_t. Must lie in [-1, 1]. Supports negative values: the magnitude of the
            correlation determines the weight given to the shared random component
            (via sqrt(|rho|)), and the sign is applied separately by flipping the
            shared component's contribution, avoiding sqrt of a negative number.
        time_cov_name : str, default "sim_temp"
            Name assigned to the simulated time-varying exogenous series.
        seed : int, optional
            Random seed used for reproducibility.
    
        Returns
        -------
        tuple
            A triple ``(X, time_cov, graph)`` where
    
            - ``X`` is the simulated edge-weight panel of shape ``(T, E)``.
            - ``time_cov`` is the simulated endogenous exogenous time series.
            - ``graph`` is the corresponding ``ArrayEdgeGraph``.
    
        Notes
        -----
        Endogeneity is introduced by allowing the innovation driving the exogenous
        AR(1) process and the edge innovation vector to share a common Gaussian
        shock. The parameter ``exog_u_correlation`` controls the strength of this
        dependence while preserving the marginal variances of both processes.
        """
    if not -1.0 <= exog_u_correlation <= 1.0:
        raise ValueError(f"exog_u_correlation must lie in [-1, 1], got {exog_u_correlation}.")

    rng = np.random.default_rng(seed)
    E = len(edge_list)

    needed_r = sorted({r for rs in stages_per_lag for r in rs})
    W_dict, graph = build_W_matrices(edge_list, K, needed_r)

    #split correlation into sign and magnitude, so sqrt is always applied to a
    #non-negative value, and the sign is applied separately to the shared component
    rho = exog_u_correlation
    sign = np.sign(rho) if rho != 0 else 1.0
    abs_rho = abs(rho)

    #first, draw the shared random component used to correlate z_t with u_t
    shared_innovations = rng.normal(0, 1, size=T)

    #build z_t as an AR(1) process, with its innovation partly driven by shared_innovations
    z_phi, z_mean, z_sd = 0.7, 30, 3
    z_innovation_sd = z_sd * np.sqrt(1 - z_phi**2)
    time_cov = np.zeros(T)
    time_cov[0] = rng.normal(z_mean, z_sd)
    for t in range(1, T):
        own_innov = rng.normal(0, z_innovation_sd)
        combined_innov = (
            sign * np.sqrt(abs_rho) * shared_innovations[t] * z_innovation_sd
            + np.sqrt(1 - abs_rho) * own_innov
        )
        time_cov[t] = z_mean + z_phi * (time_cov[t-1] - z_mean) + combined_innov

    X = np.zeros((T, E))
    X[:L, :] = rng.normal(0, noise_std, size=(L, E))

    for t in range(L, T):
        pred = np.zeros(E)
        for l in range(1, L + 1):
            pred += true_alpha[l-1] * X[t-l, :]
            for r in stages_per_lag[l-1]:
                pred += true_beta_dict[(l, r)] * (W_dict[r] @ X[t-l, :])

        for name, gamma_val in true_gamma_edge_cov.items():
            pred += gamma_val * edge_covariates[name]

        pred += true_gamma_time_cov * time_cov[t]

        for (edge_name, cov_name) in interaction_pairs:
            gamma_val = true_gamma_interaction[(edge_name, cov_name)]
            pred += gamma_val * edge_covariates[edge_name] * time_cov[t]

        #u_t's mean-across-edges component shares the same shared_innovations[t] used for z_t
        #individual edges get their own idiosyncratic noise on top of this shared component
        shared_component = sign * np.sqrt(abs_rho) * shared_innovations[t] * noise_std
        idiosyncratic = rng.normal(0, noise_std * np.sqrt(1 - abs_rho), size=E)
        u_t = shared_component + idiosyncratic

        X[t, :] = pred + u_t

    return X, time_cov, graph


def run_exog_correlation_replication(
    K,
    density,
    structure,
    rdp_radius_cache,
    L,
    stages_per_lag,
    true_alpha,
    true_beta_dict,
    true_gamma_edge_cov,
    true_gamma_time_cov,
    interaction_pairs,
    true_gamma_interaction,
    T,
    seed,
    exog_u_correlation=0.0,
):
    """
    Run one replication of the exogenous-endogeneity simulation study.

    This function simulates a GNAR-edge process in which the time-varying
    exogenous covariate is correlated with the innovation process, violating
    the standard exogeneity assumption. A GNAR-edge model is then fitted under
    the usual assumption of exogeneity, and parameter recovery is evaluated by
    comparing the estimated model parameters with their true values.

    Parameters
    ----------
    K : int
        Number of nodes in the simulated network.
    density : float
        Target network density.
    structure : {"ER", "SBM", "RDP"}
        Network generation mechanism.
    rdp_radius_cache : dict
        Cache of calibrated RDP radii used when generating radius-dependent
        random graphs.
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag.
    true_alpha : array-like
        True autoregressive coefficients used in the data-generating process.
    true_beta_dict : dict
        Dictionary mapping ``(lag, stage)`` pairs to the true neighbour-effect
        coefficients.
    true_gamma_edge_cov : dict
        Dictionary mapping edge-covariate names to their true coefficients.
    true_gamma_time_cov : float
        True coefficient for the time-varying exogenous covariate.
    interaction_pairs : list of tuple
        Edge/time covariate interaction terms included in the simulated model.
    true_gamma_interaction : dict
        Dictionary of true interaction coefficients.
    T : int
        Length of the simulated time series.
    seed : int
        Random seed used for reproducibility.
    exog_u_correlation : float, default=0.0
        Correlation between the innovation driving the exogenous AR(1) process
        and the edge innovation process. A value of zero corresponds to an
        exogenous covariate, while larger values introduce increasing
        endogeneity.

    Returns
    -------
    dict
        Dictionary containing

        - ``learner`` : fitted GNAR-edge learner.
        - ``T`` : simulated time-series length.
        - ``seed`` : random seed.
        - ``exog_u_correlation`` : imposed correlation between the exogenous
          covariate and the innovation process.
        - ``alpha_sq_errors`` : squared estimation errors for the
          autoregressive coefficients.
        - ``beta_sq_errors`` : squared estimation errors for the neighbour
          coefficients.
        - ``gamma_sq_errors`` : squared estimation errors for the covariate
          and interaction coefficients.

    Notes
    -----
    The fitted model assumes that the exogenous covariate is independent of the
    innovation process, even when this assumption is violated during
    simulation. The simulation therefore quantifies the effect of endogenous
    regressors on parameter estimation.
    """
    #generating network
    edge_list, _ = generate_network(
        K=K,
        density=density,
        structure=structure,
        rdp_radius_cache=rdp_radius_cache,
        seed=seed,
    )
    #simulating edge covariates
    ages, sexes = simulate_node_covariates(K=K, seed=seed)
    edge_covariates_full = build_synthetic_edge_covariates(edge_list, ages, sexes)
    edge_covariates_sim = {
        name: edge_covariates_full[name]
        for name in true_gamma_edge_cov.keys()
    }

    #simulating full model with exogenous correlated innovations
    X_sim, time_cov_sim, _ = simulate_full_model_exog_correlated(
        edge_list,
        K,
        L,
        stages_per_lag,
        true_alpha,
        true_beta_dict,
        edge_covariates_sim,
        true_gamma_edge_cov,
        true_gamma_time_cov,
        interaction_pairs,
        true_gamma_interaction,
        T=T,
        seed=seed,
        exog_u_correlation=exog_u_correlation,
    )

    time_index = pd.date_range("2020-01-01", periods=T, freq="D")
    exog_series_sim = {"sim_temp": pd.Series(time_cov_sim, index=time_index)}

    sim_graph = ArrayEdgeGraph.from_edge_panel(
        X_sim,
        edge_list,
        n_nodes=K,
        time_labels=time_index,
    )

    #fitting GNAR-edge learner to the simulated data 
    learner = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph,
        L=L,
        stages_per_lag=stages_per_lag,
        train_periods=time_index,
        use_ols=True,
        edge_covariates=edge_covariates_sim,
        exog_series=exog_series_sim,
        interaction_pairs=interaction_pairs,
    )

    learner.fit(use_ols=True, verbose=False)

    #calculating squared errors for each coefficient

    alpha_sq_errors = (learner.alpha.flatten() - np.array(true_alpha)) ** 2

    beta_pairs_actual = learner.ols_beta_pairs_
    beta_sq_errors = np.array([
        (learner.beta[p] - true_beta_dict.get(p, np.nan)) ** 2
        for p in beta_pairs_actual
    ])

    gamma_sq_errors = []
    for name, true_val in true_gamma_edge_cov.items():
        gamma_sq_errors.append(
            (learner.gamma_.get(name, np.nan) - true_val) ** 2
        )

    if "sim_temp" in learner.time_cov_names:
        gamma_sq_errors.append(
            (learner.gamma_.get("sim_temp", np.nan) - true_gamma_time_cov) ** 2
        )

    for (edge_name, time_name), true_val in true_gamma_interaction.items():
        gamma_sq_errors.append(
            (
                learner.gamma_.get(f"{edge_name}_X_{time_name}", np.nan)
                - true_val
            ) ** 2
        )

    gamma_sq_errors = np.array(gamma_sq_errors)

    return {
        "learner": learner,
        "T": T,
        "seed": seed,
        "exog_u_correlation": exog_u_correlation,
        "alpha_sq_errors": alpha_sq_errors,
        "beta_sq_errors": beta_sq_errors,
        "gamma_sq_errors": gamma_sq_errors,
    }

def format_condition_label(r):
    """
    Standardizes the noise-condition label for a misspecification replication result.

    Parameters
    ----------
    r : dict
        A single replication result, containing "noise_dist" and (if applicable)
        "t_df" keys.

    Returns
    -------
    str
        "normal" if the innovations were Gaussian, "t (df=X)" if t-distributed
        with X degrees of freedom, or the raw noise_dist value otherwise.
    """
    dist = r.get("noise_dist")
    if dist == "normal":
        return "normal"
    elif dist == "t":
        df = r.get("t_df")
        return f"t (df={df})" if df is not None else "t (df=3)"
    return str(dist)


def build_alpha_error_df(results, beta_pair_labels, L_to_show=3):
    """
    Builds a long-format DataFrame of per-replication absolute errors for the
    autoregressive (alpha) parameters, across heavy-tailed innovation conditions.

    Parameters
    ----------
    results : list of dict
        Replication results from run_misspecification_replication, each
        containing "structure", "noise_dist", "t_df", and "alpha_sq_errors".
    beta_pair_labels : list of str
        Unused here, kept for a consistent function signature across builders.
    L_to_show : int, default 3
        Number of autoregressive lags to include.

    Returns
    -------
    pd.DataFrame
        Columns: structure, condition, alpha_index, abs_error.
    """
    rows = []
    for r in results:
        label = format_condition_label(r)
        for l in range(L_to_show):
            rows.append({
                "structure": r["structure"],
                "condition": label,
                "alpha_index": f"alpha{l+1}",
                "abs_error": np.sqrt(r["alpha_sq_errors"][l]),
            })
    return pd.DataFrame(rows)


def build_beta_error_df(results, beta_pair_labels):
    """
    Builds a long-format DataFrame of per-replication absolute errors for the
    network-effect (beta) parameters, across heavy-tailed innovation conditions.

    Parameters
    ----------
    results : list of dict
        Replication results, each containing "structure", "noise_dist", "t_df",
        and "beta_sq_errors".
    beta_pair_labels : list of str
        Labels (e.g. "beta1,1") corresponding to the order of beta_sq_errors.

    Returns
    -------
    pd.DataFrame
        Columns: structure, condition, beta_pair, abs_error.
    """
    rows = []
    for r in results:
        label = format_condition_label(r)
        for i, sq_err in enumerate(r["beta_sq_errors"]):
            rows.append({
                "structure": r["structure"],
                "condition": label,
                "beta_pair": beta_pair_labels[i],
                "abs_error": np.sqrt(sq_err),
            })
    return pd.DataFrame(rows)


def build_gamma_error_df(results, gamma_labels):
    """
    Builds a long-format DataFrame of per-replication absolute errors for the
    covariate (gamma) parameters, across heavy-tailed innovation conditions.

    Parameters
    ----------
    results : list of dict
        Replication results, each containing "structure", "noise_dist", "t_df",
        and "gamma_sq_errors".
    gamma_labels : list of str
        Covariate names corresponding to the order of gamma_sq_errors.

    Returns
    -------
    pd.DataFrame
        Columns: structure, condition, covariate, abs_error.
    """
    rows = []
    for r in results:
        condition = format_condition_label(r)
        for i, sq_err in enumerate(r["gamma_sq_errors"]):
            rows.append({
                "structure": r.get("structure"),
                "condition": condition,
                "covariate": gamma_labels[i],
                "abs_error": np.sqrt(sq_err),
            })
    return pd.DataFrame(rows)


def build_all_param_error_df(results, beta_pair_labels, gamma_labels, correlation_to_show=0.5):
    """
    Builds a long-format DataFrame of per-replication absolute errors for all
    parameter types (alpha, beta, gamma), comparing independent versus
    correlated innovations at a specified correlation level.

    Parameters
    ----------
    results : list of dict
        Replication results, each containing "structure", "correlation",
        "alpha_sq_errors", "beta_sq_errors", and "gamma_sq_errors".
    beta_pair_labels : list of str
        Labels corresponding to the order of beta_sq_errors.
    gamma_labels : list of str
        Covariate names corresponding to the order of gamma_sq_errors.
    correlation_to_show : float, default 0.5
        The nonzero correlation level to compare against the independent
        (correlation=0.0) baseline.

    Returns
    -------
    pd.DataFrame
        Columns: structure, setting, param_type, abs_error.
    """
    rows = []
    for r in results:
        if r["correlation"] not in [0.0, correlation_to_show]:
            continue
        label = "independent" if r["correlation"] == 0.0 else f"correlated (r={correlation_to_show})"

        for i, sq_err in enumerate(r["alpha_sq_errors"]):
            rows.append({"structure": r["structure"], "setting": label, "param_type": f"alpha{i+1}", "abs_error": np.sqrt(sq_err)})
        for i, sq_err in enumerate(r["beta_sq_errors"]):
            rows.append({"structure": r["structure"], "setting": label, "param_type": beta_pair_labels[i], "abs_error": np.sqrt(sq_err)})
        for i, sq_err in enumerate(r["gamma_sq_errors"]):
            rows.append({"structure": r["structure"], "setting": label, "param_type": gamma_labels[i], "abs_error": np.sqrt(sq_err)})
    return pd.DataFrame(rows)


def build_rewiring_overall_df(results):
    """
    Builds a long-format DataFrame of overall (pooled) RMSE for each parameter
    type, at each rewiring probability tested.

    Parameters
    ----------
    results : list of dict
        Replication results, each containing "rewiring_prob", "structure",
        "alpha_sq_errors", "beta_sq_errors", and "gamma_sq_errors".

    Returns
    -------
    pd.DataFrame
        Columns: rewiring_prob, structure, param_type, RMSE.
    """
    rows = []
    for r in results:
        rows.append({"rewiring_prob": r["rewiring_prob"], "structure": r["structure"],
                    "param_type": "alpha", "RMSE": np.sqrt(np.mean(r["alpha_sq_errors"]))})
        rows.append({"rewiring_prob": r["rewiring_prob"], "structure": r["structure"],
                    "param_type": "beta", "RMSE": np.sqrt(np.mean(r["beta_sq_errors"]))})
        rows.append({"rewiring_prob": r["rewiring_prob"], "structure": r["structure"],
                    "param_type": "gamma", "RMSE": np.sqrt(np.mean(r["gamma_sq_errors"]))})
    return pd.DataFrame(rows)


def build_rewiring_gamma_df(results, gamma_labels):
    """
    Builds a long-format DataFrame of per-covariate RMSE at each rewiring
    probability tested.

    Parameters
    ----------
    results : list of dict
        Replication results, each containing "rewiring_prob", "structure", and
        "gamma_sq_errors".
    gamma_labels : list of str
        Covariate names corresponding to the order of gamma_sq_errors.

    Returns
    -------
    pd.DataFrame
        Columns: rewiring_prob, structure, covariate, RMSE.
    """
    rows = []
    for r in results:
        for i, sq_err in enumerate(r["gamma_sq_errors"]):
            rows.append({
                "rewiring_prob": r["rewiring_prob"],
                "structure": r["structure"],
                "covariate": gamma_labels[i],
                "RMSE": np.sqrt(sq_err),
            })
    return pd.DataFrame(rows)


def build_exog_corr_error_df(results, beta_pair_labels, gamma_labels, correlation_to_show=0.5):
    """
    Builds a long-format DataFrame of per-replication absolute errors for all
    parameter types, comparing independent versus exogenous-correlated
    innovations at a specified correlation level.

    Parameters
    ----------
    results : list of dict
        Replication results, each containing "structure", "exog_u_correlation",
        "alpha_sq_errors", "beta_sq_errors", and "gamma_sq_errors".
    beta_pair_labels : list of str
        Labels corresponding to the order of beta_sq_errors.
    gamma_labels : list of str
        Covariate names corresponding to the order of gamma_sq_errors.
    correlation_to_show : float, default 0.5
        The nonzero exogenous-innovation correlation level to compare against
        the independent (exog_u_correlation=0.0) baseline.

    Returns
    -------
    pd.DataFrame
        Columns: structure, setting, param_type, abs_error.
    """
    rows = []
    for r in results:
        if r["exog_u_correlation"] not in [0.0, correlation_to_show]:
            continue
        label = "independent" if r["exog_u_correlation"] == 0.0 else f"correlated (r={correlation_to_show})"

        for i, sq_err in enumerate(r["alpha_sq_errors"]):
            rows.append({"structure": r["structure"], "setting": label, "param_type": f"alpha{i+1}", "abs_error": np.sqrt(sq_err)})
        for i, sq_err in enumerate(r["beta_sq_errors"]):
            rows.append({"structure": r["structure"], "setting": label, "param_type": beta_pair_labels[i], "abs_error": np.sqrt(sq_err)})
        for i, sq_err in enumerate(r["gamma_sq_errors"]):
            rows.append({"structure": r["structure"], "setting": label, "param_type": gamma_labels[i], "abs_error": np.sqrt(sq_err)})

    return pd.DataFrame(rows)

def build_correctly_specified_boxplot_df(results_with, results_without, beta_pair_labels,
                                          gamma_labels_with, gamma_labels_without,
                                          filter_col, filter_value):
    """
    Builds a long-format DataFrame of per-replication absolute errors,
    for boxplotting, comparing with/without interaction at a fixed baseline condition.
    """
    rows = []

    for regime_label, results, gamma_labels in [
        ("with_interaction", results_with, gamma_labels_with),
        ("no_interaction", results_without, gamma_labels_without),
    ]:
        subset = [r for r in results if r[filter_col] == filter_value]

        for r in subset:
            for i, sq_err in enumerate(r["alpha_sq_errors"]):
                rows.append({"structure": r["structure"], "regime": regime_label,
                           "parameter": f"alpha{i+1}", "abs_error": np.sqrt(sq_err)})
            for i, sq_err in enumerate(r["beta_sq_errors"]):
                rows.append({"structure": r["structure"], "regime": regime_label,
                           "parameter": beta_pair_labels[i], "abs_error": np.sqrt(sq_err)})
            for i, sq_err in enumerate(r["gamma_sq_errors"]):
                cov_name = gamma_labels[i]
                #only include covariates present in both regimes, plus the interaction term itself
                if cov_name in gamma_labels_without or cov_name == "mean_age_X_sim_temp":
                    rows.append({"structure": r["structure"], "regime": regime_label,
                               "parameter": cov_name, "abs_error": np.sqrt(sq_err)})

    return pd.DataFrame(rows)

def run_heavy_tail_prediction_replication(K, density, structure, rdp_radius_cache, L, stages_per_lag,
                                            true_alpha, true_beta_dict, true_gamma_edge_cov,
                                            true_gamma_time_cov, interaction_pairs, true_gamma_interaction,
                                            T, seed, noise_dist="normal", t_df=3):
    """
    Run one replication of the heavy-tailed prediction simulation study.

    This function simulates a GNAR-edge process under either Gaussian or
    heavy-tailed innovations, fits a GNAR-edge model using the first ``T - 1``
    observations, and evaluates one-step-ahead prediction accuracy on the held-out
    final observation. Prediction performance is summarized by the root mean
    squared error (RMSE) across all edges.

    Parameters
    ----------
    K : int
        Number of nodes in the simulated network.
    density : float
        Target network density.
    structure : {"ER", "SBM", "RDP"}
        Network generation mechanism.
    rdp_radius_cache : dict
        Cache of calibrated RDP radii used when generating radius-dependent random
        graphs.
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag.
    true_alpha : array-like
        True autoregressive coefficients used in the data-generating process.
    true_beta_dict : dict
        Dictionary mapping ``(lag, stage)`` pairs to the true neighbour-effect
        coefficients.
    true_gamma_edge_cov : dict
        Dictionary mapping edge-covariate names to their true coefficients.
    true_gamma_time_cov : float
        True coefficient for the time-varying exogenous covariate.
    interaction_pairs : list of tuple
        Edge/time covariate interaction terms included in the simulated model.
    true_gamma_interaction : dict
        Dictionary of true interaction coefficients.
    T : int
        Length of the simulated time series.
    seed : int
        Random seed used for reproducibility.
    noise_dist : {"normal", "t"}, default="normal"
        Innovation distribution used when simulating the data-generating process.
    t_df : int, default=3
        Degrees of freedom of the Student's t innovations when
        ``noise_dist="t"``.

    Returns
    -------
    dict
        Dictionary containing

        - ``noise_dist`` : innovation distribution used for simulation.
        - ``t_df`` : degrees of freedom for the t-distribution.
        - ``seed`` : random seed.
        - ``structure`` : network generation mechanism.
        - ``RMSE`` : one-step-ahead prediction root mean squared error.

    Notes
    -----
    The fitted GNAR-edge model is always estimated using ordinary least squares.
    The purpose of this simulation is to assess how departures from Gaussian
    innovations affect out-of-sample prediction performance.
    """
    #generating network and simulating edge covariates
    edge_list, _ = generate_network(K=K, density=density, structure=structure, rdp_radius_cache = rdp_radius_cache, seed=seed)
    ages, sexes = simulate_node_covariates(K=K, seed=seed)
    edge_covariates_full = build_synthetic_edge_covariates(edge_list, ages, sexes)
    edge_covariates_sim = {name: edge_covariates_full[name] for name in true_gamma_edge_cov.keys()}

    #simulating full model
    X_sim, time_cov_sim, _ = simulate_full_model_misspec(
        edge_list, K, L, stages_per_lag, true_alpha, true_beta_dict,
        edge_covariates_sim, true_gamma_edge_cov, true_gamma_time_cov,
        interaction_pairs, true_gamma_interaction, T=T, seed=seed,
        noise_dist=noise_dist, t_df=t_df,
    )

    time_index = pd.date_range("2020-01-01", periods=T, freq="D")
    train_index = time_index[:-1]
    test_target = X_sim[-1, :]

    exog_series_sim = {}
    if true_gamma_time_cov != 0.0 or len(interaction_pairs) > 0:
        exog_series_sim = {"sim_temp": pd.Series(time_cov_sim[:-1], index=train_index)}

    #fitting GNAR edge model to simulated data
    sim_graph = ArrayEdgeGraph.from_edge_panel(X_sim[:-1, :], edge_list, n_nodes=K, time_labels=train_index)
    learner = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph, L=L, stages_per_lag=stages_per_lag,
        train_periods=train_index, use_ols=True,
        edge_covariates=edge_covariates_sim, exog_series=exog_series_sim,
        interaction_pairs=interaction_pairs,
    )
    learner.fit(use_ols=True, verbose=False)

    #predicting next period using fitted model
    if exog_series_sim:
        next_temp = pd.Series([time_cov_sim[-1]], index=[time_index[-1]])
        learner.exog_series["sim_temp"] = pd.concat([exog_series_sim["sim_temp"], next_temp])
    pred = learner.predict_next(periods=train_index)
    rmse = np.sqrt(np.mean((pred - test_target)**2))

    return {"noise_dist": noise_dist, "t_df": t_df, "seed": seed, "structure": structure, "RMSE": rmse}


def build_per_parameter_interaction_comparison(results_with, results_without,
                                                  beta_pair_labels, gamma_labels_with, gamma_labels_without,
                                                  rewiring_prob, structure):
    """
    Compare parameter recovery with and without interaction effects.

    This function summarises the root mean squared error (RMSE) of parameter
    estimates obtained from two simulation experiments: one in which the fitted
    GNAR-edge model includes the interaction term and one in which it does not.
    For each autoregressive coefficient, neighbour-effect coefficient, and shared
    covariate coefficient, the RMSE is computed across all simulation replications.
    The interaction coefficient is reported only for the model that includes the
    interaction term.

    Parameters
    ----------
    results_with : list of dict
        Simulation results obtained from the model including the interaction term.
        Each dictionary must contain the squared estimation errors for the
        autoregressive, neighbour, and covariate parameters.
    results_without : list of dict
        Simulation results obtained from the model excluding the interaction term.
    beta_pair_labels : list of str
        Labels corresponding to the neighbour-effect coefficients in the order they
        appear in the simulation output.
    gamma_labels_with : list of str
        Names of the covariate coefficients for the model including the interaction
        term.
    gamma_labels_without : list of str
        Names of the covariate coefficients for the model excluding the interaction
        term.
    rewiring_prob : float
        Network rewiring probability used to select the simulation results.
    structure : {"ER", "SBM", "RDP"}
        Network structure used to select the simulation results.

    Returns
    -------
    pandas.DataFrame
        Data frame containing one row per parameter with the columns

        - ``parameter`` : parameter name.
        - ``RMSE_with_interaction`` : RMSE from the model including the interaction
        term.
        - ``RMSE_no_interaction`` : RMSE from the model excluding the interaction
        term.

    Notes
    -----
    The interaction parameter ("Mean Age × Temp") is reported only for the model
    including the interaction term, since no corresponding estimate exists in the
    model without interactions.
    """
    rows = []

    subset_with = [r for r in results_with if r["rewiring_prob"] == rewiring_prob and r["structure"] == structure]
    subset_without = [r for r in results_without if r["rewiring_prob"] == rewiring_prob and r["structure"] == structure]

    n_alpha = len(subset_with[0]["alpha_sq_errors"])
    for i in range(n_alpha):
        rmse_with = np.sqrt(np.mean([r["alpha_sq_errors"][i] for r in subset_with]))
        rmse_without = np.sqrt(np.mean([r["alpha_sq_errors"][i] for r in subset_without]))
        rows.append({"parameter": f"Alpha {i+1}", "RMSE_with_interaction": rmse_with, "RMSE_no_interaction": rmse_without})

    n_beta = len(subset_with[0]["beta_sq_errors"])
    for i in range(n_beta):
        label = beta_pair_labels[i]
        rmse_with = np.sqrt(np.mean([r["beta_sq_errors"][i] for r in subset_with]))
        rmse_without = np.sqrt(np.mean([r["beta_sq_errors"][i] for r in subset_without]))
        rows.append({"parameter": label, "RMSE_with_interaction": rmse_with, "RMSE_no_interaction": rmse_without})

    shared_gamma = [c for c in gamma_labels_without if c in gamma_labels_with]
    for cov in shared_gamma:
        idx_with = gamma_labels_with.index(cov)
        idx_without = gamma_labels_without.index(cov)
        rmse_with = np.sqrt(np.mean([r["gamma_sq_errors"][idx_with] for r in subset_with]))
        rmse_without = np.sqrt(np.mean([r["gamma_sq_errors"][idx_without] for r in subset_without]))
        rows.append({"parameter": cov, "RMSE_with_interaction": rmse_with, "RMSE_no_interaction": rmse_without})

    interaction_idx = gamma_labels_with.index("mean_age_X_sim_temp")
    rmse_interaction = np.sqrt(np.mean([r["gamma_sq_errors"][interaction_idx] for r in subset_with]))
    rows.append({"parameter": "Mean Age x Temp", "RMSE_with_interaction": rmse_interaction, "RMSE_no_interaction": np.nan})

    return pd.DataFrame(rows)

def build_gamma_comparison_with_increase(
    with_df,
    without_df,
    gamma_labels_with,
    gamma_labels_without,
):
    """
    Compare gamma-parameter estimation errors between two model
    specifications and calculate absolute and percentage increases.

    Parameters
    ----------
    with_df : pandas.DataFrame
        Summary table for the model including the interaction term.
        Must contain columns ``"Parameter"`` and ``"Error"``.
    without_df : pandas.DataFrame
        Summary table for the model without the interaction term.
        Must contain columns ``"Parameter"`` and ``"Error"``.
    gamma_labels_with : list
        Parameter labels included in the model with the interaction.
    gamma_labels_without : list
        Parameter labels included in the model without the interaction.
        Only parameters present in this list are compared.

    Returns
    -------
    pandas.DataFrame
        Comparison table containing the error for each model and the
        absolute and percentage increase in error for the model with
        the interaction.

    Notes
    -----
    The interaction parameter itself is excluded because it has no
    corresponding parameter in the no-interaction specification.
    """

    # Restrict to parameters present in both regimes
    shared_params = [
        parameter
        for parameter in gamma_labels_without
        if parameter in gamma_labels_with
    ]

    with_subset = with_df[
        with_df["Parameter"].isin(shared_params)
    ].copy()

    without_subset = without_df[
        without_df["Parameter"].isin(shared_params)
    ].copy()

    # Rename error columns before merging
    with_subset = with_subset.rename(
        columns={"Error": "Error_with"}
    )

    without_subset = without_subset.rename(
        columns={"Error": "Error_without"}
    )

    merged = with_subset.merge(
        without_subset,
        on="Parameter",
        how="inner",
    )

    merged["Absolute Increase"] = (
        merged["Error_with"]
        - merged["Error_without"]
    ).round(4)

    merged["Percentage Increase"] = (
        (
            merged["Error_with"]
            - merged["Error_without"]
        )
        / merged["Error_without"]
        * 100
    ).round(1)

    return merged

def build_rewiring_gamma_comparison(
    rewiring_prob_value,
    rewiring_results_with,
    rewiring_results_without,
    gamma_labels_with,
    gamma_labels_without,
    param_labels,
    structure="SBM",
):
    """
    Compare gamma-parameter RMSE between interaction and no-interaction
    models under a specified network rewiring probability.

    Parameters
    ----------
    rewiring_prob_value : float
        Rewiring probability at which the comparison is performed.
    rewiring_results_with : list
        Simulation results for the model with the interaction term.
    rewiring_results_without : list
        Simulation results for the model without the interaction term.
    gamma_labels_with : list
        Gamma parameter labels for the model with the interaction term.
    gamma_labels_without : list
        Gamma parameter labels for the model without the interaction term.
    param_labels : dict
        Mapping from internal parameter names to display labels.
    structure : {"ER", "SBM", "RDP"}, default="SBM"
        Network structure included in the comparison.

    Returns
    -------
    pandas.DataFrame
        Wide-format table indexed by parameter. Columns contain the mean
        RMSE for each model, together with the absolute and percentage
        increase in RMSE for the model with the interaction.

    Notes
    -----
    Parameters that are not present in both model specifications are
    excluded from the comparison.
    """

    rows = []

    for label, results, gamma_labels_here in [
        (
            "With interaction",
            rewiring_results_with,
            gamma_labels_with,
        ),
        (
            "No interaction",
            rewiring_results_without,
            gamma_labels_without,
        ),
    ]:
        df = build_rewiring_gamma_df(
            results,
            gamma_labels_here,
        )

        df = df[
            (df["rewiring_prob"] == rewiring_prob_value)
            & (df["structure"] == structure)
        ]

        df_summary = (
            df.groupby("covariate")["RMSE"]
            .mean()
        )

        for parameter, value in df_summary.items():
            rows.append({
                "Regime": label,
                "Parameter": param_labels.get(
                    parameter,
                    parameter,
                ),
                "RMSE": value,
            })

    long_df = pd.DataFrame(rows)

    wide_df = long_df.pivot(
        index="Parameter",
        columns="Regime",
        values="RMSE",
    )

    # Keep only parameters shared across both regimes
    wide_df = wide_df.dropna()

    wide_df["Absolute Increase"] = (
        wide_df["With interaction"]
        - wide_df["No interaction"]
    ).round(4)

    wide_df["Percentage Increase"] = (
        (
            wide_df["With interaction"]
            - wide_df["No interaction"]
        )
        / wide_df["No interaction"]
        * 100
    ).round(1)

    return wide_df.round(4)

def build_exogcorr_gamma_comparison(
    correlation_value,
    exog_corr_results_with,
    exog_corr_results_without,
    beta_pair_labels_with,
    beta_pair_labels_without,
    gamma_labels_with,
    gamma_labels_without,
    param_labels,
    structure="SBM",
):
    """
    Compare gamma-parameter estimation errors between interaction and
    no-interaction models under a specified innovation correlation.

    Parameters
    ----------
    correlation_value : float
        Innovation correlation at which the comparison is performed.
    exog_corr_results_with : list
        Simulation results for the model with the interaction term.
    exog_corr_results_without : list
        Simulation results for the model without the interaction term.
    beta_pair_labels_with : list
        Beta parameter labels for the model with the interaction term.
    beta_pair_labels_without : list
        Beta parameter labels for the model without the interaction term.
    gamma_labels_with : list
        Gamma parameter labels for the model with the interaction term.
    gamma_labels_without : list
        Gamma parameter labels for the model without the interaction term.
    param_labels : dict
        Mapping from internal parameter names to display labels.
    structure : {"ER", "SBM", "RDP"}, default="SBM"
        Network structure included in the comparison.

    Returns
    -------
    pandas.DataFrame
        Wide-format comparison table containing the mean absolute
        estimation error for each model, together with the absolute and
        percentage increase in error for the model with the interaction.

    Notes
    -----
    Only parameters shared across both model specifications are included.
    The interaction parameter itself is therefore excluded.
    """

    rows = []

    for label, results, beta_labels_here, gamma_labels_here in [
        (
            "With interaction",
            exog_corr_results_with,
            beta_pair_labels_with,
            gamma_labels_with,
        ),
        (
            "No interaction",
            exog_corr_results_without,
            beta_pair_labels_without,
            gamma_labels_without,
        ),
    ]:
        df = build_exog_corr_error_df(
            results,
            beta_labels_here,
            gamma_labels_here,
            correlation_to_show=correlation_value,
        )

        df["param_type"] = (
            df["param_type"]
            .replace(param_labels)
        )

        gamma_names = [
            param_labels.get(
                gamma,
                gamma,
            )
            for gamma in gamma_labels_here
        ]

        df = df[
            (df["structure"] == structure)
            & (
                df["setting"]
                .str.startswith("correlated")
            )
            & (
                df["param_type"]
                .isin(gamma_names)
            )
        ]

        df_summary = (
            df.groupby("param_type")["abs_error"]
            .mean()
        )

        for parameter, value in df_summary.items():
            rows.append({
                "Regime": label,
                "Parameter": parameter,
                "Error": value,
            })

    long_df = pd.DataFrame(rows)

    wide_df = long_df.pivot(
        index="Parameter",
        columns="Regime",
        values="Error",
    )

    # Keep only parameters shared across both regimes
    wide_df = wide_df.dropna()

    wide_df["Absolute Increase"] = (
        wide_df["With interaction"]
        - wide_df["No interaction"]
    ).round(4)

    wide_df["Percentage Increase"] = (
        (
            wide_df["With interaction"]
            - wide_df["No interaction"]
        )
        / wide_df["No interaction"]
        * 100
    ).round(1)

    return wide_df.round(4)

def simulate_full_model_misspec_unscaled(
    edge_list,
    K,
    L,
    stages_per_lag,
    true_alpha,
    true_beta_dict,
    edge_covariates,
    true_gamma_edge_cov,
    true_gamma_time_cov,
    interaction_pairs,
    true_gamma_interaction,
    T,
    noise_std=1.0,
    noise_dist="normal",
    t_df=3,
    time_cov_name="sim_temp",
    seed=None,
):
    """
    Simulate edge-weight time series under alternative innovation
    distributions without variance-matching the Student's t innovations.

    This function follows ``simulate_full_model_misspec`` but deliberately
    does not rescale Student's t innovations to have the same marginal
    variance as the Gaussian innovations. Consequently, the innovation
    variance depends on the degrees of freedom ``t_df``.

    Parameters
    ----------
    edge_list : list of tuple of int
        Edges of the network, represented as ``(i, j)`` node-index pairs.
    K : int
        Number of nodes in the network.
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag.
    true_alpha : list of float
        True autoregressive coefficients.
    true_beta_dict : dict
        True neighbour-effect coefficients indexed by ``(lag, stage)``.
    edge_covariates : dict
        Mapping from edge-covariate names to arrays of length ``E``.
    true_gamma_edge_cov : dict
        True coefficients associated with edge-level covariates.
    true_gamma_time_cov : float
        True coefficient associated with the time-varying exogenous
        covariate.
    interaction_pairs : list of tuple
        Edge/time covariate interaction terms.
    true_gamma_interaction : dict
        True interaction coefficients.
    T : int
        Length of the simulated time series.
    noise_std : float, default=1.0
        Scale parameter for the innovations. For Student's t innovations,
        this is multiplied directly by the raw t random variable and is
        therefore not a marginal standard deviation.
    noise_dist : {"normal", "t"}, default="normal"
        Innovation distribution.
    t_df : int, default=3
        Degrees of freedom for Student's t innovations.
    time_cov_name : str, default="sim_temp"
        Name assigned to the simulated time-varying exogenous series.
    seed : int, optional
        Random seed used for reproducibility.

    Returns
    -------
    X : np.ndarray
        Simulated edge-weight time-series panel of shape ``(T, E)``.
    time_cov : np.ndarray
        Simulated time-varying exogenous series.
    graph : object
        Underlying network graph.

    Notes
    -----
    Unlike ``simulate_full_model_misspec``, Student's t innovations are
    generated as ``standard_t(t_df) * noise_std`` with no variance
    rescaling. Their variance is therefore ``noise_std**2 * t_df /
    (t_df - 2)`` when ``t_df > 2``.
    """

    rng = np.random.default_rng(seed)
    E = len(edge_list)

    needed_r = sorted({
        r
        for rs in stages_per_lag
        for r in rs
    })
    #building neighbourhood matrices
    W_dict, graph = build_W_matrices(
        edge_list,
        K,
        needed_r,
    )
    #building exogenous series
    time_cov = simulate_ar1_exog(
        T,
        phi=0.7,
        mean=30,
        sd=3,
        seed=seed,
    )

    X = np.zeros((T, E))

    X[:L, :] = rng.normal(
        0,
        noise_std,
        size=(L, E),
    )

    #simulating remaining parts of the model and predicting next period
    for t in range(L, T):
        pred = np.zeros(E)

        for l in range(1, L + 1):
            pred += (
                true_alpha[l - 1]
                * X[t - l, :]
            )

            for r in stages_per_lag[l - 1]:
                pred += (
                    true_beta_dict[(l, r)]
                    * (W_dict[r] @ X[t - l, :])
                )

        for name, gamma_val in true_gamma_edge_cov.items():
            pred += (
                gamma_val
                * edge_covariates[name]
            )

        pred += (
            true_gamma_time_cov
            * time_cov[t]
        )

        for edge_name, cov_name in interaction_pairs:
            gamma_val = true_gamma_interaction[
                (edge_name, cov_name)
            ]

            pred += (
                gamma_val
                * edge_covariates[edge_name]
                * time_cov[t]
            )

        #noise generation, by distribution
        if noise_dist == "normal":
            noise = rng.normal(
                0,
                noise_std,
                size=E,
            )

        elif noise_dist == "t":
            #no rescaling -- raw t-distributed noise
            noise = (
                rng.standard_t(t_df, size=E)
                * noise_std
            )

        else:
            raise ValueError(
                "noise_dist must be 'normal' or 't'"
            )

        X[t, :] = pred + noise

    return X, time_cov, graph

def run_heavy_tail_prediction_replication_unscaled(
    K,
    density,
    structure,
    rdp_radius_cache,
    L,
    stages_per_lag,
    true_alpha,
    true_beta_dict,
    true_gamma_edge_cov,
    true_gamma_time_cov,
    interaction_pairs,
    true_gamma_interaction,
    T,
    seed,
    noise_dist="normal",
    t_df=3,
):
    """
    Run one one-step-ahead prediction replication using unscaled
    Student's t innovations.

    The synthetic data are generated from the specified GNAR-edge DGP
    and the final observation is held out. The extended GNAR-edge model
    is fitted to the first ``T - 1`` observations and used to produce a
    one-step-ahead forecast for every edge.

    Parameters
    ----------
    K : int
        Number of nodes in the simulated network.
    density : float
        Target network density.
    structure : {"ER", "SBM", "RDP"}
        Network generation mechanism.
    rdp_radius_cache : dict
        Cache of calibrated RDP radii.
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag.
    true_alpha : array-like
        True autoregressive coefficients.
    true_beta_dict : dict
        True neighbour-effect coefficients.
    true_gamma_edge_cov : dict
        True edge-covariate coefficients.
    true_gamma_time_cov : float
        True time-varying exogenous coefficient.
    interaction_pairs : list of tuple
        Interaction terms in the DGP.
    true_gamma_interaction : dict
        True interaction coefficients.
    T : int
        Length of the simulated time series.
    seed : int
        Random seed.
    noise_dist : {"normal", "t"}, default="normal"
        Innovation distribution.
    t_df : int, default=3
        Degrees of freedom for the Student's t innovations.

    Returns
    -------
    dict
        Dictionary containing the one-step-ahead RMSE of the fitted
        extended GNAR-edge model and the simulation settings.
    """

    #generating network and simulating edge covariates
    edge_list, _ = generate_network(
        K=K,
        density=density,
        structure=structure,
        seed=seed,
        rdp_radius_cache=rdp_radius_cache,
    )

    E = len(edge_list)

    ages, sexes = simulate_node_covariates(
        K=K,
        seed=seed,
    )

    edge_covariates_full = build_synthetic_edge_covariates(
        edge_list,
        ages,
        sexes,
    )

    edge_covariates_sim = {
        name: edge_covariates_full[name]
        for name in true_gamma_edge_cov.keys()
    }

    #simulating the full model with unscaled innovations
    X_sim, time_cov_sim, graph = (
        simulate_full_model_misspec_unscaled(
            edge_list,
            K,
            L,
            stages_per_lag,
            true_alpha,
            true_beta_dict,
            edge_covariates_sim,
            true_gamma_edge_cov,
            true_gamma_time_cov,
            interaction_pairs,
            true_gamma_interaction,
            T=T,
            seed=seed,
            noise_dist=noise_dist,
            t_df=t_df,
        )
    )

    #defining training and test periods
    time_index = pd.date_range(
        "2020-01-01",
        periods=T,
        freq="D",
    )

    train_index = time_index[:-1]
    test_index = time_index[-1:]

    train_data = X_sim[:-1, :]
    test_target = X_sim[-1, :]

    #storing simulated exogenous series
    exog_series_sim = {}

    if (
        true_gamma_time_cov != 0.0
        or len(interaction_pairs) > 0
    ):
        exog_series_sim = {
            "sim_temp": pd.Series(
                time_cov_sim[:-1],
                index=train_index,
            )
        }

    #building graph from training data
    sim_graph = ArrayEdgeGraph.from_edge_panel(
        train_data,
        edge_list,
        n_nodes=K,
        time_labels=train_index,
    )

    #fitting the extended GNAR-edge model
    learner = GNAREdgeGlobalMultiCovLearner(
        graph=sim_graph,
        L=L,
        stages_per_lag=stages_per_lag,
        train_periods=train_index,
        use_ols=True,
        edge_covariates=edge_covariates_sim,
        exog_series=exog_series_sim,
        interaction_pairs=interaction_pairs,
    )

    learner.fit(
        use_ols=True,
        verbose=False,
    )

    #adding the held-out exogenous value for one-step-ahead prediction
    if (
        true_gamma_time_cov != 0.0
        or len(interaction_pairs) > 0
    ):
        learner.exog_series["sim_temp"] = pd.concat([
            exog_series_sim["sim_temp"],
            pd.Series(
                [time_cov_sim[-1]],
                index=test_index,
            ),
        ])

    #predicting the held-out final observation
    pred = learner.predict_next(
        periods=train_index
    )

    rmse = np.sqrt(
        np.mean(
            (pred - test_target) ** 2
        )
    )

    return {
        "df": "unscaled_t",
        "noise_dist": noise_dist,
        "t_df": t_df,
        "structure": structure,
        "seed": seed,
        "rmse": rmse,
    }

def build_true_psi_list(
    edge_list,
    K,
    L,
    stages_per_lag,
    true_alpha,
    true_beta_dict,
):
    """
    Construct the true lag coefficient matrices from the data-generating
    process.

    For each lag ``l``, this function constructs the GNAR-edge coefficient
    matrix

        Psi_l = alpha_l I + sum_r beta_{l,r} W_r,

    where ``I`` is the identity matrix and ``W_r`` is the network weight
    matrix corresponding to neighbourhood stage ``r``. The construction
    mirrors the autoregressive and neighbour-effect components used in
    ``simulate_full_model``.

    Parameters
    ----------
    edge_list : list of tuple
        Edge list defining the network topology.
    K : int
        Number of nodes in the network.
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag. The entry at index ``l - 1``
        contains the stages included for lag ``l``.
    true_alpha : array-like
        True autoregressive coefficients, with one coefficient for each lag.
    true_beta_dict : dict
        Dictionary mapping ``(lag, stage)`` pairs to the corresponding true
        neighbour-effect coefficients.

    Returns
    -------
    list of numpy.ndarray
        List containing the ``E x E`` true coefficient matrix ``Psi_l`` for
        each lag, where ``E`` is the number of edges in the network.
    """
    E = len(edge_list)

    needed_r = sorted({
        r
        for rs in stages_per_lag
        for r in rs
    })

    W_dict, _ = build_W_matrices(
        edge_list,
        K,
        needed_r,
    )

    Psi_list = []

    for l in range(1, L + 1):
        Psi_l = true_alpha[l - 1] * np.eye(E)

        for r in stages_per_lag[l - 1]:
            Psi_l = (
                Psi_l
                + true_beta_dict[(l, r)] * W_dict[r]
            )

        Psi_list.append(Psi_l)

    return Psi_list


def check_true_stationarity(
    edge_list,
    K,
    L,
    stages_per_lag,
    true_alpha,
    true_beta_dict,
):
    """
    Check the stationarity of the true GNAR-edge data-generating process
    using the companion-matrix eigenvalue condition.

    The true lag coefficient matrices are first constructed using
    ``build_true_psi_list``. These matrices are then arranged into the
    companion matrix corresponding to the VAR representation of the
    GNAR-edge process. The process satisfies the eigenvalue-based
    stationarity condition when every eigenvalue of the companion matrix
    lies strictly inside the unit circle.

    Parameters
    ----------
    edge_list : list of tuple
        Edge list defining the network topology.
    K : int
        Number of nodes in the network.
    L : int
        Number of autoregressive lags.
    stages_per_lag : list of list of int
        Neighbour stages included at each lag.
    true_alpha : array-like
        True autoregressive coefficients used in the data-generating process.
    true_beta_dict : dict
        Dictionary mapping ``(lag, stage)`` pairs to the corresponding true
        neighbour-effect coefficients.

    Returns
    -------
    max_modulus : float
        Maximum modulus among the eigenvalues of the companion matrix.
    is_stationary : bool
        ``True`` if the maximum eigenvalue modulus is strictly less than one,
        and ``False`` otherwise.

    Notes
    -----
    This function checks stationarity using the true data-generating
    coefficients rather than coefficients estimated from a simulated sample.
    The exogenous covariate coefficients do not enter the companion matrix,
    since the stationarity condition concerns the autoregressive dynamics of
    the endogenous edge-weight process.
    """
    Psi_list = build_true_psi_list(
        edge_list,
        K,
        L,
        stages_per_lag,
        true_alpha,
        true_beta_dict,
    )

    E = Psi_list[0].shape[0]

    top_row = np.hstack(Psi_list)

    if L > 1:
        identity_block = np.eye(E * (L - 1))

        bottom = np.hstack([
            identity_block,
            np.zeros((E * (L - 1), E)),
        ])

        companion = np.vstack([
            top_row,
            bottom,
        ])

    else:
        companion = top_row

    eigenvalues = np.linalg.eigvals(companion)
    max_modulus = np.max(np.abs(eigenvalues))

    return max_modulus, max_modulus < 1