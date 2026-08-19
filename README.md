# Predicting Interactions within a Primate Social Network: An Extension of the GNAR-edge Model
```text
The code in this repository accompanies my dissertation, "Predicting Interactions within a Primate Social Network: An Extension of the GNAR-edge Model". In this dissertation, I extend the GNAR-edge model, introduced by Mantziou et al. (2023) to allow for inclusion of exogenous covariates. The extension is motivated by a data set that features a multivariate time series of pairwise interaction counts within a group of baboons. This data set was originally collected and analysed by Gelardi et al. (2020).
```

## Repository Structure
```text
baboon-gnar-edge/
├── README.md
├── data
│   ├── baboon_data.csv
│   ├── baboons_proximity_data.txt
│   └── trets.csv
├── external
│   ├── ARIMA_SARIMA_Baselines_Guide.docx
│   ├── BaseEdge.py
│   ├── BaseEdgeGNAR_edge.py
│   ├── BaseEdgeGNAR_edge_global.py
│   ├── GNAR_edge_Design_and_Usage_updated.docx
│   ├── RollingEdgePredict.py
│   ├── __pycache__
│   ├── edge_baselines.py
│   ├── edge_graph.py
│   ├── orchestration_stl.py
│   ├── run_baseline_demo.py
│   ├── run_payment_graph.py
│   └── run_simulated.py
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_simulation_parameter_estimation.ipynb
│   ├── 03_simulation_prediction.ipynb
│   ├── 04_model_misspecification.ipynb
│   ├── 05_real_data_application.ipynb
│   └── output
│       ├── dataframes
│       ├── figures
│       └── tables
├── requirements.txt
└── src
    ├── __init__.py
    ├── __pycache__
    ├── diagnostics.py
    ├── model.py
    ├── real_data_pipeline.py
    └── simulation.py
```

### Repository overview

- **data/** – Dataset of raw interaction counts and additional datasets used for exogenous covariates in the real data application (environmental and demographic data).
- **external/** – Original GNAR-edge implementation. The code for this project builds heavily on this original implementation.
- **src/** – Python modules developed for my project. This includes the modules containing the model extension, the functions needed to run simulation experiments, and the real-data pipeline.
- **notebooks/** – Jupyter notebooks in which the actual experiments and real data applications are run. These notebooks reproduce the results (figures, tables etc) from my project.
- **notebooks/output/** – Automatically generated output from the Jupyter notebook experiments and real data experiments.

## Notebook-to-Dissertation-Section Mapping

| Notebook | Thesis Section |
|---|---|
| `01_eda.ipynb` | Section 2: Motivational Data Set |
| `02_simulation_parameter_estimation.ipynb` | Section 4.2: Parameter Estimation |
| `03_simulation_prediction.ipynb` | Section 4.3: Predictive Performance |
| `04_model_misspecification.ipynb` | Section 5: Model Misspecification  |
| `05_real_data_application.ipynb` | Section 6: Real Data Application |

## Setup

The project was developed using **Python 3.13**.

To create and activate a virtual environment, and install the require packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=baboon-gnar-edge --display-name "Python (baboon-gnar-edge)"
```


## Data

The raw interaction data (`baboons_proximity_data.txt`) was originally collected by Gelardi et al. (2020). It can be accessed at https://sociopatterns.org/datasets/baboons-interactions. The daily maximum temperature data (`trets.csv`) originates from Météo-France's Open Data Climatology API: https://portail-api.meteofrance.fr/web/en/api/DonneesPubliquesClimatologie. Baboon age and sex data was obtained via direct communications with CNRS Primate Center.

## Note on notation

In the dissertation text, a different parameter name is used for each type of exogenous coefficient: $\mu_{p}$ refers to edge-level covariates, $\delta_{q}$ refers to the time-varying exogenous series, and $\eta_{p,q}$ refers to their interaction. In the codebase, these exogenous coefficients are all referred to using a single parameter name, `gamma`, and are stored in a joint `gamma_` dictionary in the fitted learner objects. Each type of exogenous coefficient is distinguised by key name (e.g.`gamma_["age_diff"]`, `gamma_["sim_temp"]`, `gamma_["age_diff_X_sim_temp"]`) rather than separate variable names.

## Testing Notebooks Quickly

When run at a full scale, some of the notebooks take several hours to run. This is especially notable for `03_simulation_prediction.ipynb` and
`04_model_misspecification.ipynb`. To verify that the pipeline runs succesfully without waiting for a full scale completion, at the top of these two notebooks is a `QUICK_TEST` flag. Setting `QUICK_TEST = True` reduces the number of structures considered and the number of replications performed.

Before reproducing results from the dissertation, ensure that `QUICK_TEST = False` in every notebook before running the experiments. This is the default setting, and is what was used to produce results for the dissertation itself.


## Acknowledgements

The original GNAR-edge implementation by Mantziou et al. (2023) was developed for R, and can be found here: https://github.com/mantziou/GNAR-edge-model. A python implementation of the GNAR-edge model was developed by Tian Xie, and can be found here: https://github.com/naive4E4A55/gnar-edge. The code for my project relies heavily on Tian Xie's python implementation. 

## Citation

If referencing this work, please cite:
Lucy Cheffins. (2026). "Predicting Interactions within a Primate Social Network: An Extension of the GNAR-edge Model". University of Oxford.
