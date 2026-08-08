# Extended GNAR-edge Model: A Primate Social Network Application
```text
The code in this repository accompanies my dissertation, "[INSERT TITLE]". In this dissertation, I extend the GNAR-edge model, introduced by Mantziou et al. (2023) to allow for inclusion of exogenous covariates. The extension is motivated by a data set that features a multivariate time series of pairwise interaction counts within a group of baboons. This data set was originally collected and analysed by Gelardi et al. (2020).
```

## Repository Structure

```text
baboon-gnar-edge/
├── data/
│   ├── baboons_proximity_data.txt
│   ├── baboon_data.csv
│   └── trets.csv
│
├── external/                     # original GNAR-edge implementation
│   ├── BaseEdge.py
│   ├── BaseEdgeGNAR_edge.py
│   ├── BaseEdgeGNAR_edge_global.py
│   ├── edge_baselines.py
│   ├── edge_graph.py
│   ├── RollingEdgePredict.py
│   └── ...
│
├── src/                          # code developed for this project
│   ├── model.py
│   ├── real_data_pipeline.py
│   ├── simulation.py
│   └── utils.py
│
├── notebooks/                     # notebooks containing experiments run for this project
│   ├── 01_eda.ipynb
│   ├── 02_simulation_parameter_estimation.ipynb
│   ├── 03_simulation_prediction.ipynb
│   ├── 04_model_misspecification.ipynb
│   ├── 05_real_data_application.ipynb
│   └── output/
│       ├── dataframes/
│       ├── figures/
│       └── tables/
│
├── README.md
└── requirements.txt
```

### Repository overview

- **data/** – Dataset of raw interaction counts and additional datasets used for exogenous covariates in the real data application (environmental and demographic data).
- **external/** – Original GNAR-edge implementation. The code for this project builds heavily on this original implementation.
- **src/** – Python modules developed for my project. This includes the modules containing the model extension, the functions needed to run simulation experiments, and the real-data pipeline.
- **notebooks/** – Jupyter notebooks in which the actual experiments and real data applications are run. These notebooks reproduce the results (figures, tables etc) from my project.
- **notebooks/output/** – Automatically generated output from the Jupyter notebook experiments and real data experiments.