# Extended GNAR-edge Model: A Primate Social Network Application
```text
The code in this repository accompanies my dissertation, "[INSERT TITLE]". In this dissertation, I extend the GNAR-edge model, introduced by Mantziou et al. (2023) to allow for inclusion of exogenous covariates. The extension is motivated by a data set that features a multivariate time series of pairwise interaction counts within a group of baboons. This data set was originally collected and analysed by Gelardi et al. (2020).
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
│   │   ├── BaseEdge.cpython-314.pyc
│   │   ├── BaseEdgeGNAR_edge.cpython-314.pyc
│   │   ├── BaseEdgeGNAR_edge_global.cpython-314.pyc
│   │   ├── RollingEdgePredict.cpython-314.pyc
│   │   ├── edge_baselines.cpython-314.pyc
│   │   ├── edge_graph.cpython-314.pyc
│   │   ├── orchestration_stl.cpython-314.pyc
│   │   ├── run_baseline_demo.cpython-314.pyc
│   │   ├── run_payment_graph.cpython-314.pyc
│   │   └── run_simulated.cpython-314.pyc
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
│   ├── baboon_calibrated_sim_results.csv
│   └── output
│       ├── dataframes
│       │   └── prediction_results_backup.csv
│       ├── figures
│       │   ├── baboon_network_simulation_k13.pdf
│       │   ├── baboon_network_simulation_k13.png
│       │   ├── baboon_scale_prediction_comparison.pdf
│       │   ├── baboon_scale_prediction_comparison.png
│       │   ├── baseline_interaction_comp_er.pdf
│       │   ├── baseline_interaction_comp_er.png
│       │   ├── baseline_interaction_comp_rdp.pdf
│       │   ├── baseline_interaction_comp_rdp.png
│       │   ├── baseline_interaction_comp_sbm.pdf
│       │   ├── baseline_interaction_comp_sbm.png
│       │   ├── correctly_specified_comparison_density01.pdf
│       │   ├── correctly_specified_comparison_density01.png
│       │   ├── correctly_specified_comparison_density04.pdf
│       │   ├── correctly_specified_comparison_density04.png
│       │   ├── correlated_all_params_er.pdf
│       │   ├── correlated_all_params_er.png
│       │   ├── correlated_all_params_k13_er.pdf
│       │   ├── correlated_all_params_k13_er.png
│       │   ├── correlated_all_params_k13_rdp.pdf
│       │   ├── correlated_all_params_k13_rdp.png
│       │   ├── correlated_all_params_k13_sbm.pdf
│       │   ├── correlated_all_params_k13_sbm.png
│       │   ├── correlated_all_params_noint_er.pdf
│       │   ├── correlated_all_params_noint_er.png
│       │   ├── correlated_all_params_noint_rdp.pdf
│       │   ├── correlated_all_params_noint_rdp.png
│       │   ├── correlated_all_params_noint_sbm.pdf
│       │   ├── correlated_all_params_noint_sbm.png
│       │   ├── correlated_all_params_r01_er.pdf
│       │   ├── correlated_all_params_r01_er.png
│       │   ├── correlated_all_params_r01_rdp.pdf
│       │   ├── correlated_all_params_r01_rdp.png
│       │   ├── correlated_all_params_r01_sbm.pdf
│       │   ├── correlated_all_params_r01_sbm.png
│       │   ├── correlated_all_params_rdp.pdf
│       │   ├── correlated_all_params_rdp.png
│       │   ├── correlated_all_params_sbm.pdf
│       │   ├── correlated_all_params_sbm.png
│       │   ├── exogcorr_all_params_er.pdf
│       │   ├── exogcorr_all_params_er.png
│       │   ├── exogcorr_all_params_k13_er.pdf
│       │   ├── exogcorr_all_params_k13_er.png
│       │   ├── exogcorr_all_params_k13_rdp.pdf
│       │   ├── exogcorr_all_params_k13_rdp.png
│       │   ├── exogcorr_all_params_k13_sbm.pdf
│       │   ├── exogcorr_all_params_k13_sbm.png
│       │   ├── exogcorr_all_params_noint_er.pdf
│       │   ├── exogcorr_all_params_noint_er.png
│       │   ├── exogcorr_all_params_noint_rdp.pdf
│       │   ├── exogcorr_all_params_noint_rdp.png
│       │   ├── exogcorr_all_params_noint_sbm.pdf
│       │   ├── exogcorr_all_params_noint_sbm.png
│       │   ├── exogcorr_all_params_r01_er.pdf
│       │   ├── exogcorr_all_params_r01_er.png
│       │   ├── exogcorr_all_params_r01_rdp.pdf
│       │   ├── exogcorr_all_params_r01_rdp.png
│       │   ├── exogcorr_all_params_r01_sbm.pdf
│       │   ├── exogcorr_all_params_r01_sbm.png
│       │   ├── exogcorr_all_params_rdp.pdf
│       │   ├── exogcorr_all_params_rdp.png
│       │   ├── exogcorr_all_params_sbm.pdf
│       │   ├── exogcorr_all_params_sbm.png
│       │   ├── full_model,_l_=_6,_r_=_2_edge_corr_histogram.pdf
│       │   ├── full_model,_l_=_6,_r_=_2_edge_corr_histogram.png
│       │   ├── full_model,_l_=_6,_r_=_2_fitted_vs_observed_ANGELE_FELIPE.pdf
│       │   ├── full_model,_l_=_6,_r_=_2_fitted_vs_observed_ANGELE_FELIPE.png
│       │   ├── full_model,_l_=_6,_r_=_2_mean_residual_acf.pdf
│       │   ├── full_model,_l_=_6,_r_=_2_mean_residual_acf.png
│       │   ├── full_model,_l_=_6,_r_=_2_pooled_residual_histogram.pdf
│       │   ├── full_model,_l_=_6,_r_=_2_pooled_residual_histogram.png
│       │   ├── full_model,_l_=_6,_r_=_2_pooled_residual_qq.pdf
│       │   ├── full_model,_l_=_6,_r_=_2_pooled_residual_qq.png
│       │   ├── full_model,_l_=_6,_r_=_2_residual_boxplot.pdf
│       │   ├── full_model,_l_=_6,_r_=_2_residual_boxplot.png
│       │   ├── full_model,_l_=_6,_r_=_2_residuals_ANGELE_FELIPE.pdf
│       │   ├── full_model,_l_=_6,_r_=_2_residuals_ANGELE_FELIPE.png
│       │   ├── gamma_scale_large_density01.pdf
│       │   ├── gamma_scale_large_density01.png
│       │   ├── gamma_scale_large_density04.pdf
│       │   ├── gamma_scale_large_density04.png
│       │   ├── gamma_scale_moderate_density01.pdf
│       │   ├── gamma_scale_moderate_density01.png
│       │   ├── gamma_scale_moderate_density04.pdf
│       │   ├── gamma_scale_moderate_density04.png
│       │   ├── gamma_scale_small_density01.pdf
│       │   ├── gamma_scale_small_density01.png
│       │   ├── gamma_scale_small_density04.pdf
│       │   ├── gamma_scale_small_density04.png
│       │   ├── heavytail_alpha_er.pdf
│       │   ├── heavytail_alpha_er.png
│       │   ├── heavytail_alpha_k13_er.pdf
│       │   ├── heavytail_alpha_k13_er.png
│       │   ├── heavytail_alpha_k13_rdp.pdf
│       │   ├── heavytail_alpha_k13_rdp.png
│       │   ├── heavytail_alpha_k13_sbm.pdf
│       │   ├── heavytail_alpha_k13_sbm.png
│       │   ├── heavytail_alpha_noint_er.pdf
│       │   ├── heavytail_alpha_noint_er.png
│       │   ├── heavytail_alpha_noint_rdp.pdf
│       │   ├── heavytail_alpha_noint_rdp.png
│       │   ├── heavytail_alpha_noint_sbm.pdf
│       │   ├── heavytail_alpha_noint_sbm.png
│       │   ├── heavytail_alpha_rdp.pdf
│       │   ├── heavytail_alpha_rdp.png
│       │   ├── heavytail_alpha_sbm.pdf
│       │   ├── heavytail_alpha_sbm.png
│       │   ├── heavytail_beta_er.pdf
│       │   ├── heavytail_beta_er.png
│       │   ├── heavytail_beta_k13_er.pdf
│       │   ├── heavytail_beta_k13_er.png
│       │   ├── heavytail_beta_k13_rdp.pdf
│       │   ├── heavytail_beta_k13_rdp.png
│       │   ├── heavytail_beta_k13_sbm.pdf
│       │   ├── heavytail_beta_k13_sbm.png
│       │   ├── heavytail_beta_noint_er.pdf
│       │   ├── heavytail_beta_noint_er.png
│       │   ├── heavytail_beta_noint_rdp.pdf
│       │   ├── heavytail_beta_noint_rdp.png
│       │   ├── heavytail_beta_noint_sbm.pdf
│       │   ├── heavytail_beta_noint_sbm.png
│       │   ├── heavytail_beta_rdp.pdf
│       │   ├── heavytail_beta_rdp.png
│       │   ├── heavytail_beta_sbm.pdf
│       │   ├── heavytail_beta_sbm.png
│       │   ├── heavytail_gamma_er.pdf
│       │   ├── heavytail_gamma_er.png
│       │   ├── heavytail_gamma_k13_er.pdf
│       │   ├── heavytail_gamma_k13_er.png
│       │   ├── heavytail_gamma_k13_rdp.pdf
│       │   ├── heavytail_gamma_k13_rdp.png
│       │   ├── heavytail_gamma_k13_sbm.pdf
│       │   ├── heavytail_gamma_k13_sbm.png
│       │   ├── heavytail_gamma_noint_er.pdf
│       │   ├── heavytail_gamma_noint_er.png
│       │   ├── heavytail_gamma_noint_rdp.pdf
│       │   ├── heavytail_gamma_noint_rdp.png
│       │   ├── heavytail_gamma_noint_sbm.pdf
│       │   ├── heavytail_gamma_noint_sbm.png
│       │   ├── heavytail_gamma_rdp.pdf
│       │   ├── heavytail_gamma_rdp.png
│       │   ├── heavytail_gamma_sbm.pdf
│       │   ├── heavytail_gamma_sbm.png
│       │   ├── heavytail_predictive_rmse.pdf
│       │   ├── heavytail_predictive_rmse.png
│       │   ├── heavytail_predictive_rmse_k13.pdf
│       │   ├── heavytail_predictive_rmse_k13.png
│       │   ├── interaction_comparison_per_parameter_exogcorr05_er.pdf
│       │   ├── interaction_comparison_per_parameter_exogcorr05_er.png
│       │   ├── interaction_comparison_per_parameter_exogcorr05_rdp.pdf
│       │   ├── interaction_comparison_per_parameter_exogcorr05_rdp.png
│       │   ├── interaction_comparison_per_parameter_exogcorr05_sbm.pdf
│       │   ├── interaction_comparison_per_parameter_exogcorr05_sbm.png
│       │   ├── interaction_comparison_per_parameter_rewiring02_er.pdf
│       │   ├── interaction_comparison_per_parameter_rewiring02_er.png
│       │   ├── interaction_comparison_per_parameter_rewiring02_rdp.pdf
│       │   ├── interaction_comparison_per_parameter_rewiring02_rdp.png
│       │   ├── interaction_comparison_per_parameter_rewiring02_sbm.pdf
│       │   ├── interaction_comparison_per_parameter_rewiring02_sbm.png
│       │   ├── interaction_comparison_rewiring02_er.pdf
│       │   ├── interaction_comparison_rewiring02_er.png
│       │   ├── interaction_comparison_rewiring02_rdp.pdf
│       │   ├── interaction_comparison_rewiring02_rdp.png
│       │   ├── interaction_comparison_rewiring02_sbm.pdf
│       │   ├── interaction_comparison_rewiring02_sbm.png
│       │   ├── interaction_comparison_rewiring_full_er.pdf
│       │   ├── interaction_comparison_rewiring_full_er.png
│       │   ├── interaction_comparison_rewiring_full_rdp.pdf
│       │   ├── interaction_comparison_rewiring_full_rdp.png
│       │   ├── interaction_comparison_rewiring_full_sbm.pdf
│       │   ├── interaction_comparison_rewiring_full_sbm.png
│       │   ├── interaction_histogram.pdf
│       │   ├── interactions_temp_over_time.pdf
│       │   ├── interactions_vs_max_temp.pdf
│       │   ├── mean_age_vs_temp_corr.pdf
│       │   ├── prediction_rmse_no_covariates_(regime_4)_density04.pdf
│       │   ├── prediction_rmse_no_covariates_(regime_4)_density04.png
│       │   ├── prediction_rmse_regime_4_density04.pdf
│       │   ├── prediction_rmse_regime_4_density04.png
│       │   ├── prediction_rmse_regime_8_density04.pdf
│       │   ├── prediction_rmse_regime_8_density04.png
│       │   ├── prediction_rmse_regime_9_density04.pdf
│       │   ├── prediction_rmse_regime_9_density04.png
│       │   ├── prediction_rmse_regimes_er_density01.pdf
│       │   ├── prediction_rmse_regimes_er_density01.png
│       │   ├── prediction_rmse_regimes_er_density04.pdf
│       │   ├── prediction_rmse_regimes_er_density04.png
│       │   ├── prediction_rmse_regimes_rdp_density01.pdf
│       │   ├── prediction_rmse_regimes_rdp_density01.png
│       │   ├── prediction_rmse_regimes_rdp_density04.pdf
│       │   ├── prediction_rmse_regimes_rdp_density04.png
│       │   ├── prediction_rmse_regimes_sbm_density01.pdf
│       │   ├── prediction_rmse_regimes_sbm_density01.png
│       │   ├── prediction_rmse_regimes_sbm_density04.pdf
│       │   ├── prediction_rmse_regimes_sbm_density04.png
│       │   ├── prediction_rmse_with_covariates_interaction_(regime_9)_density04.pdf
│       │   ├── prediction_rmse_with_covariates_interaction_(regime_9)_density04.png
│       │   ├── prediction_rmse_with_covariates_no_interaction_(regime_8)_density04.pdf
│       │   ├── prediction_rmse_with_covariates_no_interaction_(regime_8)_density04.png
│       │   ├── rewiring_gamma_er.pdf
│       │   ├── rewiring_gamma_er.png
│       │   ├── rewiring_gamma_k13_er.pdf
│       │   ├── rewiring_gamma_k13_er.png
│       │   ├── rewiring_gamma_k13_rdp.pdf
│       │   ├── rewiring_gamma_k13_rdp.png
│       │   ├── rewiring_gamma_k13_sbm.pdf
│       │   ├── rewiring_gamma_k13_sbm.png
│       │   ├── rewiring_gamma_noint_er.pdf
│       │   ├── rewiring_gamma_noint_er.png
│       │   ├── rewiring_gamma_noint_rdp.pdf
│       │   ├── rewiring_gamma_noint_rdp.png
│       │   ├── rewiring_gamma_noint_sbm.pdf
│       │   ├── rewiring_gamma_noint_sbm.png
│       │   ├── rewiring_gamma_rdp.pdf
│       │   ├── rewiring_gamma_rdp.png
│       │   ├── rewiring_gamma_sbm.pdf
│       │   ├── rewiring_gamma_sbm.png
│       │   ├── rewiring_overall_er.pdf
│       │   ├── rewiring_overall_er.png
│       │   ├── rewiring_overall_k13_er.pdf
│       │   ├── rewiring_overall_k13_er.png
│       │   ├── rewiring_overall_k13_rdp.pdf
│       │   ├── rewiring_overall_k13_rdp.png
│       │   ├── rewiring_overall_k13_sbm.pdf
│       │   ├── rewiring_overall_k13_sbm.png
│       │   ├── rewiring_overall_noint_er.pdf
│       │   ├── rewiring_overall_noint_er.png
│       │   ├── rewiring_overall_noint_rdp.pdf
│       │   ├── rewiring_overall_noint_rdp.png
│       │   ├── rewiring_overall_noint_sbm.pdf
│       │   ├── rewiring_overall_noint_sbm.png
│       │   ├── rewiring_overall_rdp.pdf
│       │   ├── rewiring_overall_rdp.png
│       │   ├── rewiring_overall_sbm.pdf
│       │   ├── rewiring_overall_sbm.png
│       │   ├── rmse_advantage_full_model_vs_no_covariates.pdf
│       │   ├── rmse_advantage_full_model_vs_no_covariates.png
│       │   ├── rmse_by_lag_stage_age_diff_mean_age.pdf
│       │   ├── rmse_by_lag_stage_age_diff_mean_age.png
│       │   ├── rmse_by_lag_stage_age_diff_mean_age_same_sex.pdf
│       │   ├── rmse_by_lag_stage_age_diff_mean_age_same_sex.png
│       │   ├── rmse_by_lag_stage_age_diff_mean_age_temp_no_interaction.pdf
│       │   ├── rmse_by_lag_stage_age_diff_mean_age_temp_no_interaction.png
│       │   ├── rmse_by_lag_stage_age_diff_mean_age_temp_same_sex_no_interaction.pdf
│       │   ├── rmse_by_lag_stage_age_diff_mean_age_temp_same_sex_no_interaction.png
│       │   ├── rmse_by_lag_stage_age_diff_mean_age_temp_same_sex_with_interaction.pdf
│       │   ├── rmse_by_lag_stage_age_diff_mean_age_temp_same_sex_with_interaction.png
│       │   ├── rmse_by_lag_stage_age_diff_mean_age_temp_with_interaction.pdf
│       │   ├── rmse_by_lag_stage_age_diff_mean_age_temp_with_interaction.png
│       │   ├── rmse_by_lag_stage_age_diff_only.pdf
│       │   ├── rmse_by_lag_stage_age_diff_only.png
│       │   ├── rmse_by_lag_stage_age_diff_same_sex.pdf
│       │   ├── rmse_by_lag_stage_age_diff_same_sex.png
│       │   ├── rmse_by_lag_stage_age_diff_temp.pdf
│       │   ├── rmse_by_lag_stage_age_diff_temp.png
│       │   ├── rmse_by_lag_stage_age_diff_temp_same_sex.pdf
│       │   ├── rmse_by_lag_stage_age_diff_temp_same_sex.png
│       │   ├── rmse_by_lag_stage_all_configs.pdf
│       │   ├── rmse_by_lag_stage_all_configs.png
│       │   ├── rmse_by_lag_stage_mean_age_only.pdf
│       │   ├── rmse_by_lag_stage_mean_age_only.png
│       │   ├── rmse_by_lag_stage_mean_age_same_sex.pdf
│       │   ├── rmse_by_lag_stage_mean_age_same_sex.png
│       │   ├── rmse_by_lag_stage_mean_age_temp.pdf
│       │   ├── rmse_by_lag_stage_mean_age_temp.png
│       │   ├── rmse_by_lag_stage_mean_age_temp_same_sex.pdf
│       │   ├── rmse_by_lag_stage_mean_age_temp_same_sex.png
│       │   ├── rmse_by_lag_stage_no_covariates.pdf
│       │   ├── rmse_by_lag_stage_no_covariates.png
│       │   ├── rmse_by_lag_stage_same_sex_only.pdf
│       │   ├── rmse_by_lag_stage_same_sex_only.png
│       │   ├── rmse_by_lag_stage_temp_only.pdf
│       │   ├── rmse_by_lag_stage_temp_only.png
│       │   ├── rmse_by_lag_stage_temp_same_sex.pdf
│       │   ├── rmse_by_lag_stage_temp_same_sex.png
│       │   ├── t_=_28_edge_corr_histogram.pdf
│       │   ├── t_=_28_edge_corr_histogram.png
│       │   ├── t_=_28_fitted_vs_observed_1_2.pdf
│       │   ├── t_=_28_fitted_vs_observed_1_2.png
│       │   ├── t_=_28_mean_residual_acf.pdf
│       │   ├── t_=_28_mean_residual_acf.png
│       │   ├── t_=_28_pooled_residual_histogram.pdf
│       │   ├── t_=_28_pooled_residual_histogram.png
│       │   ├── t_=_28_pooled_residual_qq.pdf
│       │   ├── t_=_28_pooled_residual_qq.png
│       │   ├── t_=_28_residual_boxplot.pdf
│       │   ├── t_=_28_residual_boxplot.png
│       │   ├── t_=_28_residuals_1_2.pdf
│       │   └── t_=_28_residuals_1_2.png
│       └── tables
│           ├── baboon_accuracy_er.tex
│           ├── baboon_accuracy_rdp.tex
│           ├── baboon_accuracy_sbm.tex
│           ├── baboon_regime_table.tex
│           ├── covariate_regime_table.tex
│           ├── daily_network_similarity_postprocessing.csv
│           ├── full_model,_l_=_6,_r_=_2_diagnostics_table.csv
│           ├── full_model,_l_=_6,_r_=_2_edge_corr_df.csv
│           ├── full_model,_l_=_6,_r_=_2_edge_corr_summary.csv
│           ├── full_model,_l_=_6,_r_=_2_outlier_summary.csv
│           ├── gamma_scale_large.csv
│           ├── gamma_scale_large.tex
│           ├── gamma_scale_moderate.csv
│           ├── gamma_scale_moderate.tex
│           ├── gamma_scale_small.csv
│           ├── gamma_scale_small.tex
│           ├── gamma_scale_table.csv
│           ├── gamma_scale_table.tex
│           ├── gnar_regime_table.tex
│           ├── heavytail_predictive_rmse_summary.csv
│           ├── heavytail_predictive_rmse_summary_k13.csv
│           ├── interaction_comparison_per_parameter_exogcorr05_sbm.csv
│           ├── interaction_comparison_per_parameter_rewiring02_sbm.csv
│           ├── interaction_comparison_rewiring02_sbm.csv
│           ├── median_rmse_fixed_dgp.csv
│           ├── median_rmse_fixed_dgp.tex
│           ├── parameter_regime_table.tex
│           ├── prediction_results_backup.csv
│           ├── real_data_baseline_arima.csv
│           ├── real_data_baseline_arima.tex
│           ├── real_data_baseline_comparison.tex.csv
│           ├── real_data_best_lag_stage_by_config.csv
│           ├── real_data_lag6_stage2_config_comparison.csv
│           ├── real_data_models.tex
│           ├── real_data_overall_lag_stage_config_ranking.csv
│           ├── real_data_sum_of_squares_decomp.csv
│           ├── real_data_threshold_neighbour_density.csv
│           ├── real_data_timing.tex
│           ├── real_data_winner_loser.csv
│           ├── regime_1_table.tex
│           ├── regime_2_table.tex
│           ├── regime_3_table.tex
│           ├── regime_4_table.tex
│           ├── regime_5_table.tex
│           ├── regime_6_table.tex
│           ├── regime_7_table.tex
│           ├── regime_8_table.tex
│           ├── regime_9_table.tex
│           ├── runtime_comparison_isolated.csv
│           ├── sim_moderate_timing.tex
│           ├── simulation_baseline.csv
│           ├── summary_table_regimes.csv
│           ├── summary_table_regimes.tex
│           ├── t_=_28_diagnostics_table.csv
│           ├── t_=_28_edge_corr_df.csv
│           ├── t_=_28_edge_corr_summary.csv
│           └── t_=_28_outlier_summary.csv
├── requirements.txt
└── src
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-314.pyc
    │   ├── diagnostics.cpython-313.pyc
    │   ├── diagnostics.cpython-314.pyc
    │   ├── model.cpython-313.pyc
    │   ├── model.cpython-314.pyc
    │   ├── real_data_pipeline.cpython-313.pyc
    │   ├── real_data_pipeline.cpython-314.pyc
    │   ├── simulation.cpython-313.pyc
    │   └── simulation.cpython-314.pyc
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

## Acknowledgements

The original GNAR-edge implementation by Mantziou et al. (2023) was developed for R, and can be found here: https://github.com/mantziou/GNAR-edge-model. A python implementation of the GNAR-edge model was developed by Tian Xie, and can be found here: https://github.com/naive4E4A55/gnar-edge. The code for my project relies heavily on Tian Xie's python implementation. 

## Citation

If referencing this work, please cite:
Lucy Cheffins. [2026]. "[Thesis Title]". [University of Oxford].