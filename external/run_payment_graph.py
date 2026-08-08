"""
run_payment_graph.py  —— Minimal example 2: read ONS PaymentIndustryGraph data and run GNAR-edge (MoM)

Notes:
  - This subfolder is "pure algorithm" (edge_graph / BaseEdge / BaseEdgeGNAR_edge / RollingEdgePredict).
  - Reading the ONS data (IndustryPaymentGraph / ThresholdStaticGraphTS and their dependencies ipg_stl / debug)
    lives in the **parent directory**. This file adds the parent directory to sys.path and imports them — the ONS ipg/TSG is
    just "one source" of an EdgeGraph, adapted into the algorithm via ArrayEdgeGraph.from_tsg(...).
  - Place the ONS workbook (*.xlsx, with "ukindustry" in the filename) into the parent directory.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)        # subfolder takes priority (ensures the pure algorithm in this directory is used)
sys.path.append(PARENT)         # parent directory provides IndustryPaymentGraph / ThresholdStaticGraphTS

import numpy as np

import IndustryPaymentGraph as IPG
from ThresholdStaticGraphTS import ThresholdStaticGraphTS
from edge_graph import ArrayEdgeGraph
from BaseEdgeGNAR_edge import GNAREdgeLearner
from RollingEdgePredict import RollingEdgePredict


def find_excel() -> str:
    for f in sorted(os.listdir(PARENT)):
        lf = f.lower()
        if lf.endswith(".xlsx") and "ukindustry" in lf and "example" not in lf:
            return os.path.join(PARENT, f)
    raise FileNotFoundError(f"Could not find an ONS workbook in {PARENT} (*.xlsx, filename containing 'ukindustry').")


def build_mom_edge_graph(excel_path: str, *, vmin: float = -1000.0, nmin: float = 0.0) -> ArrayEdgeGraph:
    ipg = IPG.load_workbook_to_ipg(
        path=excel_path, sheets=None, years=range(2015, 2030),
        drop_unclassified=True, na_policy="zero")
    ipg.normalize_by_month_days(inplace=True)
    _, ipg_mom = ipg.to_growth_graphs()            # log MoM growth (not YoY)
    tsg = ThresholdStaticGraphTS(ipg=ipg_mom, periods=list(ipg_mom.periods),
                                 vmin=vmin, nmin=nmin, rule="all_time")
    return ArrayEdgeGraph.from_tsg(tsg)            # ONS ipg/TSG -> EdgeGraph


def main():
    excel = find_excel()
    print("ONS workbook:", os.path.basename(excel))
    g = build_mom_edge_graph(excel)
    print(f"ONS MoM EdgeGraph: E={g.n_edges} K={g.n_nodes} T={g.n_times}")

    roller = RollingEdgePredict(
        graph=g,
        learner_cls=GNAREdgeLearner,
        train_periods=g.time_labels[:-3],
        val_periods=g.time_labels[-3:],
        learner_kwargs=dict(L=2, stages_per_lag=[[1], [1]],
                            use_ols=True, device="cpu", verbose_fit=False),
        fit_kwargs=dict(use_ols=True, ols_rcond=1e-8, verbose=False),
        graph_builder=None,            # fixed graph: A is unchanged, not rebuilt
        strict_next_match=True,
        verbose=True,
    ).run()

    print(roller.summary().to_string(index=False))


if __name__ == "__main__":
    main()
