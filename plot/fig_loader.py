import json
from pathlib import Path
from typing import Dict, Any, List, Iterable

import pandas as pd

from fig_settings import EXPERIMENT_ROOTS


def classify_experiment(name: str) -> str:
    """
    Map folder name to a family label.

    We treat:
      - A_slo_*              -> "slo"
      - B_net_*              -> "net"
      - C_data_*             -> "dataset"
      - D_edge_*             -> "edge"
      - cifar10/fmnist/mnist -> "baseline"
      - default              -> "default"
      - anything else        -> "other"
    """
    if name.startswith("A_slo_"):
        return "slo"
    if name.startswith("B_net_"):
        return "net"
    if name.startswith("C_data_"):
        return "dataset"
    if name.startswith("D_edge_"):
        return "edge"
    if name in {"cifar10", "fmnist", "mnist", "cifar100"}:
        return "baseline"
    if name == "default":
        return "default"
    return "other"


def _iter_experiment_dirs(roots: Iterable[Path]):
    """Yield all subdirectories under the given roots."""
    for root in roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir():
                yield child


def load_experiments(
    roots: Iterable[Path] = EXPERIMENT_ROOTS,
) -> Dict[str, Dict[str, Any]]:
    """
    Scan all roots for subdirectories containing experiment results.


      - require config.json, train_log.json, test_metrics.json

    For baseline_models/<dataset>/:
      - require test_metrics.json
      - config.json is optional (we synthesise it if missing)
      - train_log.json is optional (ignored if missing)
    """
    experiments: Dict[str, Dict[str, Any]] = {}

    any_root_exists = any(root.exists() for root in roots)
    if not any_root_exists:
        raise RuntimeError(f"None of EXPERIMENT_ROOTS exist: {list(roots)}")

    for child in _iter_experiment_dirs(roots):
        parent_name = child.parent.name
        exp_name = child.name

        cfg_path = child / "config.json"
        # train_path = child / "train_log.json"
        test_path = child / "test_metrics.json"

        is_baseline = parent_name == "baseline_models"

        # ---------- baselines ----------
        if is_baseline:
            if not test_path.exists():
                # no metrics → nothing to plot
                continue

            # config: synthesise if missing
            if cfg_path.exists():
                with cfg_path.open("r") as f:
                    cfg = json.load(f)
            else:
                cfg = {"dataset": exp_name}

            # train log optional
            # if train_path.exists():
            #     with train_path.open("r") as f:
            #         train_log = json.load(f)
            # else:
            #     train_log = {}

            with test_path.open("r") as f:
                test_metrics = json.load(f)

        # ---------- normal experiments ----------
        else:
            if not (cfg_path.exists()  and test_path.exists()):
                continue

            with cfg_path.open("r") as f:
                cfg = json.load(f)
            # with train_path.open("r") as f:
            #     train_log = json.load(f)
            with test_path.open("r") as f:
                test_metrics = json.load(f)

        experiments[exp_name] = {
            "config": cfg,
            # "train_log": train_log,      
            "test_metrics": test_metrics,
        }

    if not experiments:
        raise RuntimeError(
            f"No complete experiments found under roots: {list(roots)}"
        )

    return experiments

def experiments_to_dataframe(
    experiments: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Flatten config + test_metrics into a DataFrame.
    One row per experiment.

    All accuracies come from test_metrics (exit-aware).

    Latency handling:
      - For non-baselines: use avg_latency_ms / p95_latency_ms (your attempt_6 logs).
      - For baselines: prefer cloud_avg_latency_ms / cloud_p95_latency_ms.
      - If those are missing, fall back to edge_* or bare avg_latency / p95_latency.
    """
    rows: List[dict] = []

    for name, data in experiments.items():
        cfg = data["config"]
        tm = data["test_metrics"]

        family = classify_experiment(name)

        exit_rates = tm.get("exit_rates", {}) or {}
        num_samples = tm.get("num_samples", 1) or 1

        # ---------- latency selection ----------
        if family == "baseline":
            avg_latency_ms = (
                tm.get("avg_latency_ms")
                or tm.get("cloud_avg_latency_ms")
                or tm.get("edge_avg_latency_ms")
                or tm.get("avg_latency")
            )
            p95_latency_ms = (
                tm.get("p95_latency_ms")
                or tm.get("cloud_p95_latency_ms")
                or tm.get("edge_p95_latency_ms")
                or tm.get("p95_latency")
            )
        else:
            avg_latency_ms = tm.get("avg_latency_ms") or tm.get("avg_latency")
            p95_latency_ms = tm.get("p95_latency_ms") or tm.get("p95_latency")

        row = {
            "exp_name": name,
            "family": family,

            # config
            "dataset": cfg.get("dataset"),
            "slo_target": cfg.get("slo_target"),
            "avg_rtt": cfg.get("avg_rtt"),
            "avg_bw": cfg.get("avg_bw"),
            "lambda_lat": cfg.get("lambda_lat"),
            "mu_slo": cfg.get("mu_slo"),
            "edge_slowdown": cfg.get("edge_slowdown"),

            # test metrics
            "test_accuracy": tm.get("test_accuracy"),
            "test_avg_latency_ms": avg_latency_ms,
            "test_p95_latency_ms": p95_latency_ms,
            "split_point": tm.get("split_point"),
            "split_block": tm.get("split_block"),
            "net_avg_bw_mbps": tm.get("net_avg_bw_mbps"),
            "net_avg_rtt_ms": tm.get("net_avg_rtt_ms"),
            "num_samples": num_samples,

            # exits
            "exit1_frac": exit_rates.get("1", 0.0),
            "exit2_frac": exit_rates.get("2", 0.0),
            "exit3_frac": exit_rates.get("3", 0.0),
            "exit_final_frac": exit_rates.get("final", 0.0),
        }
        rows.append(row)

    return pd.DataFrame(rows)



