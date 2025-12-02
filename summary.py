import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def format_table(data, headers):
    """Format a list of dicts into a pretty text table.

    Args:
        data: List of dictionaries with data for each row
        headers: List of column headers

    Returns:
        Formatted table as a string
    """
    # Calculate column widths
    col_widths = {}
    for header in headers:
        col_widths[header] = len(str(header))
        for row in data:
            col_widths[header] = max(col_widths[header], len(str(row[header])))

    # Build table
    lines = []

    # Header row
    header_row = " | ".join(f"{h:{col_widths[h]}}" for h in headers)
    lines.append(header_row)
    lines.append("-" * len(header_row))

    # Data rows
    for row in data:
        row_line = " | ".join(f"{str(row[h]):{col_widths[h]}}" for h in headers)
        lines.append(row_line)

    return "\n".join(lines)


def summarize_baseline_models(baseline_models_dir):
    """
    Summarize all baseline models in the baseline_models directory.

    Args:
        baseline_models_dir: Path to the baseline_models directory

    Returns:
        Tuple of (baseline_data list, formatted table string)
    """
    baseline_path = Path(baseline_models_dir)
    baseline_data = []

    # Get all dataset subdirectories in baseline_models folder
    for dataset_dir in sorted(baseline_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        # Construct paths to the required files
        train_log_file = dataset_dir / "train_log.json"
        test_metrics_file = dataset_dir / "test_metrics.json"

        # Check if required files exist
        if not (train_log_file.exists() and test_metrics_file.exists()):
            print(f"Warning: Skipping baseline {dataset_name} - missing required files")
            continue

        try:
            # Load train_log.json
            with open(train_log_file, "r") as f:
                train_log = json.load(f)

            # Load test_metrics.json
            with open(test_metrics_file, "r") as f:
                test_metrics = json.load(f)

            # Extract data from train_log.json (last epoch)
            last_epoch = train_log[-1] if train_log else {}
            train_accuracy = last_epoch.get("accuracy", "N/A")
            loss = last_epoch.get("loss", "N/A")

            # Extract data from test_metrics.json
            test_accuracy = test_metrics.get("test_accuracy", "N/A")
            edge_avg_latency = test_metrics.get("edge_avg_latency_ms", "N/A")
            edge_p95_latency = test_metrics.get("edge_p95_latency_ms", "N/A")
            cloud_avg_latency = test_metrics.get("cloud_avg_latency_ms", "N/A")
            cloud_p95_latency = test_metrics.get("cloud_p95_latency_ms", "N/A")
            num_samples = test_metrics.get("num_samples", "N/A")
            net_avg_bw_mbps = test_metrics.get("net_avg_bw_mbps", "N/A")
            net_avg_rtt_ms = test_metrics.get("net_avg_rtt_ms", "N/A")

            # Add to results
            baseline_data.append(
                {
                    "Dataset": dataset_name.upper(),
                    "Train Accuracy": f"{train_accuracy:.2f}" if isinstance(train_accuracy, (int, float)) else train_accuracy,
                    "Final Loss": f"{loss:.4f}" if isinstance(loss, (int, float)) else loss,
                    "Test Accuracy": f"{test_accuracy:.2f}" if isinstance(test_accuracy, (int, float)) else test_accuracy,
                    "Edge Avg Latency (ms)": f"{edge_avg_latency:.2f}" if isinstance(edge_avg_latency, (int, float)) else edge_avg_latency,
                    "Edge P95 Latency (ms)": f"{edge_p95_latency:.2f}" if isinstance(edge_p95_latency, (int, float)) else edge_p95_latency,
                    "Cloud Avg Latency (ms)": f"{cloud_avg_latency:.2f}" if isinstance(cloud_avg_latency, (int, float)) else cloud_avg_latency,
                    "Cloud P95 Latency (ms)": f"{cloud_p95_latency:.2f}" if isinstance(cloud_p95_latency, (int, float)) else cloud_p95_latency,
                    "Num Samples": num_samples,
                    "Avg BW (Mbps)": net_avg_bw_mbps,
                    "Avg RTT (ms)": net_avg_rtt_ms,
                }
            )

        except Exception as e:
            print(f"Error processing baseline {dataset_name}: {e}")
            continue

    # Format table if we have data
    if baseline_data:
        headers = list(baseline_data[0].keys())
        table = format_table(baseline_data, headers)
        return baseline_data, table
    else:
        return [], ""


def summarize_attempt(attempt_dir):
    """
    Summarize all experiments in the given attempt directory.

    Args:
        attempt_dir: Path to the attempt directory (e.g., experiments/attempt_1)
    """
    attempt_path = Path(attempt_dir)

    # Collect data from all experiment subdirectories
    experiments_data = []
    raw_data = {}  # Store raw numeric data for plotting

    # Get all subdirectories in the attempt folder
    for exp_dir in sorted(attempt_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name

        # Construct paths to the required files
        config_file = exp_dir / "config.json"
        train_log_file = exp_dir / "train_log.json"
        test_metrics_file = exp_dir / "test_metrics.json"

        # Check if all required files exist
        if not (config_file.exists() and train_log_file.exists() and test_metrics_file.exists()):
            print(f"Warning: Skipping {exp_name} - missing required files")
            continue

        try:
            # Load config.json
            with open(config_file, "r") as f:
                config = json.load(f)

            # Load train_log.json
            with open(train_log_file, "r") as f:
                train_log = json.load(f)

            # Load test_metrics.json
            with open(test_metrics_file, "r") as f:
                test_metrics = json.load(f)

            # Extract data from config.json
            dataset = config.get("dataset", "N/A")
            slo_target = config.get("slo_target", "N/A")
            avg_rtt = config.get("avg_rtt", "N/A")
            avg_bw = config.get("avg_bw", "N/A")
            mu_slo = config.get("mu_slo", "N/A")  # Extract mu parameter

            # Convert slo_target to ms
            slo_target_ms = slo_target * 1000 if isinstance(slo_target, (int, float)) else "N/A"

            # Extract data from train_log.json (last epoch)
            last_epoch = train_log[-1] if train_log else {}
            train_accuracy = last_epoch.get("accuracy", "N/A")
            train_latency_ms = last_epoch.get("latency_ms", "N/A")
            exit_probs = last_epoch.get("exit_probs", [])
            exit_prob_0 = exit_probs[0] if len(exit_probs) > 0 else "N/A"
            exit_prob_1 = exit_probs[1] if len(exit_probs) > 1 else "N/A"
            loss = last_epoch.get("loss", "N/A")

            # Extract data from test_metrics.json
            test_accuracy = test_metrics.get("test_accuracy", "N/A")
            avg_latency = test_metrics.get("avg_latency_ms", "N/A")
            p95_latency = test_metrics.get("p95_latency_ms", "N/A")
            split_point = test_metrics.get("split_block", "N/A")
            exit_distribution = test_metrics.get("exit_distribution", {})
            # Extract exit distributions for all heads dynamically
            exit_dist_str = ", ".join(
                f"Exit {k}: {int(v):>5}" 
                for k, v in sorted(exit_distribution.items())
                if k != "final"
            )
            exit_final = exit_distribution.get("final", "N/A")

            # Add to results (for text summary)
            experiments_data.append(
                {
                    "Experiment": exp_name,
                    "Dataset": dataset,
                    "SLO Target (ms)": f"{slo_target_ms:.1f}" if isinstance(slo_target_ms, float) else slo_target_ms,
                    "Avg RTT": avg_rtt,
                    "Avg BW": avg_bw,
                    "Mu (SLO Penalty)": f"{mu_slo:.2f}" if isinstance(mu_slo, (int, float)) else mu_slo,
                    "Train Accuracy": f"{train_accuracy:.2f}" if isinstance(train_accuracy, (int, float)) else train_accuracy,
                    "Train Latency (ms)": f"{train_latency_ms:.2f}" if isinstance(train_latency_ms, (int, float)) else train_latency_ms,
                    "Final Loss": f"{loss:.4f}" if isinstance(loss, (int, float)) else loss,
                    "Test Accuracy": f"{test_accuracy:.2f}" if isinstance(test_accuracy, (int, float)) else test_accuracy,
                    "Test Avg Latency (ms)": f"{avg_latency:.2f}" if isinstance(avg_latency, (int, float)) else avg_latency,
                    "Test P95 Latency (ms)": f"{p95_latency:.2f}" if isinstance(p95_latency, (int, float)) else p95_latency,
                    "Exit Distribution": exit_dist_str,
                    "Exit Head Final": exit_final,
                    "Split Point": split_point,
                }
            )

            # Store raw numeric data for plotting
            def _to_float(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return None

            def _to_int(x):
                try:
                    return int(x)
                except (TypeError, ValueError):
                    return None

            raw_data[exp_name] = {
                "dataset": dataset,
                "slo_target_ms": _to_float(slo_target_ms),
                "avg_rtt": _to_float(avg_rtt),
                "avg_bw": _to_float(avg_bw),
                "mu_slo": _to_float(mu_slo),
                "train_accuracy": _to_float(train_accuracy),
                "train_latency_ms": _to_float(train_latency_ms),
                "exit_prob_0": _to_float(exit_prob_0),
                "exit_prob_1": _to_float(exit_prob_1),
                "loss": _to_float(loss),
                "test_accuracy": _to_float(test_accuracy),
                "avg_latency": _to_float(avg_latency),
                "p95_latency": _to_float(p95_latency),
                "split_point": _to_int(split_point),
                "exit_distribution": exit_distribution,
                "exit_final": _to_int(exit_final),
                "train_log": train_log,
            }

        except Exception as e:
            print(f"Error processing {exp_name}: {e}")
            continue

    # Generate table using manual formatting
    if experiments_data:
        # Get headers from the first entry
        headers = list(experiments_data[0].keys())

        # Format table
        table = format_table(experiments_data, headers)

        # Generate baseline models table
        baseline_models_path = attempt_path.parent / "baseline_models"
        baseline_table = ""
        if baseline_models_path.exists():
            _, baseline_table = summarize_baseline_models(baseline_models_path)

        # Write to summary file
        summary_file = attempt_path / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENT RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(table)
            
            if baseline_table:
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("BASELINE MODELS\n")
                f.write("=" * 80 + "\n\n")
                f.write(baseline_table)

        print(f"Summary saved to {summary_file}")
        print("\nExperiment Results Table:")
        print(table)
        
        if baseline_table:
            print("\nBaseline Models Table:")
            print(baseline_table)
    else:
        print("No valid experiments found to summarize")


def build_ablation_sets(raw_data):
    """
    Group experiments into ablation sets dynamically based on name prefix.

    Convention:
        - Experiments are named like 'A_slo_100', 'B_net_moderate', 'E_new_thing', etc.
        - The token before the first underscore is treated as the ablation set ID.
        - Experiments without an underscore are grouped into a 'misc' set.
        - If a 'default' experiment exists, it is added as '<SET>_baseline' to every set.
    """
    ablation_sets = defaultdict(dict)

    for exp_name, data in raw_data.items():
        if exp_name == "default":
            continue

        if "_" in exp_name:
            set_name = exp_name.split("_", 1)[0]
        else:
            set_name = "misc"

        ablation_sets[set_name][exp_name] = data

    # Optionally add the default/baseline run to each set
    if "default" in raw_data:
        baseline_data = raw_data["default"]
        for set_name in ablation_sets.keys():
            baseline_name = f"{set_name}_baseline"
            if baseline_name not in ablation_sets[set_name]:
                ablation_sets[set_name][baseline_name] = baseline_data

    return dict(ablation_sets)


def build_set_styles(ablation_sets):
    """
    Assign a color and marker to each ablation set for consistent plotting.
    """
    set_names = sorted(ablation_sets.keys())
    if not set_names:
        return {}

    # Use tab10 for distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(set_names), 1)))
    markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]

    styles = {}
    for idx, set_name in enumerate(set_names):
        styles[set_name] = {
            "color": colors[idx % len(colors)],
            "marker": markers[idx % len(markers)],
            "label": f"Set {set_name}",
        }
    return styles

if __name__ == "__main__":
    # Path to attempt_ directory (adjust as needed)
    attempt_path = Path(__file__).parent / "experiments" / "attempt_11"
    summarize_attempt(attempt_path)