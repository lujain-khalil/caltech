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
            split_point = test_metrics.get("split_point", "N/A")
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
                    "Avg Latency (ms)": f"{avg_latency:.2f}" if isinstance(avg_latency, (int, float)) else avg_latency,
                    "P95 Latency (ms)": f"{p95_latency:.2f}" if isinstance(p95_latency, (int, float)) else p95_latency,
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

        # Write to summary file
        summary_file = attempt_path / "summary.txt"
        with open(summary_file, "w") as f:
            f.write(table)

        print(f"Summary saved to {summary_file}")
        print("\nSummary Table:")
        print(table)

        # Generate figures
        generate_figures(attempt_path, raw_data)
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


def generate_figures(attempt_path, raw_data):
    """
    Generate comparison figures for each ablation set and across all experiments.

    Args:
        attempt_path: Path to the attempt directory
        raw_data: Dictionary of experiment data
    """
    # Create figures directory
    figures_dir = attempt_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    print(f"\nGenerating figures in {figures_dir}...")

    # --- Dynamic ablation set construction ---
    ablation_sets = build_ablation_sets(raw_data)
    if not ablation_sets:
        print("No ablation sets could be created (check experiment naming).")
        return

    set_styles = build_set_styles(ablation_sets)

    # Generate figures for each ablation set
    for ablation, exps in sorted(ablation_sets.items()):
        if not exps:
            continue

        print(f"  Ablation set '{ablation}' – {len(exps)} experiments")

        # Figure 1: Accuracy vs Latency Trade-off
        plot_accuracy_latency_tradeoff(figures_dir, ablation, exps)

        # Figure 2: Test Accuracy Comparison
        plot_test_accuracy_comparison(figures_dir, ablation, exps)

        # Figure 3: Latency Metrics Comparison (Avg and P95)
        plot_latency_metrics_comparison(figures_dir, ablation, exps)

        # Figure 4: Training Metrics Evolution (loss and accuracy across epochs)
        plot_training_evolution(figures_dir, ablation, exps)

        # Figure 5: Loss vs Latency Trade-off
        plot_loss_latency_tradeoff(figures_dir, ablation, exps)

    # --- Comparative figures across all ablation sets ---
    plot_all_accuracy_latency_comparison(figures_dir, ablation_sets, set_styles)
    plot_all_test_accuracy_by_set(figures_dir, ablation_sets, set_styles)
    plot_all_latency_comparison(figures_dir, ablation_sets, set_styles)

    # Extra interpretation figures:
    #  - Global Pareto front of (latency, accuracy)
    #  - SLO target vs observed P95 latency
    plot_global_accuracy_latency_pareto(figures_dir, ablation_sets, set_styles)
    plot_slo_vs_p95_latency(figures_dir, ablation_sets, set_styles)

    print("Figure generation complete!")


def plot_accuracy_latency_tradeoff(figures_dir, ablation, experiments):
    """Plot accuracy vs latency trade-off for an ablation set."""
    fig, ax = plt.subplots(figsize=(10, 6))

    exp_names = []
    test_accs = []
    avg_lats = []

    colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

    for i, (exp_name, data) in enumerate(sorted(experiments.items())):
        if data["test_accuracy"] is not None and data["avg_latency"] is not None:
            exp_names.append(exp_name)
            test_accs.append(data["test_accuracy"])
            avg_lats.append(data["avg_latency"])

    if not exp_names:
        plt.close(fig)
        return

    ax.scatter(
        avg_lats,
        test_accs,
        s=200,
        c=colors[: len(avg_lats)],
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
    )

    # Add labels to points
    for i, exp_name in enumerate(exp_names):
        ax.annotate(
            exp_name,
            (avg_lats[i], test_accs[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Average Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Ablation Set {ablation}: Accuracy vs Latency Trade-off",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        figures_dir / f"set_{ablation}_01_accuracy_vs_latency.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved: set_{ablation}_01_accuracy_vs_latency.png")


def plot_test_accuracy_comparison(figures_dir, ablation, experiments):
    """Plot test accuracy comparison for an ablation set."""
    fig, ax = plt.subplots(figsize=(10, 6))

    exp_names = []
    test_accs = []

    for exp_name, data in sorted(experiments.items()):
        if data["test_accuracy"] is not None:
            exp_names.append(exp_name)
            test_accs.append(data["test_accuracy"])

    if not exp_names:
        plt.close(fig)
        return

    colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
    bars = ax.bar(
        range(len(exp_names)),
        test_accs,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, test_accs)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha="right")
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Ablation Set {ablation}: Test Accuracy Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        figures_dir / f"set_{ablation}_02_test_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved: set_{ablation}_02_test_accuracy_comparison.png")


def plot_latency_metrics_comparison(figures_dir, ablation, experiments):
    """Plot average and P95 latency comparison for an ablation set."""
    fig, ax = plt.subplots(figsize=(11, 6))

    exp_names = []
    avg_lats = []
    p95_lats = []

    for exp_name, data in sorted(experiments.items()):
        if data["avg_latency"] is not None and data["p95_latency"] is not None:
            exp_names.append(exp_name)
            avg_lats.append(data["avg_latency"])
            p95_lats.append(data["p95_latency"])

    if not exp_names:
        plt.close(fig)
        return

    x = np.arange(len(exp_names))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        avg_lats,
        width,
        label="Average Latency",
        color="skyblue",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        p95_lats,
        width,
        label="P95 Latency",
        color="lightcoral",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Experiments", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Ablation Set {ablation}: Latency Metrics Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        figures_dir / f"set_{ablation}_03_latency_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved: set_{ablation}_03_latency_metrics.png")


def plot_training_evolution(figures_dir, ablation, experiments):
    """Plot training loss and accuracy evolution across epochs for an ablation set."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    any_data = False
    for idx, (exp_name, data) in enumerate(sorted(experiments.items())):
        train_log = data.get("train_log", [])
        if not train_log:
            continue

        any_data = True
        epochs = [entry.get("epoch", i) for i, entry in enumerate(train_log)]
        losses = [entry.get("loss", np.nan) for entry in train_log]
        accuracies = [entry.get("accuracy", np.nan) for entry in train_log]

        ax1.plot(
            epochs,
            losses,
            marker="o",
            linewidth=2,
            markersize=4,
            label=exp_name,
            color=colors[idx],
            alpha=0.8,
        )
        ax2.plot(
            epochs,
            accuracies,
            marker="s",
            linewidth=2,
            markersize=4,
            label=exp_name,
            color=colors[idx],
            alpha=0.8,
        )

    if not any_data:
        plt.close(fig)
        return

    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Ablation Set {ablation}: Training Loss Evolution",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"Ablation Set {ablation}: Training Accuracy Evolution",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        figures_dir / f"set_{ablation}_04_training_evolution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved: set_{ablation}_04_training_evolution.png")


def plot_loss_latency_tradeoff(figures_dir, ablation, experiments):
    """Plot loss vs latency trade-off for an ablation set."""
    fig, ax = plt.subplots(figsize=(10, 6))

    exp_names = []
    losses = []
    avg_lats = []

    colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

    for i, (exp_name, data) in enumerate(sorted(experiments.items())):
        if data["loss"] is not None and data["avg_latency"] is not None:
            exp_names.append(exp_name)
            losses.append(data["loss"])
            avg_lats.append(data["avg_latency"])

    if not exp_names:
        plt.close(fig)
        return

    ax.scatter(
        avg_lats,
        losses,
        s=200,
        c=colors[: len(avg_lats)],
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
    )

    # Add labels to points
    for i, exp_name in enumerate(exp_names):
        ax.annotate(
            exp_name,
            (avg_lats[i], losses[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Average Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Ablation Set {ablation}: Loss vs Latency Trade-off",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        figures_dir / f"set_{ablation}_05_loss_vs_latency.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved: set_{ablation}_05_loss_vs_latency.png")


def plot_all_accuracy_latency_comparison(figures_dir, ablation_sets, set_styles):
    """Plot accuracy vs latency for all experiments colored by ablation set."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for set_name, exps in sorted(ablation_sets.items()):
        xs = []
        ys = []
        labels = []
        for exp_name, data in sorted(exps.items()):
            if data["test_accuracy"] is not None and data["avg_latency"] is not None:
                xs.append(data["avg_latency"])
                ys.append(data["test_accuracy"])
                labels.append(exp_name)

        if not xs:
            continue

        style = set_styles.get(
            set_name,
            {"color": "#777777", "marker": "o", "label": f"Set {set_name}"},
        )
        ax.scatter(
            xs,
            ys,
            s=220,
            c=[style["color"]],
            marker=style["marker"],
            alpha=0.7,
            edgecolors="black",
            linewidth=2,
            label=style["label"],
        )

        for x, y, label in zip(xs, ys, labels):
            ax.annotate(
                label,
                (x, y),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel("Average Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("All Experiments: Accuracy vs Latency Trade-off", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=11, loc="best")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "all_01_accuracy_vs_latency_all_sets.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: all_01_accuracy_vs_latency_all_sets.png")


def plot_all_test_accuracy_by_set(figures_dir, ablation_sets, set_styles):
    """Plot test accuracy for all experiments grouped by ablation set."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x_pos = 0
    tick_positions = []
    tick_labels = []

    for set_name, exps in sorted(ablation_sets.items()):
        style = set_styles.get(
            set_name,
            {"color": "#777777", "label": f"Set {set_name}"},
        )
        color = style["color"]

        had_any = False
        for exp_name, data in sorted(exps.items()):
            if data["test_accuracy"] is None:
                continue
            had_any = True
            ax.bar(
                x_pos,
                data["test_accuracy"],
                color=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
                width=0.8,
            )
            ax.text(
                x_pos,
                data["test_accuracy"] + 0.5,
                f"{data['test_accuracy']:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
            tick_positions.append(x_pos)
            tick_labels.append(exp_name)
            x_pos += 1

        if had_any:
            x_pos += 0.5  # spacing between sets

    if not tick_positions:
        plt.close(fig)
        return

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("All Experiments: Test Accuracy Comparison by Ablation Set", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=style["color"], edgecolor="black", label=style["label"])
        for set_name, style in sorted(set_styles.items())
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "all_02_test_accuracy_all_experiments.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: all_02_test_accuracy_all_experiments.png")


def plot_all_latency_comparison(figures_dir, ablation_sets, set_styles):
    """Plot average and P95 latency for all experiments grouped by ablation set."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x_pos = 0
    tick_positions = []
    tick_labels = []
    width = 0.35

    all_avg_lats = []
    all_p95_lats = []
    all_x_pos_avg = []
    all_x_pos_p95 = []

    for set_name, exps in sorted(ablation_sets.items()):
        style = set_styles.get(
            set_name,
            {"color": "#777777", "label": f"Set {set_name}"},
        )
        had_any = False
        for exp_name, data in sorted(exps.items()):
            if data["avg_latency"] is None or data["p95_latency"] is None:
                continue

            had_any = True
            all_avg_lats.append(data["avg_latency"])
            all_p95_lats.append(data["p95_latency"])
            all_x_pos_avg.append(x_pos - width / 2)
            all_x_pos_p95.append(x_pos + width / 2)

            tick_positions.append(x_pos)
            tick_labels.append(exp_name)
            x_pos += 1

        if had_any:
            x_pos += 0.5  # spacing between sets

    if not all_avg_lats:
        plt.close(fig)
        return

    # Create bars
    bars1 = ax.bar(
        all_x_pos_avg,
        all_avg_lats,
        width,
        label="Average Latency",
        color="skyblue",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        all_x_pos_p95,
        all_p95_lats,
        width,
        label="P95 Latency",
        color="lightcoral",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title(
        "All Experiments: Latency Metrics Comparison by Ablation Set",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "all_03_latency_all_experiments.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: all_03_latency_all_experiments.png")


def plot_global_accuracy_latency_pareto(figures_dir, ablation_sets, set_styles):
    """
    Plot all experiments in accuracy-latency space and highlight the Pareto front.

    A point is Pareto-optimal if no other point is strictly better in both accuracy and latency.
    """
    # Collect points
    points = []
    for set_name, exps in ablation_sets.items():
        for exp_name, data in exps.items():
            acc = data.get("test_accuracy")
            lat = data.get("avg_latency")
            if acc is None or lat is None:
                continue
            points.append(
                {
                    "exp": exp_name,
                    "set": set_name,
                    "lat": lat,
                    "acc": acc,
                }
            )

    if not points:
        return

    # Compute Pareto front (O(n^2), fine for small numbers of experiments)
    n = len(points)
    is_pareto = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (
                points[j]["lat"] <= points[i]["lat"]
                and points[j]["acc"] >= points[i]["acc"]
                and (
                    points[j]["lat"] < points[i]["lat"]
                    or points[j]["acc"] > points[i]["acc"]
                )
            ):
                is_pareto[i] = False
                break

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot dominated points (faded)
    for p, on_pareto in zip(points, is_pareto):
        style = set_styles.get(
            p["set"],
            {"color": "#777777", "marker": "o", "label": f"Set {p['set']}"},
        )
        if not on_pareto:
            ax.scatter(
                p["lat"],
                p["acc"],
                s=140,
                c=[style["color"]],
                marker=style["marker"],
                alpha=0.3,
                edgecolors="none",
            )

    # Plot Pareto front points (highlighted)
    for p, on_pareto in zip(points, is_pareto):
        if not on_pareto:
            continue
        style = set_styles.get(
            p["set"],
            {"color": "#777777", "marker": "o", "label": f"Set {p['set']}"},
        )
        ax.scatter(
            p["lat"],
            p["acc"],
            s=260,
            c=[style["color"]],
            marker=style["marker"],
            alpha=0.95,
            edgecolors="black",
            linewidth=2,
        )
        ax.annotate(
            p["exp"],
            (p["lat"], p["acc"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("Average Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("All Experiments: Pareto Front (Accuracy vs Latency)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=style["color"], edgecolor="black", label=style["label"])
        for set_name, style in sorted(set_styles.items())
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc="best", title="Ablation Sets")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "all_04_accuracy_latency_pareto.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: all_04_accuracy_latency_pareto.png")


def plot_slo_vs_p95_latency(figures_dir, ablation_sets, set_styles):
    """
    Plot SLO target vs observed P95 latency.

    Points above the diagonal line violate the SLO (p95 > SLO target).
    """
    xs = []
    ys = []
    sets = []
    labels = []

    for set_name, exps in ablation_sets.items():
        for exp_name, data in exps.items():
            slo_ms = data.get("slo_target_ms")
            p95 = data.get("p95_latency")
            if slo_ms is None or p95 is None:
                continue
            xs.append(slo_ms)
            ys.append(p95)
            sets.append(set_name)
            labels.append(exp_name)

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot points grouped by set
    unique_sets = sorted(set(sets))
    for set_name in unique_sets:
        idxs = [i for i, s in enumerate(sets) if s == set_name]
        style = set_styles.get(
            set_name,
            {"color": "#777777", "marker": "o", "label": f"Set {set_name}"},
        )
        x_vals = [xs[i] for i in idxs]
        y_vals = [ys[i] for i in idxs]
        ax.scatter(
            x_vals,
            y_vals,
            s=180,
            c=[style["color"]],
            marker=style["marker"],
            alpha=0.8,
            edgecolors="black",
            linewidth=1.5,
            label=style["label"],
        )

    # Annotate all points (usually a small number, works fine)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )

    # Diagonal line (SLO == observed)
    max_val = max(max(xs), max(ys))
    ax.plot(
        [0, max_val],
        [0, max_val],
        "k--",
        linewidth=1.5,
        label="Ideal (P95 = SLO target)",
    )

    ax.set_xlabel("SLO Target (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Observed P95 Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("All Experiments: SLO Target vs P95 Latency", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "all_05_slo_vs_p95_latency.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: all_05_slo_vs_p95_latency.png")


if __name__ == "__main__":
    # Path to attempt_ directory (adjust as needed)
    attempt_path = Path(__file__).parent / "experiments" / "attempt_6"
    summarize_attempt(attempt_path)