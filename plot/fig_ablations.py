import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fig_settings import (
    ABLATION_FIG_DIR,
    EXIT_COLOURS,
    DATASET_BASELINES,
    COLOR_LAT_AVG,
    COLOR_LAT_P95,
    COLOR_ACC_BAR,
    COLOR_ACC_BASELINE,
    COLOR_ACC_OURS,
    FAMILY_COLOURS,
)
from fig_loader import classify_experiment

def _family_colour(family: str) -> str:
    return FAMILY_COLOURS.get(family, "#888888")


# ------------------------- 1. SLO ablation -----------------------------

def plot_slo_ablation(df: pd.DataFrame) -> None:
    """
    default + A_slo_*:
    Figure 1: top: SLO target vs (avg, p95) latency, bottom: SLO target vs accuracy.
    Figure 2: SLO target vs split_block.
    """
    slo_df = df[df["family"] == "slo"].copy()
    default_df = df[df["exp_name"] == "default"].copy()

    if slo_df.empty and default_df.empty:
        print("[ablations] no default / A_slo_*; skipping SLO ablation")
        return

    slo_df = slo_df.sort_values("slo_target")
    slo_ms = slo_df["slo_target"] * 1000.0

    # ==========================================================
    # Figure 1: latency (avg + p95) + accuracy vs SLO
    # ==========================================================
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # ----- top: latency -----
    ax0 = axes[0]
    if not slo_df.empty:
        ax0.plot(
            slo_ms,
            slo_df["test_avg_latency_ms"],
            marker="o",
            linestyle="-",
            color=COLOR_LAT_AVG,
            label="avg latency",
        )
        ax0.plot(
            slo_ms,
            slo_df["test_p95_latency_ms"],
            marker="s",
            linestyle="--",
            color=COLOR_LAT_P95,
            label="p95 latency",
        )
        if not slo_ms.isna().all():
            min_slo = slo_ms.min()
            max_slo = slo_ms.max()
            ax0.plot(
                [min_slo, max_slo],
                [min_slo, max_slo],
                linestyle=":",
                color="#777777",
                label="y = SLO",
            )

    if not default_df.empty and pd.notna(default_df.iloc[0]["slo_target"]):
        d = default_df.iloc[0]
        d_slo_ms = d["slo_target"] * 1000.0
        ax0.scatter(
            [d_slo_ms],
            [d["test_avg_latency_ms"]],
            color="#000000",
            marker="*",
            s=120,
            label="default avg",
            zorder=5,
        )
        if pd.notna(d["test_p95_latency_ms"]):
            ax0.scatter(
                [d_slo_ms],
                [d["test_p95_latency_ms"]],
                color="#000000",
                marker="x",
                s=80,
                label="default p95",
                zorder=5,
            )

    ax0.set_ylabel("Latency (ms)")
    ax0.set_title("Effects of different SLOs – latency")
    ax0.legend()

    # ----- bottom: accuracy -----
    ax1 = axes[1]
    if not slo_df.empty:
        ax1.plot(
            slo_ms,
            slo_df["test_accuracy"],
            marker="o",
            linestyle="-",
            color=COLOR_ACC_BAR,
            label="SLO ablations",
        )
    if not default_df.empty and pd.notna(default_df.iloc[0]["slo_target"]):
        d = default_df.iloc[0]
        d_slo_ms = d["slo_target"] * 1000.0
        ax1.scatter(
            [d_slo_ms],
            [d["test_accuracy"]],
            color="#000000",
            marker="*",
            s=120,
            label="default",
            zorder=5,
        )

    ax1.set_xlabel("SLO target (ms)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Effects of different SLOs – accuracy")
    ax1.legend()

    fig.tight_layout()
    fig.savefig(ABLATION_FIG_DIR / "slo_ablation_latency_accuracy.png", dpi=300)
    fig.savefig(ABLATION_FIG_DIR / "slo_ablation_latency_accuracy.eps", format='eps')
    plt.close(fig)

    # ==========================================================
    # Figure 2: SLO vs split_block (actual split decision)
    # ==========================================================
    fig2, ax = plt.subplots(figsize=(6, 4))
    if not slo_df.empty:
        ax.plot(
            slo_ms,
            slo_df["split_block"],
            marker="o",
            linestyle="-",
            color=COLOR_LAT_AVG,
            label="SLO ablations",
        )
        ax.set_yticks(sorted(slo_df["split_block"].dropna().unique()))

    if not default_df.empty:
        d = default_df.iloc[0]
        if pd.notna(d["slo_target"]) and pd.notna(d["split_block"]):
            d_slo_ms = d["slo_target"] * 1000.0
            ax.scatter(
                [d_slo_ms],
                [d["split_block"]],
                color="#000000",
                marker="*",
                s=120,
                label="default",
                zorder=5,
            )

    ax.set_xlabel("SLO target (ms)")
    ax.set_ylabel("Chosen split block index")
    ax.set_title("SLO target vs split block (default + A_slo_*)")
    ax.legend()

    fig2.tight_layout()
    fig2.savefig(ABLATION_FIG_DIR / "slo_vs_split_point.png", dpi=300)
    fig2.savefig(ABLATION_FIG_DIR / "slo_vs_split_point.eps", format='eps')
    plt.close(fig2)


# --------------------- 2. Exit distributions (all) --------------------

def plot_exit_distributions(df: pd.DataFrame) -> None:
    """
    Stacked bar chart: for each experiment, fraction of samples exiting at 1, 2, 3, final.
    """
    df2 = df.copy()
    df2 = df2[df2["family"] != "baseline"]  

    if df2.empty:
        print("[ablations] exit distributions: nothing to plot after filtering")
        return

    desired_order = [
        "default",
        "A_slo_100",
        "A_slo_200",
        "A_slo_60",
        "B_net_fast",
        "B_net_slow",
        "B_net_impossible",
        "C_data_mnist",
        "C_data_cifar10",
        "C_data_cifar100",
        "D_edge_05",
        "D_edge_100",
        "D_edge_40",
    ]

    present_names = df2["exp_name"].tolist()
    ordered_names = [n for n in desired_order if n in present_names] + [
        n for n in present_names if n not in desired_order
    ]

    df2 = (
        df2.set_index("exp_name")
        .loc[ordered_names]
        .reset_index()
    )

    labels = df2["exp_name"].tolist()
    x = np.arange(len(labels))

    e1 = df2["exit1_frac"].values
    e2 = df2["exit2_frac"].values
    e3 = df2["exit3_frac"].values
    ef = df2["exit_final_frac"].values

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))

    bottom = np.zeros_like(e1)
    ax.bar(x, e1, bottom=bottom, label="exit 1", color=EXIT_COLOURS["1"])
    bottom += e1
    ax.bar(x, e2, bottom=bottom, label="exit 2", color=EXIT_COLOURS["2"])
    bottom += e2
    ax.bar(x, e3, bottom=bottom, label="exit 3", color=EXIT_COLOURS["3"])
    bottom += e3
    ax.bar(x, ef, bottom=bottom, label="final", color=EXIT_COLOURS["final"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Fraction of samples")
    ax.set_title("Exit distribution per experiment")
    ax.legend()

    fig.tight_layout()
    fig.savefig(ABLATION_FIG_DIR / "exit_distributions_all.png", dpi=300)
    fig.savefig(ABLATION_FIG_DIR / "exit_distributions_all.eps", format='eps')
    plt.close(fig)


# --------------------- 3. Network ablation (FIXED) ---------------------------

def plot_network_ablation(df: pd.DataFrame) -> None:
    net_df = df[df["family"] == "net"].copy()
    default_df = df[df["exp_name"] == "default"].copy()

    if net_df.empty and default_df.empty:
        print("[ablations] no default / B_net_*; skipping network ablation")
        return

    rows = []
    if not default_df.empty:
        d = default_df.iloc[0]
        rows.append({
            "label": "default",
            "test_avg_latency_ms": d["test_avg_latency_ms"],
            "test_p95_latency_ms": d["test_p95_latency_ms"],
            "test_accuracy": d["test_accuracy"],
        })

    for _, r in net_df.iterrows():
        rows.append({
            "label": r["exp_name"],
            "test_avg_latency_ms": r["test_avg_latency_ms"],
            "test_p95_latency_ms": r["test_p95_latency_ms"],
            "test_accuracy": r["test_accuracy"],
        })

    plot_df = pd.DataFrame(rows)

    desired_order = ["B_net_fast", "default", "B_net_slow", "B_net_impossible"]
    present = plot_df["label"].tolist()
    ordered_labels = [l for l in desired_order if l in present] + [l for l in present if l not in desired_order]
    plot_df = plot_df.set_index("label").loc[ordered_labels].reset_index()

    pretty = {
        "B_net_fast": "Fast",
        "default": "Moderate",
        "B_net_slow": "Slow",
        "B_net_impossible": "Impossible",
    }

    x = np.arange(len(plot_df))
    labels = plot_df["label"].tolist()
    display_labels = [pretty.get(l, l) for l in labels]
    width = 0.35

    slo_ms = None
    if not default_df.empty:
        val = default_df.iloc[0].get("slo_target")
        if pd.notna(val):
            slo_ms = val * 1000.0
    if slo_ms is None:
        slo_series = df["slo_target"].dropna()
        if not slo_series.empty:
            slo_ms = slo_series.iloc[0] * 1000.0

    # ===================== Figure 1: latency =====================
    fig_lat, ax_lat = plt.subplots(figsize=(7, 4))

    ax_lat.bar(x - width / 2, plot_df["test_avg_latency_ms"], width, label="avg latency", color=COLOR_LAT_AVG)
    ax_lat.bar(x + width / 2, plot_df["test_p95_latency_ms"], width, label="p95 latency", color=COLOR_LAT_P95)
    
    if slo_ms is not None:
        ax_lat.axhline(y=slo_ms, linestyle="--", color="#777777", label=f"SLO ({slo_ms:.0f} ms)")

    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Effects of different networks")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(display_labels, rotation=0)
    ax_lat.legend(loc="upper right", frameon=True)

    fig_lat.tight_layout()
    fig_lat.savefig(ABLATION_FIG_DIR / "network_ablation_latency.png", dpi=300)
    fig_lat.savefig(ABLATION_FIG_DIR / "network_ablation_latency.eps", format='eps')
    plt.close(fig_lat)

    # ===================== Figure 2: accuracy ====================
    fig_acc, ax_acc = plt.subplots(figsize=(7, 4))
    ax_acc.bar(x, plot_df["test_accuracy"], color=COLOR_ACC_BAR)
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Effects of different networks – accuracy")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(display_labels, rotation=0)

    fig_acc.tight_layout()
    fig_acc.savefig(ABLATION_FIG_DIR / "network_ablation_accuracy.png", dpi=300, bbox_inches="tight")
    fig_acc.savefig(ABLATION_FIG_DIR / "network_ablation_accuracy.eps", format='eps', bbox_inches="tight")
    plt.close(fig_acc)


# --------------------- 4. Edge ablation (FIXED) ---------------------------

def plot_edge_ablation(df: pd.DataFrame) -> None:
    edge_df = df[df["family"] == "edge"].copy()
    default_df = df[df["exp_name"] == "default"].copy()

    if edge_df.empty and default_df.empty:
        print("[ablations] no default / D_edge_*; skipping edge ablation")
        return

    rows = []
    if not default_df.empty:
        d = default_df.iloc[0]
        rows.append({
            "label": "default",
            "edge_slowdown": d["edge_slowdown"],
            "test_avg_latency_ms": d["test_avg_latency_ms"],
            "test_p95_latency_ms": d["test_p95_latency_ms"],
            "test_accuracy": d["test_accuracy"],
        })

    for _, r in edge_df.iterrows():
        rows.append({
            "label": r["exp_name"],
            "edge_slowdown": r["edge_slowdown"],
            "test_avg_latency_ms": r["test_avg_latency_ms"],
            "test_p95_latency_ms": r["test_p95_latency_ms"],
            "test_accuracy": r["test_accuracy"],
        })

    plot_df = pd.DataFrame(rows).sort_values("edge_slowdown")

    x = np.arange(len(plot_df))
    width = 0.35
    slowdown_vals = plot_df["edge_slowdown"].values

    slo_ms = None
    if not default_df.empty:
        val = default_df.iloc[0].get("slo_target")
        if pd.notna(val):
            slo_ms = val * 1000.0
    if slo_ms is None:
        slo_series = df["slo_target"].dropna()
        if not slo_series.empty:
            slo_ms = slo_series.iloc[0] * 1000.0

    # ===================== Figure 1: latency =====================
    fig_lat, ax_lat = plt.subplots(figsize=(7, 4))

    ax_lat.bar(x - width / 2, plot_df["test_avg_latency_ms"], width, label="avg latency", color=COLOR_LAT_AVG)
    ax_lat.bar(x + width / 2, plot_df["test_p95_latency_ms"], width, label="p95 latency", color=COLOR_LAT_P95)
    
    if slo_ms is not None:
        ax_lat.axhline(y=slo_ms, linestyle="--", color="#777777", label=f"SLO ({slo_ms:.0f} ms)")

    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Effects of edge slowdown")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels([f"{v:.0f}" if pd.notna(v) else "" for v in slowdown_vals])
    ax_lat.set_xlabel("Edge slowdown factor")
    ax_lat.legend()

    fig_lat.tight_layout()
    fig_lat.savefig(ABLATION_FIG_DIR / "edge_ablation_latency.png", dpi=300)
    fig_lat.savefig(ABLATION_FIG_DIR / "edge_ablation_latency.eps", format='eps')
    plt.close(fig_lat)

    # ===================== Figure 2: accuracy ====================
    fig_acc, ax_acc = plt.subplots(figsize=(7, 4))
    ax_acc.bar(x, plot_df["test_accuracy"], color=COLOR_ACC_BAR)
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Effects of edge slowdown – accuracy")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([f"{v:.0f}" if pd.notna(v) else "" for v in slowdown_vals])
    ax_acc.set_xlabel("Edge slowdown factor")

    fig_acc.tight_layout()
    fig_acc.savefig(ABLATION_FIG_DIR / "edge_ablation_accuracy.png", dpi=300)
    fig_acc.savefig(ABLATION_FIG_DIR / "edge_ablation_accuracy.eps", format='eps')
    plt.close(fig_acc)

# --------------------- 5. Dataset comparison ablation ----------------

def plot_dataset_comparison(df: pd.DataFrame) -> None:
    """
    Dataset ablation:
      - baselines (cifar10/fmnist/mnist) vs our method per dataset.
    """
    if "dataset" not in df.columns:
        return

    relevant = df[(df["family"].isin(["default", "dataset"]))].copy()
    if relevant.empty:
        print("[ablations] no default / C_data_*; skipping dataset comparison")
        return

    datasets = sorted(relevant["dataset"].dropna().unique())
    rows = []

    for ds in datasets:
        baseline_name = DATASET_BASELINES.get(ds)
        baseline_row = df[df["exp_name"] == baseline_name] if baseline_name else pd.DataFrame()
        baseline = baseline_row.iloc[0] if not baseline_row.empty else None

        ours_rows = relevant[relevant["dataset"] == ds]
        ours = ours_rows.iloc[0] if not ours_rows.empty else None

        if baseline is None and ours is None:
            continue

        if baseline is not None:
            rows.append({
                "dataset": ds,
                "kind": "baseline",
                "test_accuracy": baseline["test_accuracy"],
                "test_avg_latency_ms": baseline["test_avg_latency_ms"],
            })

        if ours is not None:
            rows.append({
                "dataset": ds,
                "kind": "ours",
                "test_accuracy": ours["test_accuracy"],
                "test_avg_latency_ms": ours["test_avg_latency_ms"],
            })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        print("[ablations] dataset comparison: nothing to plot")
        return

    datasets = sorted(plot_df["dataset"].unique())
    kinds = ["baseline", "ours"]

    x = np.arange(len(datasets))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # Accuracy
    for i, kind in enumerate(kinds):
        sub = plot_df[plot_df["kind"] == kind]
        vals = [
            sub[sub["dataset"] == ds]["test_accuracy"].iloc[0]
            if not sub[sub["dataset"] == ds].empty else np.nan
            for ds in datasets
        ]
        color = COLOR_ACC_BASELINE if kind == "baseline" else COLOR_ACC_OURS
        axes[0].bar(x + (i - 0.5) * width, vals, width, label=kind, color=color)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Baselines vs ours across datasets")
    axes[0].legend()

    # Latency
    for i, kind in enumerate(kinds):
        sub = plot_df[plot_df["kind"] == kind]
        vals = [
            sub[sub["dataset"] == ds]["test_avg_latency_ms"].iloc[0]
            if not sub[sub["dataset"] == ds].empty else np.nan
            for ds in datasets
        ]
        color = COLOR_ACC_BASELINE if kind == "baseline" else COLOR_ACC_OURS
        axes[1].bar(x + (i - 0.5) * width, vals, width, label=kind, color=color)

    axes[1].set_ylabel("Avg latency (ms)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_xlabel("Dataset")

    fig.tight_layout()
    fig.savefig(ABLATION_FIG_DIR / "dataset_baseline_vs_ours.png", dpi=300)
    fig.savefig(ABLATION_FIG_DIR / "dataset_baseline_vs_ours.eps", format='eps')
    plt.close(fig)


# --------------------- 6. Baseline cloud vs edge ---------------------

def plot_baseline_cloud_vs_edge(experiments: dict) -> None:
    """
    Compare cloud vs edge latency for baseline models.
    """
    rows = []

    for name, data in experiments.items():
        if classify_experiment(name) != "baseline":
            continue

        cfg = data["config"]
        tm = data["test_metrics"]

        dataset = (
            tm.get("dataset")
            or cfg.get("dataset")
            or name
        )

        cloud_avg = (
            tm.get("cloud_avg_latency_ms")
            or tm.get("cloud_avg_latency")
        )
        cloud_p95 = (
            tm.get("cloud_p95_latency_ms")
            or tm.get("cloud_p95_latency")
        )
        edge_avg = (
            tm.get("edge_avg_latency_ms")
            or tm.get("edge_avg_latency")
        )
        edge_p95 = (
            tm.get("edge_p95_latency_ms")
            or tm.get("edge_p95_latency")
        )

        if cloud_avg is None or edge_avg is None:
            continue

        rows.append({
            "dataset": dataset,
            "cloud_avg": cloud_avg,
            "cloud_p95": cloud_p95,
            "edge_avg": edge_avg,
            "edge_p95": edge_p95,
        })

    if not rows:
        print("[ablations] no usable baselines for cloud vs edge; skipping")
        return

    import pandas as _pd
    df = _pd.DataFrame(rows).sort_values("dataset")

    x = np.arange(len(df))
    labels = df["dataset"].tolist()
    width = 0.35

    # ---------- Figure: avg & p95 latency ----------
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # top: avg latency
    ax0 = axes[0]
    ax0.bar(x - width / 2, df["cloud_avg"], width,
            label="cloud", color=COLOR_ACC_BASELINE)
    ax0.bar(x + width / 2, df["edge_avg"], width,
            label="edge", color=COLOR_ACC_OURS)
    ax0.set_ylabel("Avg latency (ms)")
    ax0.set_title("Baseline models – cloud vs edge (avg latency)")
    ax0.legend()

    # bottom: p95 latency
    ax1 = axes[1]
    if df["cloud_p95"].notna().any() or df["edge_p95"].notna().any():
        ax1.bar(x - width / 2, df["cloud_p95"], width,
                label="cloud", color=COLOR_ACC_BASELINE)
        ax1.bar(x + width / 2, df["edge_p95"], width,
                label="edge", color=COLOR_ACC_OURS)
        ax1.set_ylabel("p95 latency (ms)")
        ax1.set_title("Baseline models – cloud vs edge (p95 latency)")
        ax1.legend()
    else:
        # if no p95 data, just repeat avg
        ax1.bar(x - width / 2, df["cloud_avg"], width,
                label="cloud", color=COLOR_ACC_BASELINE)
        ax1.bar(x + width / 2, df["edge_avg"], width,
                label="edge", color=COLOR_ACC_OURS)
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Baseline models – cloud vs edge (latency)")
        ax1.legend()

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Dataset")

    fig.tight_layout()
    fig.savefig(ABLATION_FIG_DIR / "baseline_cloud_vs_edge_latency.png", dpi=300)
    fig.savefig(ABLATION_FIG_DIR / "baseline_cloud_vs_edge_latency.eps", format='eps')
    plt.close(fig)

# --------------------- 7. Family-specific Pareto plots ----------------
# --------------------- 7b. Family-specific Pareto plots (P95) ----------------

def plot_family_paretos_p95(df: pd.DataFrame) -> None:
    """
    Make an accuracy–latency (P95) Pareto figure for each ablation family.
    Output files will have '_p95' suffix.
    """
    family_display_names = {
        "slo": "SLO target",
        "edge": "Edge slowdown",
        "net": "Network profile",
        "dataset": "Dataset",
        "baseline": "Baseline"
    }

    def _make_fig(sub_df, title, fname, label_map=None, legend_loc="upper left"):
        if sub_df.empty:
            print(f"[ablations] {fname}: nothing to plot, skipping")
            return

        fig, ax = plt.subplots(figsize=(7, 5))

        non_default = sub_df[sub_df["exp_name"] != "default"]
        for family, group in non_default.groupby("family"):
            
            label_text = family_display_names.get(family, family)

            ax.scatter(
                group["test_p95_latency_ms"], 
                group["test_accuracy"],
                label=label_text,
                color=_family_colour(family),
                s=40,
            )
            for _, row in group.iterrows():
                raw_name = row["exp_name"]
                txt = label_map.get(raw_name, raw_name) if label_map else raw_name
                
                ax.annotate(
                    txt,
                    (row["test_p95_latency_ms"], row["test_accuracy"]), 
                    fontsize=7,
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                )

        default_df = sub_df[sub_df["exp_name"] == "default"]
        if not default_df.empty:
            ax.scatter(
                default_df["test_p95_latency_ms"], 
                default_df["test_accuracy"],
                label="default",
                color="#000000",
                marker="*",
                s=120,
                zorder=5,
            )

        xs = sub_df["test_p95_latency_ms"].astype(float)
        ys = sub_df["test_accuracy"].astype(float)

        x_span = xs.max() - xs.min()
        y_span = ys.max() - ys.min()

        pad_x = max(0.5, 0.05 * x_span) if x_span > 0 else 1.0
        pad_y = max(0.2, 0.05 * y_span) if y_span > 0 else 0.5

        ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
        ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)

        ax.set_xlabel("P95 latency (ms)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(title)

        ax.legend(loc=legend_loc)

        fig.tight_layout()
        # Save PNG
        fig.savefig(ABLATION_FIG_DIR / fname, dpi=300)
        # Save EPS
        eps_fname = fname.replace(".png", ".eps")
        fig.savefig(ABLATION_FIG_DIR / eps_fname, format='eps')
        plt.close(fig)

    net_label_map = {
        "B_net_fast": "Fast",
        "default": "Moderate",
        "B_net_slow": "Slow",
        "B_net_impossible": "Impossible",
    }

    dataset_label_map = {
        "C_data_mnist": "mnist",
        "C_data_cifar10": "cifar10",
        "C_data_cifar100": "cifar100",
    }

    # -----(P95) -------
    slo_mask = (df["family"] == "slo") | (df["exp_name"] == "default")
    slo_df = df[slo_mask].copy()
    slo_label_map = {}
    for _, r in slo_df.iterrows():
        if r["exp_name"] == "default":
            continue
        if pd.notna(r.get("slo_target")):
            slo_ms = float(r["slo_target"]) * 1000.0
            slo_label_map[r["exp_name"]] = f"{slo_ms:.0f}"
    _make_fig(
        slo_df,
        "Accuracy–P95 Latency trade-off",
        "pareto_slo_p95.png",
        label_map=slo_label_map,
        legend_loc="upper left" 
    )

    # ------- Edge slowdown (P95) -------
    edge_mask = (df["family"] == "edge") | (df["exp_name"] == "default")
    edge_df = df[edge_mask].copy()
    edge_label_map = {}
    for _, r in edge_df.iterrows():
        if r["exp_name"] == "default":
            continue
        if pd.notna(r.get("edge_slowdown")):
            edge_label_map[r["exp_name"]] = f"{float(r['edge_slowdown']):.0f}"
    _make_fig(
        edge_df,
        "Accuracy–P95 Latency trade-off",
        "pareto_edge_p95.png",
        label_map=edge_label_map,
        legend_loc="upper left" 
    )

    # ------- Network (P95) -------
    net_mask = (df["family"] == "net") | (df["exp_name"] == "default")
    _make_fig(
        df[net_mask],
        "Accuracy–P95 Latency trade-off",
        "pareto_net_p95.png",
        label_map=net_label_map,
        legend_loc="upper left" 
    )

    # ------- Dataset (P95) -------
    data_mask = df["family"].isin(["dataset", "baseline"]) | (df["exp_name"] == "default")
    _make_fig(
        df[data_mask],
        "Accuracy–P95 Latency trade-off",
        "pareto_dataset_p95.png",
        label_map=dataset_label_map,
        legend_loc="upper right" 
    )


def plot_family_paretos(df: pd.DataFrame) -> None:
    """
    Make an accuracy–latency Pareto figure for each ablation family.
    """

    family_display_names = {
        "slo": "SLO Ablation",       
        "edge": "Edge Slowdown Ablation",  
        "net": "Network Ablation",
        "dataset": "Dataset",
        "baseline": "Baseline"
    }

    def _make_fig(sub_df, title, fname, label_map=None, legend_loc="upper left"):
        if sub_df.empty:
            print(f"[ablations] {fname}: nothing to plot, skipping")
            return

        fig, ax = plt.subplots(figsize=(7, 5))

        non_default = sub_df[sub_df["exp_name"] != "default"]
        for family, group in non_default.groupby("family"):
            
            label_text = family_display_names.get(family, family)

            ax.scatter(
                group["test_avg_latency_ms"],
                group["test_accuracy"],
                label=label_text, 
                color=_family_colour(family),
                s=40,
            )
            for _, row in group.iterrows():
                raw_name = row["exp_name"]
                txt = label_map.get(raw_name, raw_name) if label_map else raw_name
                ax.annotate(
                    txt,
                    (row["test_avg_latency_ms"], row["test_accuracy"]),
                    fontsize=7,
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                )

        default_df = sub_df[sub_df["exp_name"] == "default"]
        if not default_df.empty:
            ax.scatter(
                default_df["test_avg_latency_ms"],
                default_df["test_accuracy"],
                label="default",
                color="#000000",
                marker="*",
                s=120,
                zorder=5,
            )

        # padding
        xs = sub_df["test_avg_latency_ms"].astype(float)
        ys = sub_df["test_accuracy"].astype(float)

        x_span = xs.max() - xs.min()
        y_span = ys.max() - ys.min()

        pad_x = max(0.5, 0.05 * x_span) if x_span > 0 else 1.0
        pad_y = max(0.2, 0.05 * y_span) if y_span > 0 else 0.5

        ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
        ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)

        ax.set_xlabel("Average latency (ms)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(title)

        ax.legend(loc=legend_loc)

        fig.tight_layout()
        # Save PNG
        fig.savefig(ABLATION_FIG_DIR / fname, dpi=300)
        # Save EPS
        eps_fname = fname.replace(".png", ".eps")
        fig.savefig(ABLATION_FIG_DIR / eps_fname, format='eps')
        plt.close(fig)

    net_label_map = {
        "B_net_fast": "Fast",
        "default": "Moderate",
        "B_net_slow": "Slow",
        "B_net_impossible": "Impossible",
    }

    dataset_label_map = {
        "C_data_mnist": "mnist",
        "C_data_cifar10": "cifar10",
        "C_data_cifar100": "cifar100",
    }

    # ------- SLO -------
    slo_mask = (df["family"] == "slo") | (df["exp_name"] == "default")
    slo_df = df[slo_mask].copy()
    slo_label_map = {}
    for _, r in slo_df.iterrows():
        if r["exp_name"] == "default":
            continue
        if pd.notna(r.get("slo_target")):
            slo_ms = float(r["slo_target"]) * 1000.0
            slo_label_map[r["exp_name"]] = f"{slo_ms:.0f}"
    _make_fig(
        slo_df,
        "Accuracy–latency trade-off",
        "pareto_slo.png",
        label_map=slo_label_map,
        legend_loc="upper left" 
    )

    # ------- Edge slowdown -------
    edge_mask = (df["family"] == "edge") | (df["exp_name"] == "default")
    edge_df = df[edge_mask].copy()
    edge_label_map = {}
    for _, r in edge_df.iterrows():
        if r["exp_name"] == "default":
            continue
        if pd.notna(r.get("edge_slowdown")):
            edge_label_map[r["exp_name"]] = f"{float(r['edge_slowdown']):.0f}"
    _make_fig(
        edge_df,
        "Accuracy–latency trade-off",
        "pareto_edge.png",
        label_map=edge_label_map,
        legend_loc="upper left" 
    )

    # ------- Network -------
    net_mask = (df["family"] == "net") | (df["exp_name"] == "default")
    _make_fig(
        df[net_mask],
        "Accuracy–latency trade-off",
        "pareto_net.png",
        label_map=net_label_map,
        legend_loc="upper left" 
    )

    # ------- Dataset -------
    data_mask = df["family"].isin(["dataset", "baseline"]) | (df["exp_name"] == "default")
    _make_fig(
        df[data_mask],
        "Accuracy–latency trade-off",
        "pareto_dataset.png",
        label_map=dataset_label_map,
        legend_loc="upper right" 
    )

# --------------------- 8. Full cloud / edge / ours per dataset ----------------

def plot_full_env_latency_per_dataset(experiments: dict, df: pd.DataFrame) -> None:
    """
    Compare end-to-end avg latency of:
        - full cloud baseline
        - full edge baseline
        - our default method
    """
    data = {}

    # 1) baselines
    for exp_name, exp_data in experiments.items():
        if classify_experiment(exp_name) != "baseline":
            continue

        cfg = exp_data["config"]
        tm = exp_data["test_metrics"]

        dataset = tm.get("dataset") or cfg.get("dataset") or exp_name

        cloud_avg = (
            tm.get("cloud_avg_latency_ms")
            or tm.get("cloud_avg_latency")
        )
        edge_avg = (
            tm.get("edge_avg_latency_ms")
            or tm.get("edge_avg_latency")
        )

        if cloud_avg is None and edge_avg is None:
            continue

        entry = data.setdefault(dataset, {})
        if cloud_avg is not None:
            entry["baseline_cloud"] = cloud_avg
        if edge_avg is not None:
            entry["baseline_edge"] = edge_avg

    # 2) our method
    for dataset in df["dataset"].dropna().unique():
        if dataset not in data:
            continue

        ours_rows = df[
            (df["dataset"] == dataset)
            & (df["family"].isin(["default", "dataset"]))
        ]
        if ours_rows.empty:
            continue

        entry = data[dataset]
        entry["ours"] = float(ours_rows.iloc[0]["test_avg_latency_ms"])

    if not data:
        print("[ablations] full-env latency: no datasets found, skipping")
        return

    # Colours
    baseline_cloud_color = "#C0DAFC"  # light grey
    baseline_edge_color = "#C3CEDA"   # darker grey
    ours_color = "#C0C0F5"
    desired_order = ["mnist", "fmnist", "cifar10", "cifar100"]
    datasets_ordered = [ds for ds in desired_order if ds in data] + [
        ds for ds in data.keys() if ds not in desired_order
    ]

    # ----------------- A) One figure per dataset -----------------
    for dataset in datasets_ordered:
        vals = data[dataset]
        labels = []
        heights = []
        colors = []

        if "baseline_cloud" in vals:
            labels.append("baseline cloud")
            heights.append(vals["baseline_cloud"])
            colors.append(baseline_cloud_color)

        if "baseline_edge" in vals:
            labels.append("baseline edge")
            heights.append(vals["baseline_edge"])
            colors.append(baseline_edge_color)

        if "ours" in vals:
            labels.append("ours")
            heights.append(vals["ours"])
            colors.append(ours_color)

        if not labels:
            continue

        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(x, heights, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Avg latency (ms)")
        ax.set_title(f"End-to-end latency – {dataset}")

        fig.tight_layout()
        fig.savefig(ABLATION_FIG_DIR / f"full_env_latency_{dataset}.png", dpi=300)
        fig.savefig(ABLATION_FIG_DIR / f"full_env_latency_{dataset}.eps", format='eps')
        plt.close(fig)

    # ----------------- B) Combined figure (all datasets) -----------------
    datasets = datasets_ordered
    n = len(datasets)

    x = np.arange(n) * 1.6  
    width = 0.3

    cloud_vals = []
    edge_vals = []
    ours_vals = []

    for ds in datasets:
        vals = data[ds]
        cloud_vals.append(vals.get("baseline_cloud", np.nan))
        edge_vals.append(vals.get("baseline_edge", np.nan))
        ours_vals.append(vals.get("ours", np.nan))

    fig, ax = plt.subplots(figsize=(max(7, n * 1.8), 6))

    ax.bar(
        x - width,
        cloud_vals,
        width,
        label="baseline cloud",
        color=baseline_cloud_color,
    )
    ax.bar(
        x,
        edge_vals,
        width,
        label="baseline edge",
        color=baseline_edge_color,
    )
    ax.bar(
        x + width,
        ours_vals,
        width,
        label="ours",
        color=ours_color,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Avg latency (ms)")
    ax.set_title("End-to-end latency: cloud vs edge baselines vs ours")
    ax.legend()

    # fig.tight_layout()
    fig.savefig(ABLATION_FIG_DIR / "full_env_latency_all_datasets.png", dpi=300)
    fig.savefig(ABLATION_FIG_DIR / "full_env_latency_all_datasets.eps", format='eps')
    plt.close(fig)