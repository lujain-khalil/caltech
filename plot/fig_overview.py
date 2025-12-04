import matplotlib.pyplot as plt
import pandas as pd

from fig_settings import (
    OVERVIEW_FIG_DIR,
    FAMILY_COLOURS,
    COLOR_DATASET_LAT,
    COLOR_DATASET_ACC,
)


def _family_colour(family: str) -> str:
    return FAMILY_COLOURS.get(family, "#888888")


def plot_accuracy_latency_pareto(df: pd.DataFrame) -> None:
    """
    Scatter of all experiments:
      x = avg latency (ms)
      y = test accuracy (%)
      colour = experiment family,
      default highlighted.
    """
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    non_default = df[df["exp_name"] != "default"]
    for family, group in non_default.groupby("family"):
        ax.scatter(
            group["test_avg_latency_ms"],
            group["test_accuracy"],
            label=family,
            color=_family_colour(family),
            s=40,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["exp_name"],
                (row["test_avg_latency_ms"], row["test_accuracy"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    # default highlighted
    default_df = df[df["exp_name"] == "default"]
    if not default_df.empty:
        ax.scatter(
            default_df["test_avg_latency_ms"],
            default_df["test_accuracy"],
            label="default (ours)",
            color="#000000",
            marker="*",
            s=120,
            zorder=5,
        )
        row = default_df.iloc[0]
        ax.annotate(
            "default",
            (row["test_avg_latency_ms"], row["test_accuracy"]),
            fontsize=9,
            fontweight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Average latency (ms)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy–latency trade-off (all experiments)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OVERVIEW_FIG_DIR / "overview_accuracy_latency_pareto.png", dpi=300)
    plt.close(fig)


def plot_accuracy_latency_p95_pareto(df: pd.DataFrame) -> None:
    """
    Same as plot_accuracy_latency_pareto, but x = p95 latency (ms).
    """
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    non_default = df[df["exp_name"] != "default"]
    for family, group in non_default.groupby("family"):
        ax.scatter(
            group["test_p95_latency_ms"],
            group["test_accuracy"],
            label=family,
            color=_family_colour(family),
            s=40,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["exp_name"],
                (row["test_p95_latency_ms"], row["test_accuracy"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    default_df = df[df["exp_name"] == "default"]
    if not default_df.empty:
        ax.scatter(
            default_df["test_p95_latency_ms"],
            default_df["test_accuracy"],
            label="default (ours)",
            color="#000000",
            marker="*",
            s=120,
            zorder=5,
        )
        row = default_df.iloc[0]
        ax.annotate(
            "default",
            (row["test_p95_latency_ms"], row["test_accuracy"]),
            fontsize=9,
            fontweight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("p95 latency (ms)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy–p95-latency trade-off (all experiments)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OVERVIEW_FIG_DIR / "overview_accuracy_latency_p95_pareto.png",
        dpi=300,
    )
    plt.close(fig)


def plot_dataset_overview(df: pd.DataFrame) -> None:
    """
    Dataset overview, split into two separate figures:

      - overview_dataset_latency.png
      - overview_dataset_accuracy.png
    """
    if "dataset" not in df.columns:
        return

    grouped = df.groupby("dataset", as_index=False).agg({
        "test_accuracy": "mean",
        "test_avg_latency_ms": "mean",
    })
    if grouped.empty:
        return

    x = range(len(grouped))
    labels = grouped["dataset"].tolist()

    # -------- Figure 1: latency --------
    fig_lat, ax_lat = plt.subplots(figsize=(6, 4))

    ax_lat.bar(x, grouped["test_avg_latency_ms"], color=COLOR_DATASET_LAT)
    ax_lat.set_ylabel("Avg latency (ms)")
    ax_lat.set_title("Dataset impact – latency (mean over runs)")
    ax_lat.set_xticks(list(x))
    ax_lat.set_xticklabels(labels)
    ax_lat.set_xlabel("Dataset")

    fig_lat.tight_layout()
    fig_lat.savefig(OVERVIEW_FIG_DIR / "overview_dataset_latency.png", dpi=300)
    plt.close(fig_lat)

    # -------- Figure 2: accuracy --------
    fig_acc, ax_acc = plt.subplots(figsize=(6, 4))

    ax_acc.bar(x, grouped["test_accuracy"], color=COLOR_DATASET_ACC)
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Dataset impact – accuracy (mean over runs)")
    ax_acc.set_xticks(list(x))
    ax_acc.set_xticklabels(labels)
    ax_acc.set_xlabel("Dataset")

    fig_acc.tight_layout()
    fig_acc.savefig(OVERVIEW_FIG_DIR / "overview_dataset_accuracy.png", dpi=300)
    plt.close(fig_acc)
