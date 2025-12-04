from fig_loader import load_experiments, experiments_to_dataframe
from fig_overview import (
    plot_accuracy_latency_pareto,
    plot_accuracy_latency_p95_pareto,   
    plot_dataset_overview,
)
from fig_ablations import (
    plot_slo_ablation,
    plot_exit_distributions,
    plot_network_ablation,
    plot_edge_ablation,
    plot_dataset_comparison,
    plot_baseline_cloud_vs_edge,
    plot_family_paretos,        
    plot_full_env_latency_per_dataset,  
    plot_family_paretos_p95,


)
from fig_settings import EXPERIMENT_ROOTS, FIG_ROOT


def main():
    print("Loading experiments from roots:")
    for root in EXPERIMENT_ROOTS:
        print(f"  - {root}")

    experiments = load_experiments(EXPERIMENT_ROOTS)
    print(f"\nFound {len(experiments)} experiments (including baselines):")
    for name in sorted(experiments):
        print(f"  - {name}")

    df = experiments_to_dataframe(experiments)
    print("\nSummary (short):")
    cols = [
        "exp_name", "family", "dataset",
        "slo_target", "edge_slowdown",
        "test_accuracy", "test_avg_latency_ms",
    ]
    print(df[cols])

    print(f"\nSaving figures under: {FIG_ROOT}")

    # Overview
    plot_accuracy_latency_pareto(df)
    plot_accuracy_latency_p95_pareto(df)
    plot_dataset_overview(df)

    # Ablations
    plot_slo_ablation(df)
    plot_exit_distributions(df)
    plot_network_ablation(df)
    plot_edge_ablation(df)
    plot_dataset_comparison(df)
    plot_baseline_cloud_vs_edge(experiments)   
    plot_family_paretos(df)
    plot_full_env_latency_per_dataset(experiments, df)   
    plot_family_paretos_p95(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
