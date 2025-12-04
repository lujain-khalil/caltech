# from typing import Dict, Any

# import matplotlib.pyplot as plt

# from fig_settings import TRAINING_FIG_DIR


# def plot_training_curves_for_experiment(exp_name: str,
#                                         exp: Dict[str, Any]) -> None:
#     train_log = exp["train_log"]
#     cfg = exp["config"]

#     epochs = [e.get("epoch", i + 1) for i, e in enumerate(train_log)]
#     acc = [e.get("accuracy") for e in train_log]
#     lat_ms = [e.get("latency_ms") for e in train_log]
#     norm_lat = [e.get("normalized_latency") for e in train_log]
#     loss = [e.get("loss") for e in train_log]

#     split_decision = [e.get("split_decision") for e in train_log]
#     split_conf = [e.get("split_confidence") for e in train_log]

#     slo_target = cfg.get("slo_target")  # seconds

#     # ---------------------------------------------------------------
#     # Core metrics
#     # ---------------------------------------------------------------
#     fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 9))

#     axes[0].plot(epochs, acc, marker="o")
#     axes[0].set_ylabel("Acc (%)")
#     axes[0].set_title(f"Training – {exp_name}")

#     axes[1].plot(epochs, lat_ms, marker="o")
#     axes[1].set_ylabel("Lat (ms)")
#     if slo_target is not None:
#         slo_ms = slo_target * 1000.0
#         axes[1].axhline(
#             y=slo_ms,
#             linestyle="--",
#             color="grey",
#             label=f"SLO ({slo_ms:.0f} ms)",
#         )
#         axes[1].legend()

#     axes[2].plot(epochs, norm_lat, marker="o")
#     axes[2].set_ylabel("Norm. lat")

#     axes[3].plot(epochs, loss, marker="o")
#     axes[3].set_ylabel("Loss")
#     axes[3].set_xlabel("Epoch")

#     fig.tight_layout()
#     fig.savefig(TRAINING_FIG_DIR / f"training_core_{exp_name}.png", dpi=300)
#     plt.close(fig)

#     # ---------------------------------------------------------------
#     # Split behaviour
#     # ---------------------------------------------------------------
#     fig, ax1 = plt.subplots(figsize=(6, 4))
#     ax1.plot(epochs, split_decision, marker="o", label="split index")
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Split index")
#     ax1.set_title(f"Split behaviour – {exp_name}")
#     if any(d is not None for d in split_decision):
#         ax1.set_yticks(sorted({d for d in split_decision if d is not None}))

#     ax2 = ax1.twinx()
#     ax2.plot(epochs, split_conf, marker="x", linestyle="--",
#              label="split confidence")
#     ax2.set_ylabel("Split confidence")

#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

#     fig.tight_layout()
#     fig.savefig(TRAINING_FIG_DIR / f"training_split_{exp_name}.png", dpi=300)
#     plt.close(fig)

#     # ---------------------------------------------------------------
#     # Exit thresholds / scales / probabilities
#     # ---------------------------------------------------------------
#     exit_keys = sorted(train_log[0]["exit_thresholds"].keys(),
#                        key=lambda k: int(k) if k.isdigit() else k)

#     # thresholds + scales
#     fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

#     for k in exit_keys:
#         thr_series = [e["exit_thresholds"][k] for e in train_log]
#         axes[0].plot(epochs, thr_series, label=f"thr {k}")
#     axes[0].set_ylabel("Threshold")
#     axes[0].set_title(f"Exit thresholds / scales – {exp_name}")
#     axes[0].legend()

#     for k in exit_keys:
#         scale_series = [e["exit_scales"][k] for e in train_log]
#         axes[1].plot(epochs, scale_series, label=f"scale {k}")
#     axes[1].set_ylabel("Scale")
#     axes[1].set_xlabel("Epoch")
#     axes[1].legend()

#     fig.tight_layout()
#     fig.savefig(
#         TRAINING_FIG_DIR / f"training_exits_params_{exp_name}.png", dpi=300
#     )
#     plt.close(fig)

#     # exit probabilities
#     fig, ax = plt.subplots(figsize=(6, 4))
#     for k in exit_keys:
#         prob_series = [e["exit_probs_avg"][k] for e in train_log]
#         ax.plot(epochs, prob_series, label=f"exit {k}")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Avg exit prob")
#     ax.set_title(f"Avg exit probabilities – {exp_name}")
#     ax.legend()

#     fig.tight_layout()
#     fig.savefig(
#         TRAINING_FIG_DIR / f"training_exits_probs_{exp_name}.png", dpi=300
#     )
#     plt.close(fig)


# def plot_all_training_curves(experiments: Dict[str, Dict[str, Any]]) -> None:
#     for name, exp in experiments.items():
#         print(f"[training] {name}")
#         plot_training_curves_for_experiment(name, exp)

