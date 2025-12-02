import os
import json
import torch

# Local imports
import src.config as Config
from src.resnet_split import SplittableResNet18
from src.data_utils import get_dataloaders
from src.network_sim import NetworkSimulator

def evaluate_baseline(model, dataset_name, profiles, device, net_sim, batch_size):
    """
    Evaluate accuracy and simulate latency for:
      - edge-only execution (all blocks on edge, no comm)
      - cloud-only execution (input -> cloud, all blocks on cloud)

    This mirrors the style of src/evaluate.py but without exits/splits.
    """
    _, test_loader, _, _ = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
    )

    model.eval()
    total = 0
    correct = 0

    # Determine number of blocks and their times from profile
    block_indices = sorted(
        int(k.split("_")[1]) for k in profiles.keys() if k.startswith("block_")
    )
    edge_block_times = [profiles[f"block_{i}"]["edge_time_sec"] for i in block_indices]
    cloud_block_times = [profiles[f"block_{i}"]["cloud_time_sec"] for i in block_indices]

    edge_compute_total = float(sum(edge_block_times))
    cloud_compute_total = float(sum(cloud_block_times))

    latencies_edge = []
    latencies_cloud = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass (same for both paths; we only use profiles for latency)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            batch_size_actual = target.size(0)
            total += batch_size_actual

            # ---- Latency simulation ----
            # Per-batch network sample, like evaluate.py
            bw_bps, rtt_sec = net_sim.sample_network_state()

            # Size of input tensor per sample (float32)
            element_size = data.element_size()         # 4 bytes typically
            numel_per_sample = data[0].numel()
            input_size_bytes = element_size * numel_per_sample

            # EDGE-ONLY:
            # all blocks on edge; no network cost
            per_sample_edge_latency = edge_compute_total / batch_size_actual
            latencies_edge.extend([per_sample_edge_latency] * batch_size_actual)

            # CLOUD-ONLY:
            # send input once, then all blocks on cloud
            t_comm = net_sim.estimate_transmission_time(
                data_size_bytes=input_size_bytes,
                bw_bps=bw_bps,
                rtt_sec=rtt_sec,
            )
            per_sample_cloud_latency = (cloud_compute_total + t_comm) / batch_size_actual
            latencies_cloud.extend([per_sample_cloud_latency] * batch_size_actual)

    import numpy as np
    latencies_edge = np.array(latencies_edge)
    latencies_cloud = np.array(latencies_cloud)

    acc = 100.0 * correct / total

    results = {
        "dataset": dataset_name,
        "test_accuracy": acc,

        "edge_avg_latency_ms": float(latencies_edge.mean() * 1000.0),
        "edge_p95_latency_ms": float(np.percentile(latencies_edge, 95) * 1000.0),

        "cloud_avg_latency_ms": float(latencies_cloud.mean() * 1000.0),
        "cloud_p95_latency_ms": float(np.percentile(latencies_cloud, 95) * 1000.0),

        "num_samples": int(total),
        "edge_compute_total_sec": edge_compute_total,
        "cloud_compute_total_sec": cloud_compute_total,

        "edge_slowdown": float(edge_block_times[0] / (cloud_block_times[0] + 1e-8)),
        "net_avg_bw_mbps": getattr(net_sim, "avg_bw", None),
        "net_avg_rtt_ms": getattr(net_sim, "avg_rtt", None),
    }

    return results


def main():
    # Load profiles
    profile_file = "latency_profile.json"
    with open(profile_file, "r") as f:
        profiles = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_sim = NetworkSimulator(avg_bw_mbps=Config.DEFAULT_BW, avg_rtt_ms=Config.DEFAULT_RTT)

    model_dir = "experiments/baseline_models"
    batch_size = Config.DEFAULT_BATCH_SIZE
    datasets = ["mnist", "fmnist", "cifar10"]

    print(f"Evaluating baseline models from: {model_dir}\n")

    for ds in datasets:
        run_dir = os.path.join(model_dir, ds)
        model_path = os.path.join(run_dir, "model.pth")

        if not os.path.exists(model_path):
            print(f"[WARNING] Model for {ds.upper()} not found at {model_path}. Skipping...")
            continue

        print(f"Loading model for {ds.upper()}...")
        # Load the model
        _, _, num_classes, input_channels = get_dataloaders(
            dataset_name=ds,
            batch_size=batch_size,
        )

        model = SplittableResNet18(
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=False
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))

        # Evaluate
        print(f"Evaluating {ds.upper()}...")
        results = evaluate_baseline(
            model=model,
            dataset_name=ds,
            profiles=profiles,
            device=device,
            net_sim=net_sim,
            batch_size=batch_size,
        )

        # Update test_metrics.json
        with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
            json.dump(results, f, indent=4)

        print(f"[{ds.upper()}] Baseline evaluation results:")
        print(f"  Accuracy: {results['test_accuracy']:.2f}%")
        print(f"  Edge:  avg {results['edge_avg_latency_ms']:.2f} ms | p95 {results['edge_p95_latency_ms']:.2f} ms")
        print(f"  Cloud: avg {results['cloud_avg_latency_ms']:.2f} ms | p95 {results['cloud_p95_latency_ms']:.2f} ms")
        print(f"  Saved to: {os.path.join(run_dir, 'test_metrics.json')}")
        print("-" * 60)


if __name__ == "__main__":
    main()
