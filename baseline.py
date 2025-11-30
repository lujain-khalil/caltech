import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import sys
import time

# Local imports
import src.config as Config
from src.resnet_split import SplittableResNet18
from src.data_utils import get_dataloaders
from src.network_sim import NetworkSimulator

def run_command(command, description):
    """
    Helper to run a shell command and print status.
    """
    print(f"\n{'='*60}")
    print(f"STARTING: {description}")
    print(f"CMD: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        # Check if python executable is correct
        full_command = [sys.executable] + command
        subprocess.run(full_command, check=True)
        duration = time.time() - start_time
        print(f"\n>>> SUCCESS: {description} completed in {duration:.2f}s")
    except subprocess.CalledProcessError as e:
        print(f"\n>>> ERROR: {description} failed with return code {e.returncode}")
        # Optional: Stop execution on error
        # sys.exit(1)

# ---------------------------
# 1. TRAINING (CE ONLY)
# ---------------------------

def train_ce_model(dataset_name, device, epochs, batch_size, lr):
    """
    Train a vanilla SplittableResNet18 on the given dataset using CE loss only.
    Returns: trained model, training_history (list of dicts)
    """
    train_loader, _, num_classes, input_channels = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
    )

    # Always 3-channel input due to your transforms
    model = SplittableResNet18(
        num_classes=num_classes,
        input_channels=input_channels,  # should be 3
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []

    print(f"\n=== Training baseline model on {dataset_name.upper()} ===")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch:02d}/{epochs}...")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

            preds = outputs.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)

        avg_loss = running_loss / total
        acc = 100.0 * correct / total
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_acc": acc,
        })

    return model, history


# ---------------------------
# 2. EVALUATION + LATENCY
# ---------------------------

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
            batch_edge_latency = edge_compute_total
            latencies_edge.extend([batch_edge_latency] * batch_size_actual)

            # CLOUD-ONLY:
            # send input once, then all blocks on cloud
            t_comm = net_sim.estimate_transmission_time(
                data_size_bytes=input_size_bytes,
                bw_bps=bw_bps,
                rtt_sec=rtt_sec,
            )
            batch_cloud_latency = cloud_compute_total + t_comm
            latencies_cloud.extend([batch_cloud_latency] * batch_size_actual)

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


# ---------------------------
# 3. MAIN SCRIPT: LOOP 3 DATASETS
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline CE-only ResNet18 on MNIST/FMNIST/CIFAR10 with edge vs cloud latency."
    )
    parser.add_argument("--epochs", type=int, default=Config.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.DEFAULT_LR)

    parser.add_argument("--profile_file", type=str, default="latency_profile.json",
                        help="Latency profile JSON generated by profile_env.py")

    parser.add_argument("--bw_mbps", type=float, default=Config.DEFAULT_BW,
                        help="Average uplink bandwidth (for cloud-only comm).")
    parser.add_argument("--rtt_ms", type=float, default=Config.DEFAULT_RTT,
                        help="Average RTT (for cloud-only comm).")

    args = parser.parse_args()

    run_command(["profile_env.py"], f"Hardware Profiling for default EDGE_SLOWDOWN={Config.DEFAULT_SLOWDOWN}")
    if not os.path.exists(args.profile_file):
        raise FileNotFoundError(
            f"{args.profile_file} not found. Run profile_env.py first to generate it."
        )

    with open(args.profile_file, "r") as f:
        profiles = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_sim = NetworkSimulator(avg_bw_mbps=args.bw_mbps, avg_rtt_ms=args.rtt_ms)

    datasets = ["mnist", "fmnist", "cifar10"]

    base_dir = os.path.join("experiments", f"baseline_models")
    os.makedirs(base_dir, exist_ok=True)

    print(f"Saving baseline results under: {base_dir}")

    for ds in datasets:
        run_dir = os.path.join(base_dir, ds)
        os.makedirs(run_dir, exist_ok=True)

        # --- Train ---
        model, train_history = train_ce_model(
            dataset_name=ds,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        # Save model
        model_path = os.path.join(run_dir, "model.pth")
        torch.save(model.state_dict(), model_path)

        # Save train log
        with open(os.path.join(run_dir, "train_log.json"), "w") as f:
            json.dump(train_history, f, indent=4)

        # --- Evaluate (accuracy + latency edge vs cloud) ---
        results = evaluate_baseline(
            model=model,
            dataset_name=ds,
            profiles=profiles,
            device=device,
            net_sim=net_sim,
            batch_size=args.batch_size,
        )

        with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
            json.dump(results, f, indent=4)

        print(f"\n[{ds.upper()}] Baseline results:")
        print(f"  Accuracy: {results['test_accuracy']:.2f}%")
        print(f"  Edge:  avg {results['edge_avg_latency_ms']:.2f} ms | p95 {results['edge_p95_latency_ms']:.2f} ms")
        print(f"  Cloud: avg {results['cloud_avg_latency_ms']:.2f} ms | p95 {results['cloud_p95_latency_ms']:.2f} ms")
        print("-" * 60)


if __name__ == "__main__":
    main()