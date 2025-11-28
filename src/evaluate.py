import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from .data_utils import get_dataloaders

def evaluate_model(model, net_sim, profiles, device, dataset_name="fmnist", batch_size=64):
    """
    Simulates inference on the test set.
    
    Args:
        model: The deployment-aware model with early exits
        net_sim: Network simulator for bandwidth and latency
        profiles: Hardware profiles with timing information
        device: torch device to run on
        dataset_name: Dataset to test on ('cifar10', 'mnist', 'fmnist')
        batch_size: Batch size for testing
    """
    model.eval()
    
    # Load Test Data using the appropriate dataset
    dataset_name_lower = dataset_name.lower() if isinstance(dataset_name, str) else "fmnist"
    
    # Validate dataset choice
    valid_datasets = {"cifar10", "mnist", "fmnist"}
    if dataset_name_lower not in valid_datasets:
        print(f"Warning: Dataset '{dataset_name}' not recognized. Using 'fmnist' instead.")
        dataset_name_lower = "fmnist"
    
    print(f"Loading {dataset_name_lower.upper()} test dataset for evaluation...")
    
    # Get the correct test loader
    _, test_loader, _, _ = get_dataloaders(
        dataset_name=dataset_name_lower,
        batch_size=batch_size,
    )
    
    # Metrics
    total_samples = 0
    correct = 0
    latencies = []
    exit_counts = {k: 0 for k in model.exit_points}  # How many times did we exit at each point?
    exit_counts['final'] = 0

    # Track confidence values for samples that actually exit at each head
    exit_conf_values = {k: [] for k in model.exit_points}

    
    # 1. Determine the Fixed Split Point (The one with highest logit)
    split_probs = F.softmax(model.split_logits, dim=0)
    split_point = torch.argmax(split_probs).item()
    
    print(f"Running Evaluation on {dataset_name_lower.upper()}. Fixed Split Point: Block {split_point}")


    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_curr_size = data.size(0)
            
            # Sample Network (Stochastic per batch, or fix it for deterministic testing)
            bw, rtt = net_sim.sample_network_state()
            
            # We must manually run the blocks to simulate the decision flow
            x = data
            exited_mask = torch.zeros(batch_curr_size, dtype=torch.bool, device=device)
            final_preds = torch.zeros(batch_curr_size, 10, device=device)
            
            # --- EDGE EXECUTION ---
            current_latency = torch.zeros(batch_curr_size, device=device)
            
            for i, block in enumerate(model.backbone.blocks):
                
                # Check if we crossed the split point -> Add Communication Latency
                if i == split_point:
                    # Calc Comm Time
                    data_size = profiles[f"block_{i-1}"]["output_bytes"] if i > 0 else profiles["block_0"]["output_bytes"] # Approx
                    t_comm = net_sim.estimate_transmission_time(data_size, bw, rtt)
                    
                    # Add comm time only to those who haven't exited yet
                    current_latency[~exited_mask] += t_comm

                # Execute Block
                # If on edge (i <= split), add edge time. If cloud (i > split), add cloud time.
                hw_key = f"block_{i}"
                if i <= split_point:
                    exec_time = profiles[hw_key]["edge_time_sec"]
                else:
                    exec_time = profiles[hw_key]["cloud_time_sec"]
                
                # Only run computation for samples that haven't exited
                # (In PyTorch we usually run the whole batch, but we logically account for latency)
                current_latency[~exited_mask] += exec_time
                x = block(x)

                # Check Early Exit
                if i in model.exit_points:
                    # Run Exit Head
                    exit_out = model.exit_heads[str(i)](x)
                    
                    # Calculate Probability
                    confidence, _ = torch.max(F.softmax(exit_out, dim=1), dim=1)
                    
                    # Get Thresholds
                    ex_idx = sorted(list(model.exit_points)).index(i)
                    gamma = model.exit_scale[ex_idx]
                    tau = model.exit_threshold[ex_idx]
                    
                    p_exit = torch.sigmoid((confidence - tau) * gamma)
                    
                    # HARD DECISION: Exit if p > 0.5
                    should_exit = (p_exit > 0.5) & (~exited_mask)
                    
                    if should_exit.any():
                        # Store predictions for those exiting now
                        final_preds[should_exit] = exit_out[should_exit]
                        
                        # Update counts
                        count = should_exit.sum().item()
                        exit_counts[i] += count

                        # Log confidences for those who actually exited here
                        exit_conf_values[i].extend(
                            confidence[should_exit].detach().cpu().tolist()
                        )

                        # Mark as exited
                        exited_mask = exited_mask | should_exit

            # --- FINAL CLASSIFIER (Cloud) ---
            # If anyone is left
            if (~exited_mask).any():
                x_final = model.backbone.avgpool(x)
                x_final = torch.flatten(x_final, 1)
                out_final = model.backbone.fc(x_final)

                # Only assign predictions for the samples that haven't exited
                remaining = ~exited_mask
                final_preds[remaining] = out_final[remaining]
                exit_counts['final'] += remaining.sum().item()
            
            # Compute Accuracy
            preds = final_preds.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total_samples += batch_curr_size
            
            # Store latencies
            latencies.extend(current_latency.cpu().tolist())

    # Final Stats
    accuracy = 100. * correct / total_samples
    avg_latency = np.mean(latencies) * 1000 # to ms
    p95_latency = np.percentile(latencies, 95) * 1000
    
    # Exit rates (fraction of samples)
    exit_rates = {}
    for k, v in exit_counts.items():
        exit_rates[str(k)] = v / total_samples

    # Exit thresholds and scales per head
    sorted_exits = sorted(model.exit_points)
    exit_thresholds = {}
    exit_scales = {}
    for idx, block_idx in enumerate(sorted_exits):
        exit_thresholds[int(block_idx)] = float(model.exit_threshold[idx].item())
        exit_scales[int(block_idx)] = float(model.exit_scale[idx].item())

    # Confidence stats for samples that actually exited at each head
    exit_conf_stats = {}
    for k in model.exit_points:
        vals = exit_conf_values[k]
        if len(vals) > 0:
            exit_conf_stats[int(k)] = {
                "mean_conf_exit": float(np.mean(vals)),
                "min_conf_exit": float(np.min(vals)),
                "max_conf_exit": float(np.max(vals)),
                "num_exited": int(len(vals))
            }
        else:
            exit_conf_stats[int(k)] = {
                "mean_conf_exit": None,
                "min_conf_exit": None,
                "max_conf_exit": None,
                "num_exited": 0
            }

    # Split probs snapshot used during evaluation
    split_probs_list = split_probs.detach().cpu().tolist()

    # Edge slowdown (approx from profiles)
    edge_slowdown = profiles["block_0"]["edge_time_sec"] / profiles["block_0"]["cloud_time_sec"]

    # Network config from simulator
    net_avg_bw_mbps = getattr(net_sim, "avg_bw", None)
    net_avg_rtt_ms = getattr(net_sim, "avg_rtt", None)

    results = {
        "test_accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "exit_distribution": exit_counts,
        "exit_rates": exit_rates,
        "split_point": split_point,
        "split_probs": split_probs_list,
        "exit_thresholds": exit_thresholds,
        "exit_scales": exit_scales,
        "exit_confidence_stats": exit_conf_stats,
        "num_samples": total_samples,
        "dataset": dataset_name_lower,
        "edge_slowdown": edge_slowdown,
        "net_avg_bw_mbps": net_avg_bw_mbps,
        "net_avg_rtt_ms": net_avg_rtt_ms,
    }
    
    return results