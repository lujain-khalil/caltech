import torch
import torch.optim as optim
import os
import numpy as np

# Relative imports assuming this is run as a module or from root with path set
from .loss_function import SLOAwareLoss
from .data_utils import get_dataloaders

def train_model(config, profiles, net_sim, device, save_dir):
    # Unpack config
    batch_size = config['batch_size']
    epochs = config['epochs']
    slo_target = config['slo_target']
    
    # Data Setup
    train_loader, _, _, _ = get_dataloaders(
        dataset_name=config['dataset'],
        batch_size=batch_size,
    )

    # Model Setup
    model = config['model_instance']
    model.to(device)

    # Loss Setup
    criterion = SLOAwareLoss(
        profiles=profiles, 
        network_sim=net_sim, 
        slo_target_sec=slo_target,
        lambda_lat=config['lambda_lat'],
        mu_slo=config['mu_slo']
    )

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Logging storage
    training_history = []

    print(f"\nStarting Training (SLO: {slo_target*1000:.1f}ms)...")
    print(f"{'Epoch':<6} | {'Acc':<8} | {'Avg Latency':<12} | {'Avg p95 Latency':<12} | {'Split Layer':<15} | {'Exit Probs (Avg)'}")
    print("-" * 85)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_latency = 0
        correct = 0
        total_samples = 0

        sum_norm_lat = 0.0          # sum of normalized avg lat, weighted by samples
        sum_raw_p95 = 0.0           # sum of batch p95 lat (seconds), unweighted
        sum_norm_p95 = 0.0          # sum of batch normalized p95, unweighted
        num_batches = 0

        # Track average exit probabilities per head for this epoch
        exit_prob_tracker = {k: [] for k in model.exit_points}

        # Anneal Temperature
        current_temp = config['temp_start'] - (epoch * (config['temp_start'] - config['temp_end']) / epochs)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            network_state = net_sim.sample_network_state()
            
            optimizer.zero_grad()
            final_pred, exit_preds, split_probs, exit_confidences = model(data, temperature=current_temp)
            
            loss, exp_lat_mean_sec, exp_lat_p95_sec, norm_lat_mean, norm_lat_p95, acc_loss = criterion(
                final_pred, exit_preds, split_probs, exit_confidences, target, network_state
            )

            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()

            # Avg latency: keep a sample-weighted mean over the epoch
            total_latency += exp_lat_mean_sec * len(data)   # seconds * batch_size

            # Normalized avg latency: same weighting
            sum_norm_lat += norm_lat_mean * len(data)

            # p95 terms: average over batches (each batch's distribution)
            sum_raw_p95 += exp_lat_p95_sec
            sum_norm_p95 += norm_lat_p95

            num_batches += 1
            
            # Simple Accuracy (Final Head)
            pred = final_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)

            # Log exit probabilities
            for k, v in exit_confidences.items():
                exit_prob_tracker[k].extend(v.detach().cpu().numpy().tolist())

        # --- EPOCH STATS ---
        avg_acc = 100. * correct / total_samples

        # Actual average expected latency over epoch (seconds → ms)
        avg_lat_sec = total_latency / total_samples
        avg_lat_ms = (total_latency / total_samples) * 1000
        
        # Sample-weighted normalized avg latency over epoch
        normalized_avg_lat = sum_norm_lat / total_samples  # dimensionless

        # Average of per-batch p95 (approximate epoch p95)
        avg_p95_lat_sec = sum_raw_p95 / num_batches
        avg_p95_lat_ms = avg_p95_lat_sec * 1000.0

        normalized_p95_lat = sum_norm_p95 / num_batches  # dimensionless

        # Determine best split and full split distribution
        with torch.no_grad():
            split_probs_soft = torch.nn.functional.softmax(model.split_logits, dim=0)
            best_split = torch.argmax(split_probs_soft).item()
            split_conf = split_probs_soft[best_split].item()
            split_probs_list = split_probs_soft.detach().cpu().tolist()

            # Exit thresholds/scales per block index
            sorted_exits = sorted(model.exit_points)
            exit_thresholds = {}
            exit_scales = {}
            for idx, block_idx in enumerate(sorted_exits):
                exit_thresholds[int(block_idx)] = float(model.exit_threshold[idx].item())
                exit_scales[int(block_idx)] = float(model.exit_scale[idx].item())

        # Calculate Average Exit Probabilities (per head)
        avg_exit_probs = {}
        for k in sorted(exit_prob_tracker.keys()):
            if len(exit_prob_tracker[k]) > 0:
                avg_exit_probs[int(k)] = float(np.mean(exit_prob_tracker[k]))
            else:
                avg_exit_probs[int(k)] = 0.0

        probs_str = str([f"{avg_exit_probs[int(k)]:.2f}" for k in sorted(exit_prob_tracker.keys())])

        print(
            f"{epoch+1:<6} | {avg_acc:<7.1f}%"
            f" | {avg_lat_ms:<9.2f} ms"
            f" | p95 {avg_p95_lat_ms:<9.2f} ms"
            f" | Block {best_split} ({split_conf:.2f})"
            f" | {probs_str}"
        )

        # Save logs
        log_entry = {
            "epoch": epoch + 1,
            "accuracy": avg_acc,
            "latency_ms": avg_lat_ms,
            "p95_latency_ms": avg_p95_lat_ms,                 # actual p95 expected latency
            "normalized_latency": normalized_avg_lat,         # E[T]/SLO
            "normalized_p95_latency": normalized_p95_lat,     # p95(T)/SLO
            "split_decision": best_split,
            "split_confidence": split_conf,
            "split_probs": split_probs_list,
            "exit_probs_avg": avg_exit_probs,
            "exit_thresholds": exit_thresholds,
            "exit_scales": exit_scales,
            "loss": total_loss / len(train_loader)
        }
        training_history.append(log_entry)

    # Save Model
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    
    return training_history