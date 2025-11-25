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
    print(f"{'Epoch':<6} | {'Acc':<8} | {'Avg Latency':<12} | {'Split Layer':<15} | {'Exit Probs (Avg)'}")
    print("-" * 85)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_latency = 0
        correct = 0
        total_samples = 0
        
        # Track average exit probabilities per head for this epoch
        exit_prob_tracker = {k: [] for k in model.exit_points}

        # Anneal Temperature
        current_temp = config['temp_start'] - (epoch * (config['temp_start'] - config['temp_end']) / epochs)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            network_state = net_sim.sample_network_state()
            
            optimizer.zero_grad()
            final_pred, exit_preds, split_probs, exit_confidences = model(data, temperature=current_temp)
            
            loss, exp_lat, acc_loss = criterion(
                final_pred, exit_preds, split_probs, exit_confidences, target, network_state
            )
            
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_latency += exp_lat
            
            # Simple Accuracy (Final Head)
            pred = final_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)

            # Log exit probabilities
            for k, v in exit_confidences.items():
                exit_prob_tracker[k].extend(v.detach().cpu().numpy().tolist())

        # --- EPOCH STATS ---
        avg_acc = 100. * correct / total_samples
        avg_lat_ms = (total_latency / len(train_loader)) * 1000
        
        # Determine best split
        with torch.no_grad():
            split_probs_soft = torch.nn.functional.softmax(model.split_logits, dim=0)
            best_split = torch.argmax(split_probs_soft).item()
            split_conf = split_probs_soft[best_split].item()

        # Calculate Average Exit Probabilities
        avg_exit_probs = [np.mean(exit_prob_tracker[k]) for k in sorted(exit_prob_tracker.keys())]
        probs_str = str([f"{p:.2f}" for p in avg_exit_probs])

        print(f"{epoch+1:<6} | {avg_acc:<7.1f}% | {avg_lat_ms:<9.2f} ms | Block {best_split} ({split_conf:.2f}) | {probs_str}")

        # Save logs
        log_entry = {
            "epoch": epoch + 1,
            "accuracy": avg_acc,
            "latency_ms": avg_lat_ms,
            "split_decision": best_split,
            "split_confidence": split_conf,
            "exit_probs": avg_exit_probs,
            "loss": total_loss / len(train_loader)
        }
        training_history.append(log_entry)

    # Save Model
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    
    return training_history