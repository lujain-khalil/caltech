import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from .data_utils import get_dataloaders

def evaluate_model(model, net_sim, profiles, device, dataset_name="fashionmnist", batch_size=64):
    """
    Simulates inference on the test set.
    """
    model.eval()
    
    # Load Test Data
    transform = transforms.Compose([
        transforms.Resize((28, 28)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    _, test_loader, _, _ = get_dataloaders(
        dataset_name=profiles.get('dataset_used', 'fashionmnist'), # Fallback or pass from config
        batch_size=batch_size,
    )
    
    # Metrics
    total_samples = 0
    correct = 0
    latencies = []
    exit_counts = {k: 0 for k in model.exit_points} # How many times did we exit at 1? at 3?
    exit_counts['final'] = 0
    
    # 1. Determine the Fixed Split Point (The one with highest logit)
    split_probs = F.softmax(model.split_logits, dim=0)
    split_point = torch.argmax(split_probs).item()
    
    print(f"Running Evaluation. Fixed Split Point: Block {split_point}")

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
                    # We need to find which index in exit_scale corresponds to block i
                    # Since exit_points is a set/list, we need the index mapping
                    # Assuming exit_points passed to model were sorted or we can find index
                    # Let's rely on the model finding it. 
                    # Note: In deployment_model.py, we iterated. Here we need to map i -> param_index
                    # Simplification: We access parameters directly if we stored mapping, 
                    # but for now let's grab the raw params from the list logic.
                    # We'll assume exit_points were [1, 3] so index 0 is 1, index 1 is 3.
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
                        
                        # Mark as exited
                        exited_mask = exited_mask | should_exit

            # --- FINAL CLASSIFIER (Cloud) ---
            # If anyone is left
            if (~exited_mask).any():
                x_final = model.backbone.avgpool(x)
                x_final = torch.flatten(x_final, 1)
                out_final = model.backbone.fc(x_final)
                
                final_preds[~exited_mask] = out_final
                exit_counts['final'] += (~exited_mask).sum().item()
            
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
    
    results = {
        "test_accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "exit_distribution": exit_counts,
        "split_point": split_point
    }
    
    return results