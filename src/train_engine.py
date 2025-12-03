import torch
import torch.optim as optim
import os
import numpy as np

from .loss_function import SLOAwareLoss
from .data_utils import get_dataloaders
from . import config as Config

def train_model(config, profiles, net_sim, device, save_dir):
    """
    FIXED: Three-stage training to prevent premature convergence to Exit 1
    
    Stage 1: Train backbone + final head only (no exits)
    Stage 2: Freeze backbone, train exit heads separately (deepest first)
    Stage 3: Joint fine-tuning with SLO-aware loss
    """
    
    batch_size = config['batch_size']
    epochs = config['epochs']
    slo_target = config['slo_target']
    
    train_loader, _, _, _ = get_dataloaders(
        dataset_name=config['dataset'],
        batch_size=batch_size,
    )

    model = config['model_instance']
    model.to(device)
    
    # Track which stage we're in  
    stage1_epochs = max(5, int(epochs * Config.STAGE1_RATIO))
    stage2_epochs = max(5, int(epochs * Config.STAGE2_RATIO))
    stage3_epochs = epochs - stage1_epochs - stage2_epochs
    
    training_history = []
    
    print(f"\n{'='*80}")
    print(f"THREE-STAGE TRAINING PROTOCOL")
    print(f"{'='*80}")
    print(f"Stage 1 (Epochs 1-{stage1_epochs}): Train backbone + final head ONLY")
    print(f"Stage 2 (Epochs {stage1_epochs+1}-{stage1_epochs+stage2_epochs}): Train exit heads (backbone frozen)")
    print(f"Stage 3 (Epochs {stage1_epochs+stage2_epochs+1}-{epochs}): Joint fine-tuning with SLO loss")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # STAGE 1: Train Backbone + Final Head Only
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 1: Training Backbone (No Early Exits)")
    print(f"{'='*60}")
    
    # Disable all exit heads during Stage 1
    for param in model.exit_heads.parameters():
        param.requires_grad = False
    
    # Only train backbone
    optimizer_stage1 = optim.Adam(
        model.backbone.parameters(),
        lr=config['lr']
    )
    criterion_ce = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, stage1_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer_stage1.zero_grad()
            
            # Forward through backbone only
            x = data
            for block in model.backbone.blocks:
                x = block(x)
            x = model.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            final_pred = model.backbone.fc(x)
            
            # Only CE loss on final head
            loss = criterion_ce(final_pred, target)
            loss.backward()
            optimizer_stage1.step()
            
            running_loss += loss.item() * data.size(0)
            pred = final_pred.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        acc = 100.0 * correct / total
        avg_loss = running_loss / total
        
        print(f"Epoch {epoch:2d}/{stage1_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        
        training_history.append({
            "epoch": epoch,
            "stage": 1,
            "accuracy": acc,
            "loss": avg_loss,
        })
    
    # ========================================================================
    # STAGE 2: Train Exit Heads (Deepest First, Backbone Frozen)
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 2: Training Exit Heads (Branch-wise, Deepest First)")
    print(f"{'='*60}")
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Enable exit heads
    for param in model.exit_heads.parameters():
        param.requires_grad = True
    
    # Also train exit thresholds and scales
    trainable_params = [
        {'params': model.exit_heads.parameters()},
        {'params': [model.raw_exit_threshold, model.raw_exit_scale]},
    ]
    
    optimizer_stage2 = optim.Adam(trainable_params, lr=config['lr'])
    
    # Train each exit head separately (deepest to shallowest)
    sorted_exits = sorted(model.exit_points, reverse=True)  # [3, 2, 1]
    
    for exit_idx in sorted_exits:
        print(f"\n  Training Exit Head at Block {exit_idx}...")
        
        for epoch in range(1, stage2_epochs + 1):
            model.train()
            running_loss = 0.0
            correct_exit = 0
            correct_final = 0
            total = 0
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer_stage2.zero_grad()
                
                # Forward to this exit point
                x = data
                for i, block in enumerate(model.backbone.blocks):
                    x = block(x)
                    if i == exit_idx:
                        break
                
                # Get prediction from this exit
                exit_pred = model.exit_heads[str(exit_idx)](x)
                
                # Also get final prediction for comparison
                x_temp = x
                for j in range(exit_idx + 1, len(model.backbone.blocks)):
                    x_temp = model.backbone.blocks[j](x_temp)
                x_temp = model.backbone.avgpool(x_temp)
                x_temp = torch.flatten(x_temp, 1)
                final_pred = model.backbone.fc(x_temp)
                
                # Loss: CE on exit + KL divergence to match final head
                ce_loss = criterion_ce(exit_pred, target)
                
                # Distillation from final head (soft targets)
                kd_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(exit_pred / 2.0, dim=1),
                    torch.nn.functional.softmax(final_pred.detach() / 2.0, dim=1),
                    reduction='batchmean'
                ) * (2.0 ** 2)
                
                # Combined loss (more weight on CE for early exits)
                alpha = Config.KD_ALPHA  # Weight for CE loss
                loss = alpha * ce_loss + (1 - alpha) * kd_loss
                
                loss.backward()
                optimizer_stage2.step()
                
                running_loss += loss.item() * data.size(0)
                
                pred_exit = exit_pred.argmax(dim=1)
                pred_final = final_pred.argmax(dim=1)
                correct_exit += pred_exit.eq(target).sum().item()
                correct_final += pred_final.eq(target).sum().item()
                total += target.size(0)
            
            acc_exit = 100.0 * correct_exit / total
            acc_final = 100.0 * correct_final / total
            avg_loss = running_loss / total
            
            print(f"    Epoch {epoch:2d}/{stage2_epochs} | Loss: {avg_loss:.4f} | "
                  f"Exit Acc: {acc_exit:.2f}% | Final Acc: {acc_final:.2f}%")
            
            training_history.append({
                "epoch": stage1_epochs + epoch,
                "stage": 2,
                "training_exit": exit_idx,
                "exit_accuracy": acc_exit,
                "final_accuracy": acc_final,
                "loss": avg_loss,
            })
    
    # ========================================================================
    # STAGE 3: Joint Fine-tuning with SLO-Aware Loss
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 3: Joint Fine-tuning with SLO-Aware Loss")
    print(f"{'='*60}")
    
    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True
    
    # CRITICAL FIX: Much lower weights for latency penalties
    criterion = SLOAwareLoss(
        profiles=profiles, 
        network_sim=net_sim, 
        slo_target_sec=slo_target,
        lambda_lat=config['lambda_lat'] if 'lambda_lat' in config else Config.DEFAULT_LAMBDA_LAT,   
        mu_slo=config['mu_slo'] if 'mu_slo' in config else Config.DEFAULT_MU,
    )
    
    optimizer_stage3 = optim.Adam(model.parameters(), lr=config['lr'] * 0.1)  # Lower LR
    
    print(f"{'Epoch':<6} | {'Acc':<8} | {'Avg Lat':<10} | {'p95 Lat':<10} | "
          f"{'Split':<8} | {'Exit Distribution'}")
    print("-" * 85)
    
    for epoch in range(1, stage3_epochs + 1):
        model.train()
        total_loss = 0
        total_latency = 0
        correct = 0
        total_samples = 0
        
        sum_norm_lat = 0.0
        sum_raw_p95 = 0.0
        sum_norm_p95 = 0.0
        num_batches = 0
        
        exit_prob_tracker = {k: [] for k in model.exit_points}
        
        # Anneal temperature
        current_epoch = stage1_epochs + stage2_epochs + epoch
        current_temp = config['temp_start'] - (
            current_epoch * (config['temp_start'] - config['temp_end']) / epochs
        )
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            network_state = net_sim.sample_network_state()
            
            optimizer_stage3.zero_grad()
            final_pred, exit_preds, split_probs, exit_confidences = model(
                data, temperature=current_temp
            )
            
            loss, exp_lat_mean_sec, exp_lat_p95_sec, norm_lat_mean, norm_lat_p95, acc_loss = criterion(
                final_pred, exit_preds, split_probs, exit_confidences, target, network_state
            )
            
            loss.backward()
            optimizer_stage3.step()
            
            total_loss += loss.item()
            total_latency += exp_lat_mean_sec * len(data)
            sum_norm_lat += norm_lat_mean * len(data)
            sum_raw_p95 += exp_lat_p95_sec
            sum_norm_p95 += norm_lat_p95
            num_batches += 1
            
            pred = final_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)
            
            for k, v in exit_confidences.items():
                exit_prob_tracker[k].extend(v.detach().cpu().numpy().tolist())
        
        # Epoch stats
        avg_acc = 100. * correct / total_samples
        avg_lat_ms = (total_latency / total_samples) * 1000
        avg_p95_lat_ms = (sum_raw_p95 / num_batches) * 1000
        
        with torch.no_grad():
            split_probs_soft = torch.nn.functional.softmax(model.split_logits, dim=0)
            best_split = torch.argmax(split_probs_soft).item()
        
        # Calculate average exit probabilities
        avg_exit_probs = {
            int(k): float(np.mean(exit_prob_tracker[k])) 
            for k in sorted(exit_prob_tracker.keys())
        }
        
        exit_dist_str = ", ".join([f"{k}:{v:.2f}" for k, v in avg_exit_probs.items()])
        
        print(f"{current_epoch:<6} | {avg_acc:<7.1f}% | {avg_lat_ms:<9.2f} | "
              f"{avg_p95_lat_ms:<9.2f} | Blk {best_split:<4} | {exit_dist_str}")
        
        training_history.append({
            "epoch": current_epoch,
            "stage": 3,
            "accuracy": avg_acc,
            "latency_ms": avg_lat_ms,
            "p95_latency_ms": avg_p95_lat_ms,
            "split_decision": best_split,
            "exit_probs_avg": avg_exit_probs,
            "loss": total_loss / len(train_loader)
        })
    
    # Save Model
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Model saved to {model_path}")
    print(f"{'='*60}")
    
    return training_history