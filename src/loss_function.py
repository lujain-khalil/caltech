import torch
import torch.nn as nn
import numpy as np

class SLOAwareLoss(nn.Module):
    def __init__(self, profiles, network_sim, slo_target_sec=0.1, alpha_cvar=0.05, lambda_lat=0.1, mu_slo=1.0):
        super().__init__()
        self.profiles = profiles # The Loaded JSON
        self.net_sim = network_sim
        self.slo_target = slo_target_sec
        self.alpha_cvar = alpha_cvar # For P95 latency
        
        # Hyperparameters from paper [cite: 155, 157]
        self.lambda_lat = lambda_lat # Weight for average latency
        self.mu_slo = mu_slo         # Weight for SLO violation
        
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, final_pred, exit_preds, split_probs, exit_confidences, targets, batch_network_state):
        """
        Calculates the 3-part loss from the paper.
        """
        bw, rtt = batch_network_state
        device = final_pred.device
        batch_size = final_pred.size(0)

        # --- PART 1: ACCURACY LOSS
        # Loss is weighted average of all heads (early exits + final)
        
        # Initialize with Final Head Loss
        # Probability of reaching final head = 1 - sum(p_exits)
        p_continue = torch.ones(batch_size, device=device)
        total_acc_loss = 0
        
        sorted_exits = sorted(exit_preds.keys())
        
        for idx in sorted_exits:
            pred = exit_preds[idx]
            p_exit = exit_confidences[idx] # Probability of exiting here
            
            # The probability of ACTUALLY exiting here is:
            # P(surviving previous exits) * P(exiting now)
            p_actual_exit = p_continue * p_exit
            
            loss_i = self.ce_loss(pred, targets)
            total_acc_loss += (p_actual_exit * loss_i).mean()
            
            # Update probability of continuing to next layer
            p_continue = p_continue * (1 - p_exit)
            
        # Add Final Head Loss
        loss_final = self.ce_loss(final_pred, targets)
        total_acc_loss += (p_continue * loss_final).mean()

        # --- PART 2 & 3: LATENCY & SLO 
        # We must calculate expected latency T(s) for every sample based on split_probs
        
        # 1. Calculate Latency for EVERY possible split point s
        # This allows us to take a weighted average using split_probs
        candidate_latencies = []
        
        # Iterate through every possible split block index
        num_blocks = len(split_probs)
        for s in range(num_blocks):
            # Calculate T_edge (sum of blocks <= s)
            t_edge = 0
            for b in range(s + 1):
                t_edge += self.profiles[f"block_{b}"]["edge_time_sec"]
            
            # Calculate T_offload (Comm + Cloud blocks > s)
            # data size at split point s
            data_size = self.profiles[f"block_{s}"]["output_bytes"]
            t_comm = self.net_sim.estimate_transmission_time(data_size, bw, rtt)
            
            t_cloud = 0
            for b in range(s + 1, num_blocks):
                t_cloud += self.profiles[f"block_{b}"]["cloud_time_sec"]
                
            total_t = t_edge + t_comm + t_cloud
            candidate_latencies.append(total_t)
            
        # Convert to tensor
        candidate_latencies = torch.tensor(candidate_latencies, device=device, dtype=torch.float32)
        
        # Expected Latency = dot product of (Split Probs * Candidate Latencies)
        # This gives us a single scalar "Expected Time" for the architecture choice
        # Note: We treat latency as constant across the batch for the split-decision, 
        # but exit-decisions vary per image. 
        # For simplicity (and stability), we apply split logic globally per batch.
        expected_latency = torch.sum(split_probs * candidate_latencies)
        
        # --- SLO PENALTY (CVaR) ---
        # Calculate violation: ReLU(Latency - Target)
        violation = torch.relu(expected_latency - self.slo_target)
        
        # Since expected_latency here is a scalar (averaged over split probs), 
        # CVaR simplifies to just the violation itself in this specific formulation 
        # unless we sample multiple network states. 
        # To strictly follow the paper's CVaR on the batch, we would need 
        # per-sample latencies, but latency is dominated by the shared Network State.
        # We will use the scalar violation for gradient stability.
        
        # TOTAL LOSS
        total_loss = total_acc_loss + \
                     (self.lambda_lat * expected_latency) + \
                     (self.mu_slo * violation)
                     
        return total_loss, expected_latency.item(), total_acc_loss.item()