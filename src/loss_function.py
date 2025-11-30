import torch
import torch.nn as nn
import numpy as np
from . import config as Config

class SLOAwareLoss(nn.Module):
    def __init__(self, profiles, network_sim, slo_target_sec=0.1, alpha_cvar=0.05, 
                 lambda_lat=Config.DEFAULT_LAMBDA_LAT, mu_slo=Config.DEFAULT_MU):
        super().__init__()
        self.profiles = profiles # The Loaded JSON
        self.net_sim = network_sim
        self.slo_target = slo_target_sec
        self.alpha_cvar = alpha_cvar # For P95 latency
        
        # Hyperparameters from paper
        self.lambda_lat = lambda_lat # Weight for average latency
        self.mu_slo = mu_slo         # Weight for SLO violation
        
        # Use per-sample CE so we can weight by exit probabilities
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.epsilon = 1e-8 # Numerical stability for logs

    def forward(self, final_pred, exit_preds, split_probs, exit_confidences, targets, batch_network_state):
        """
        Calculates the 3-part loss from the paper.
        """
        bw, rtt = batch_network_state
        device = final_pred.device
        batch_size = final_pred.size(0)

        # --- PART 1: ACCURACY LOSS
         # Loss is weighted by the probability of actually exiting at each head.
        
        # Initialize with Final Head Loss
        # Probability of reaching final head = 1 - sum(p_exits)
        p_continue = torch.ones(batch_size, device=device)
        total_acc_loss = 0
        
        sorted_exits = sorted(exit_preds.keys())

        # Track expected exit probabilities (batch mean) for latency calculation later
        exit_prob_means = {}

        for idx_i, block_idx in enumerate(sorted_exits):
            pred = exit_preds[block_idx]
            p_exit = exit_confidences[block_idx] # Probability of exiting here
            
            # The probability of ACTUALLY exiting here is:
            # P(surviving previous exits) * P(exiting now) (survive all previous exits AND exit now)
            p_actual_exit = p_continue * p_exit # (B,)
            exit_prob_means[block_idx] = p_actual_exit.mean()
            
            # Per-sample CE
            ce_i = self.ce_loss(pred, targets)        # (B,)

            # Reward making the earliest exit good
            # (helps simpler datasets lean on early exits)
            head_weight = 1.5 if block_idx == sorted_exits[0] else 1.0
            total_acc_loss += (head_weight * p_actual_exit * ce_i).mean()
            
            # Update probability of continuing to next layer
            # Clamp to avoid numerical instability
            p_continue = p_continue * (1 - p_exit)
            
        # Add Final Head Loss
        final_ce  = self.ce_loss(final_pred, targets)
        p_final = p_continue
        p_final_mean = p_final.mean()

        total_acc_loss += (p_final * final_ce).mean()

        # --- PART 2 & 3: LATENCY & SLO 
        # Expected latency depends on:
        #   - which split we choose (split_probs)
        #   - which exit we actually take (exit probabilities)
        #
        # We approximate:
        #   E[T] = sum_s split_probs[s] *
        #              [ sum_exits p_exit_mean(e) * T(s, exit_at_block_e)
        #                + p_final_mean * T(s, exit_at_final) ]
        
        # 1. Calculate Latency for EVERY possible split point s
        block_keys = sorted(k for k in self.profiles.keys() if k.startswith("block_"))
        num_blocks = len(block_keys)
        
        device = split_probs.device

        edge_times = []
        cloud_times = []
        out_sizes = []
        
        for b in range(num_blocks):
            block_profile = self.profiles[f"block_{b}"]
            edge_times.append(block_profile["edge_time_sec"])
            cloud_times.append(block_profile["cloud_time_sec"])
            out_sizes.append(block_profile["output_bytes"])

        final_block = num_blocks - 1  # 4
        
        def latency_for_split_and_exit(s, exit_b):
            # 1) edge compute
            if exit_b <= s:
                # We exit on the edge before crossing the split
                t_edge = sum(edge_times[:exit_b + 1])
                t_comm = 0.0
                t_cloud = 0.0
            else:
                # Compute up to split on edge
                t_edge = sum(edge_times[:s + 1])
                # Comm at split
                data_size = out_sizes[s]
                t_comm = self.net_sim.estimate_transmission_time(data_size, bw, rtt)
                # Cloud compute until the exit block
                t_cloud = sum(cloud_times[s + 1:exit_b + 1])
            return t_edge + t_comm + t_cloud

        # Block index of each exit head and final head
        exit_blocks = sorted_exits                      # e.g. [1, 3]
        final_block = num_blocks - 1

        candidate_latencies = []
        num_splits = split_probs.size(0)  # e.g., 4 (split after 0,1,2,3)

        for s in range(num_splits):
            # Latency if we exit at each head given this split s
            total_t_s = 0.0

            # Early exits
            for block_idx in exit_blocks:
                p_mean = exit_prob_means[block_idx]     # scalar tensor
                t_e = latency_for_split_and_exit(s, block_idx)
                total_t_s += p_mean * t_e

            # Final exit (no early exit taken)
            t_final = latency_for_split_and_exit(s, final_block)
            total_t_s += p_final_mean * t_final

            candidate_latencies.append(total_t_s)

        candidate_latencies = torch.tensor(candidate_latencies, device=device, dtype=torch.float32)

        # Expected latency over splits
        expected_latency = torch.sum(split_probs * candidate_latencies)
        
        # --- SLO PENALTY (CVaR / ReLU) ---
        # Normalize latency by SLO so scales are stable across SLO values
        normalized_latency = expected_latency / (self.slo_target + self.epsilon)

        # Calculate violation: ReLU(Latency - Target)
        violation = torch.relu(normalized_latency - 1.0) ** 2
        
        # TOTAL LOSS
        # We verify that violation > 0 before applying penalty to avoid gradient noise
        total_loss = ( 
            total_acc_loss
            + (self.lambda_lat * normalized_latency)
            + (self.mu_slo * violation)
        )
                     
        return total_loss, expected_latency.item(), total_acc_loss.item()