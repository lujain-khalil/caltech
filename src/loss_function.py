# import torch
# import torch.nn as nn
# import numpy as np
# from . import config as Config

# class SLOAwareLoss(nn.Module):
#     def __init__(self, profiles, network_sim, slo_target_sec=0.1, alpha_cvar=0.05, 
#                  lambda_lat=Config.DEFAULT_LAMBDA_LAT, mu_slo=Config.DEFAULT_MU):
#         super().__init__()
#         self.profiles = profiles  # The Loaded JSON
#         self.net_sim = network_sim
#         self.slo_target = slo_target_sec
#         self.alpha_cvar = alpha_cvar  # e.g. 0.05 -> p95

#         # Hyperparameters
#         self.lambda_lat = lambda_lat  # weight for average latency
#         self.mu_slo = mu_slo          # weight for SLO (tail) violation

#         # Per-sample CE so we can weight by exit probabilities
#         self.ce_loss = nn.CrossEntropyLoss(reduction='none')
#         self.epsilon = 1e-8  # numerical stability

#     def forward(self, final_pred, exit_preds, split_probs, exit_confidences, targets, batch_network_state):
#         """
#         Calculates accuracy + latency + SLO (p95) loss.

#         SLO is enforced as: p95(normalized_latency) <= 1
#         where normalized_latency = per-sample expected latency / SLO.
#         """
#         bw, rtt = batch_network_state
#         device = final_pred.device
#         batch_size = final_pred.size(0)

#         # ----------------------
#         # PART 1: ACCURACY LOSS
#         # ----------------------
#         # p_continue: probability (per-sample) of NOT having exited yet
#         p_continue = torch.ones(batch_size, device=device)
#         total_acc_loss = 0.0

#         sorted_exits = sorted(exit_preds.keys())  # e.g., [1, 2, 3]

#         # Track per-sample actual exit probabilities for latency calculation
#         # exit_prob_per_sample[block_idx]: shape (B,)
#         exit_prob_per_sample = {}

#         for idx_i, block_idx in enumerate(sorted_exits):
#             pred = exit_preds[block_idx]                  # (B, num_classes)
#             p_exit = exit_confidences[block_idx]          # (B,)

#             # Probability of actually exiting here (per sample):
#             # P(survive all previous exits) * P(exit at this head)
#             p_actual_exit = p_continue * p_exit          # (B,)
#             exit_prob_per_sample[block_idx] = p_actual_exit

#             # Per-sample CE
#             ce_i = self.ce_loss(pred, targets)           # (B,)
#             total_acc_loss += (p_actual_exit * ce_i).mean()

#             # Update probability of continuing to next layer
#             p_continue = p_continue * (1.0 - p_exit)

#         # Final head CE weighted by probability of reaching final head
#         final_ce = self.ce_loss(final_pred, targets)      # (B,)
#         p_final = p_continue                              # (B,)
#         total_acc_loss += (p_final * final_ce).mean()

#         # ----------------------------
#         # PART 2 & 3: LATENCY & SLO
#         # ----------------------------
#         # We now build a per-sample expected latency L_i.
#         #
#         # Latency depends on:
#         #   - split point s (soft, via split_probs)
#         #   - which exit we take (soft, via p_actual_exit and p_final)
#         #
#         # For each split s and exit block e:
#         #   T(s, e) = edge + comm + cloud (deterministic given bw, rtt, profiles)
#         # Then for each sample:
#         #   E[T | s] = sum_e p_actual_exit_i(e) * T(s, e) + p_final_i * T(s, final)
#         #   L_i = E_s[ E[T | s] ] = sum_s split_probs[s] * E[T | s].

#         # 1. Precompute edge/cloud times and output sizes from profiles
#         block_keys = sorted(k for k in self.profiles.keys() if k.startswith("block_"))
#         num_blocks = len(block_keys)

#         edge_times = []
#         cloud_times = []
#         out_sizes = []
#         for b in range(num_blocks):
#             block_profile = self.profiles[f"block_{b}"]
#             edge_times.append(block_profile["edge_time_sec"])
#             cloud_times.append(block_profile["cloud_time_sec"])
#             out_sizes.append(block_profile["output_bytes"])

#         edge_times = np.array(edge_times)
#         cloud_times = np.array(cloud_times)
#         out_sizes = np.array(out_sizes)

#         final_block = num_blocks - 1  # index of final block

#         def latency_for_split_and_exit(s, exit_b):
#             """
#             Latency if we split after block s and exit at block exit_b.
#             Times are deterministic given profiles + (bw, rtt).
#             """
#             if exit_b <= s:
#                 # Exit on edge before crossing split
#                 t_edge = edge_times[:exit_b + 1].sum()
#                 t_comm = 0.0
#                 t_cloud = 0.0
#             else:
#                 # Compute up to split on edge
#                 t_edge = edge_times[:s + 1].sum()
#                 # Comm at split
#                 data_size = out_sizes[s]
#                 t_comm = self.net_sim.estimate_transmission_time(data_size, bw, rtt)
#                 # Cloud compute until exit block
#                 t_cloud = cloud_times[s + 1:exit_b + 1].sum()
#             return float(t_edge + t_comm + t_cloud)

#         # Block index for each exit head and the final head
#         exit_blocks = sorted_exits  # e.g., [1, 2, 3]
#         num_exits = len(exit_blocks)

#         # 2. Precompute T(s, e) for all splits s and all exit events e (including final)
#         num_splits = split_probs.size(0)  # e.g., 4 (split after 0,1,2,3)
#         # T_matrix[s, k]: latency if we split at s and take event k
#         # events k = 0..num_exits-1: early exits at exit_blocks[k]
#         # event k = num_exits: final head
#         T_matrix = torch.zeros(num_splits, num_exits + 1, device=device, dtype=torch.float32)

#         for s in range(num_splits):
#             # Early exits
#             for k, block_idx in enumerate(exit_blocks):
#                 T_matrix[s, k] = latency_for_split_and_exit(s, block_idx)
#             # Final exit (no early exit taken)
#             T_matrix[s, num_exits] = latency_for_split_and_exit(s, final_block)

#         # 3. Build per-sample event probabilities
#         # event_probs[k, i] = P(sample i exits via event k)
#         # k = 0..num_exits-1: early exits, k = num_exits: final head
#         event_probs = []
#         for block_idx in exit_blocks:
#             event_probs.append(exit_prob_per_sample[block_idx])  # (B,)
#         event_probs.append(p_final)  # final head
#         event_probs = torch.stack(event_probs, dim=0)            # (num_exits+1, B)

#         # 4. For each split s, compute per-sample E[T | s]
#         # per_split_latencies[s, i] = sum_k event_probs[k, i] * T_matrix[s, k]
#         # Shape: (num_splits, B)
#         per_split_latencies = torch.einsum('sk,kb->sb', T_matrix, event_probs)

#         # 5. Now average over splits with split_probs
#         # L[i] = sum_s split_probs[s] * per_split_latencies[s, i]
#         L = torch.matmul(split_probs.view(1, -1), per_split_latencies).squeeze(0)  # (B,)

#         # Mean expected latency (for logging)
#         expected_latency_mean = L.mean()
#         expected_latency_p95 = torch.quantile(L, 0.95)
        
#         # Normalize per-sample latencies by SLO (dimensionless)
#         normalized_latencies = L / (self.slo_target + self.epsilon)  # (B,)
#         normalized_latency_mean = normalized_latencies.mean()
#         normalized_latency_p95 = torch.quantile(normalized_latencies, 1.0 - self.alpha_cvar)

#         # Violation if p95 > 1.0
#         violation = torch.relu(normalized_latency_p95 - 1.0) ** 2

#         # ----------------------
#         # TOTAL LOSS
#         # ----------------------
#         total_loss = (
#             total_acc_loss
#             + self.lambda_lat * normalized_latency_mean   # mean latency term
#             + self.mu_slo * violation                    # p95 SLO term
#         )

#         # Return: total loss, mean expected latency (seconds), and accuracy loss
#         return (
#             total_loss, 
#             expected_latency_mean.item(), 
#             expected_latency_p95.item(),
#             normalized_latency_mean.item(),
#             normalized_latency_p95.item(),
#             total_acc_loss.item(),
#         )

import torch
import torch.nn as nn
import numpy as np

class SLOAwareLoss(nn.Module):
    def __init__(self, profiles, network_sim, slo_target_sec=0.1, alpha_cvar=0.05, 
                 lambda_lat=0.5, mu_slo=1.0, min_accuracy_weight=5.0):
        """
        Key changes:
        - Reduced lambda_lat and mu_slo to not overwhelm accuracy
        - Added min_accuracy_weight to protect against accuracy collapse
        - More balanced loss components
        """
        super().__init__()
        self.profiles = profiles
        self.net_sim = network_sim
        self.slo_target = slo_target_sec
        self.alpha_cvar = alpha_cvar

        # REBALANCED: Lower latency penalties
        self.lambda_lat = lambda_lat  # Reduced from 2.0
        self.mu_slo = mu_slo          # Reduced from 10.0
        self.min_accuracy_weight = min_accuracy_weight  # NEW: Protect accuracy

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.epsilon = 1e-8

    def forward(self, final_pred, exit_preds, split_probs, exit_confidences, targets, batch_network_state):
        bw, rtt = batch_network_state
        device = final_pred.device
        batch_size = final_pred.size(0)

        # ----------------------
        # PART 1: ACCURACY LOSS (with diversity incentive)
        # ----------------------
        p_continue = torch.ones(batch_size, device=device)
        total_acc_loss = 0.0
        
        # Track exit distribution for diversity
        exit_distributions = []

        sorted_exits = sorted(exit_preds.keys())
        exit_prob_per_sample = {}

        for idx_i, block_idx in enumerate(sorted_exits):
            pred = exit_preds[block_idx]
            p_exit = exit_confidences[block_idx]

            p_actual_exit = p_continue * p_exit
            exit_prob_per_sample[block_idx] = p_actual_exit
            exit_distributions.append(p_actual_exit.mean())

            ce_i = self.ce_loss(pred, targets)
            total_acc_loss += (p_actual_exit * ce_i).mean()

            p_continue = p_continue * (1.0 - p_exit)

        final_ce = self.ce_loss(final_pred, targets)
        p_final = p_continue
        total_acc_loss += (p_final * final_ce).mean()
        
        # NEW: Diversity bonus - encourage using multiple exits
        # Entropy of exit distribution (higher = more diverse)
        exit_dist_tensor = torch.stack(exit_distributions + [p_final.mean()])
        exit_dist_normalized = exit_dist_tensor / (exit_dist_tensor.sum() + self.epsilon)
        diversity_entropy = -(exit_dist_normalized * torch.log(exit_dist_normalized + self.epsilon)).sum()
        max_entropy = np.log(len(exit_distributions) + 1)
        diversity_bonus = -0.1 * (max_entropy - diversity_entropy)  # Penalty for low diversity

        # ----------------------------
        # PART 2 & 3: LATENCY & SLO
        # ----------------------------
        block_keys = sorted(k for k in self.profiles.keys() if k.startswith("block_"))
        num_blocks = len(block_keys)

        edge_times = []
        cloud_times = []
        out_sizes = []
        for b in range(num_blocks):
            block_profile = self.profiles[f"block_{b}"]
            edge_times.append(block_profile["edge_time_sec"])
            cloud_times.append(block_profile["cloud_time_sec"])
            out_sizes.append(block_profile["output_bytes"])

        edge_times = np.array(edge_times)
        cloud_times = np.array(cloud_times)
        out_sizes = np.array(out_sizes)

        final_block = num_blocks - 1

        def latency_for_split_and_exit(s, exit_b):
            if exit_b <= s:
                t_edge = edge_times[:exit_b + 1].sum()
                t_comm = 0.0
                t_cloud = 0.0
            else:
                t_edge = edge_times[:s + 1].sum()
                data_size = out_sizes[s]
                t_comm = self.net_sim.estimate_transmission_time(data_size, bw, rtt)
                t_cloud = cloud_times[s + 1:exit_b + 1].sum()
            return float(t_edge + t_comm + t_cloud)

        exit_blocks = sorted_exits
        num_exits = len(exit_blocks)

        num_splits = split_probs.size(0)
        T_matrix = torch.zeros(num_splits, num_exits + 1, device=device, dtype=torch.float32)

        for s in range(num_splits):
            for k, block_idx in enumerate(exit_blocks):
                T_matrix[s, k] = latency_for_split_and_exit(s, block_idx)
            T_matrix[s, num_exits] = latency_for_split_and_exit(s, final_block)

        event_probs = []
        for block_idx in exit_blocks:
            event_probs.append(exit_prob_per_sample[block_idx])
        event_probs.append(p_final)
        event_probs = torch.stack(event_probs, dim=0)

        per_split_latencies = torch.einsum('sk,kb->sb', T_matrix, event_probs)
        L = torch.matmul(split_probs.view(1, -1), per_split_latencies).squeeze(0)

        expected_latency_mean = L.mean()
        expected_latency_p95 = torch.quantile(L, 0.95)
        
        normalized_latencies = L / (self.slo_target + self.epsilon)
        normalized_latency_mean = normalized_latencies.mean()
        normalized_latency_p95 = torch.quantile(normalized_latencies, 1.0 - self.alpha_cvar)

        # MODIFIED: Softer SLO violation penalty
        # Only penalize if we're significantly over SLO
        violation = torch.relu(normalized_latency_p95 - 1.0)
        
        # NEW: Adaptive penalty based on how much we violate
        if normalized_latency_p95 > 1.2:  # More than 20% over SLO
            slo_penalty = self.mu_slo * (violation ** 2)
        else:
            slo_penalty = self.mu_slo * violation  # Linear for small violations

        # ----------------------
        # TOTAL LOSS (REBALANCED)
        # ----------------------
        total_loss = (
            self.min_accuracy_weight * total_acc_loss  # NEW: Higher weight on accuracy
            + self.lambda_lat * normalized_latency_mean
            + slo_penalty
            + diversity_bonus  # NEW: Encourage exit diversity
        )

        return (
            total_loss, 
            expected_latency_mean.item(), 
            expected_latency_p95.item(),
            normalized_latency_mean.item(),
            normalized_latency_p95.item(),
            total_acc_loss.item(),
        )