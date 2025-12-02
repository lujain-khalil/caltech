import torch
import torch.nn as nn
import numpy as np
from . import config as Config

class SLOAwareLoss(nn.Module):
    def __init__(self, profiles, network_sim, slo_target_sec=0.1, alpha_cvar=0.05, 
                 lambda_lat=Config.DEFAULT_LAMBDA_LAT, mu_slo=Config.DEFAULT_MU, min_accuracy_weight=5.0):
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

        # Latency penalties
        self.lambda_lat = lambda_lat 
        self.mu_slo = mu_slo         
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
        
        # ----------------------
        # TOTAL LOSS (REBALANCED)
        # ----------------------
        total_loss = (
            self.min_accuracy_weight * total_acc_loss  # NEW: Higher weight on accuracy
            + self.lambda_lat * normalized_latency_mean
            + self.mu_slo * (violation ** 2)
        )

        return (
            total_loss, 
            expected_latency_mean.item(), 
            expected_latency_p95.item(),
            normalized_latency_mean.item(),
            normalized_latency_p95.item(),
            total_acc_loss.item(),
        )