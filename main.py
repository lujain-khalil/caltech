import subprocess
import sys
import time
import os

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

def main(attempt_num = None):
    if attempt_num is None:
        print("Attempt number not provided. Set number in main.py __main__.")
        sys.exit(1)

    print("### AUTOMATED ABLATION RUNNER ###")
    print("This script will run Profiling + Ablation Studies")
    
    # --- EXPERIMENT CONFIGURATIONS ---
    default_slo = 70
    default_rtt = 50
    default_bw = 15
    default_mu = 10.0
    default_data = "fmnist"
    default_slowdown = 20.0

    # 1. Hardware Profiling (Run once)
    run_command(["profile_env.py"], f"Hardware Profiling for default EDGE_SLOWDOWN={default_slowdown}")


    default_experiment = [
        {"name": "default", "slo": default_slo, "rtt": default_rtt, "bw": default_bw, "data": default_data, "mu": default_mu, "edge_slowdown": default_slowdown}, # Baseline
    ]

    # ABLATION A: SLO SENSITIVITY
    ablation_a = [
        {"name": "A_slo_100", "slo": 100, "rtt": default_rtt, "bw": default_bw, "data": default_data, "mu": default_mu, "edge_slowdown": default_slowdown}, # Relaxed
        {"name": "A_slo_060",  "slo": 60, "rtt": default_rtt, "bw": default_bw, "data": default_data, "mu": default_mu, "edge_slowdown": default_slowdown},
        {"name": "A_slo_040",  "slo": 40, "rtt": default_rtt, "bw": default_bw, "data": default_data, "mu": default_mu, "edge_slowdown": default_slowdown}, # Stress Test
    ]

    # ABLATION B: NETWORK CONDITIONS
    ablation_b = [
        {"name": "B_net_fast",       "slo": default_slo, "rtt": 10,  "bw": 50, "data": default_data, "mu": default_mu, "edge_slowdown": default_slowdown}, # 5G
        {"name": "B_net_slow",       "slo": default_slo, "rtt": 75,  "bw": 3,  "data": default_data, "mu": default_mu, "edge_slowdown": default_slowdown}, # 3G
        {"name": "B_net_impossible", "slo": default_slo, "rtt": 500, "bw": 1,  "data": default_data, "mu": default_mu, "edge_slowdown": default_slowdown}, # Satellite
    ]

    # ABLATION C: DATASET DIFFICULTY
    ablation_c = [
        {"name": "C_data_mnist", "slo": default_slo, "rtt": default_rtt, "bw": default_bw, "data": "mnist",   "mu": default_mu, "edge_slowdown": default_slowdown},
        {"name": "C_data_cifar", "slo": default_slo, "rtt": default_rtt, "bw": default_bw, "data": "cifar10", "mu": default_mu, "edge_slowdown": default_slowdown},
    ]

    # ABLATION D: EDGE SLOWDOWN SENSITIVITY
    # Here we vary the edge slowdown factor used during profiling.
    # Each experiment will get its own latency_profile JSON.
    ablation_d = [
        {"name": "D_edge_003",   "slo": default_slo, "rtt": default_rtt, "bw": default_bw, "data": default_data, "mu": default_mu, "edge_slowdown": 3.0},
        {"name": "D_edge_050",  "slo": default_slo, "rtt": default_rtt, "bw": default_bw, "data": default_data, "mu": default_mu, "edge_slowdown": 50.0},
        {"name": "D_edge_100", "slo": default_slo, "rtt": default_rtt, "bw": default_bw, "data": default_data, "mu": default_mu, "edge_slowdown": 100.0},
    ]

    # Combine all lists
    all_experiments = default_experiment + ablation_a + ablation_b + ablation_c + ablation_d


    # --- EXECUTION LOOP ---
    for i, exp in enumerate(all_experiments):
        # Default profile file
        profile_file = "latency_profile.json"

        # If this experiment is part of ablation D, re-profile with a custom slowdown
        edge_slow = exp["edge_slowdown"]
        if edge_slow != default_slowdown:
            edge_slow = exp["edge_slowdown"]
            profile_file = f"latency_profile_edge_{edge_slow}.json"

            # Run profiling for this slowdown
            run_command(["profile_env.py", "--edge_slowdown", str(edge_slow), "--output", profile_file],
                        f"Hardware Profiling for {exp['name']} (EDGE_SLOWDOWN={edge_slow})")
            
        
        cmd = [
            "run_experiment.py",
            "--exp_name", exp["name"],
            "--dataset", exp["data"],
            "--slo_ms", str(exp["slo"]),
            "--rtt_ms", str(exp["rtt"]),
            "--bw_mbps", str(exp["bw"]),
            "--mu_slo", str(exp["mu"]), 
            "--epochs", "30", # Standardized on 30 epochs for convergence
            "--edge_slowdown", str(edge_slow),
            "--profile_file", profile_file,
            "--attempt_num", str(attempt_num),
        ]
        
        step_desc = f"Experiment {i+1}/{len(all_experiments)}: {exp['name']}"
        run_command(cmd, step_desc)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED.")
    print("Results are stored in the 'experiments/' directory.")
    print("="*60)

if __name__ == "__main__":
    attempt = 5
    main(attempt_num=attempt)