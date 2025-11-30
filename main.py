import subprocess
import sys
import time
import os
import src.config as Config

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

    # 1. Hardware Profiling (Run once)
    # run_command(["profile_env.py"], f"Hardware Profiling for default EDGE_SLOWDOWN={Config.DEFAULT_SLOWDOWN}")

    default_experiment = [
        {"name": "default", "slo": Config.DEFAULT_SLO, "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, "data": Config.DEFAULT_DATASET, "mu": Config.DEFAULT_MU, "edge_slowdown": Config.DEFAULT_SLOWDOWN}, # Baseline
    ]

    # ABLATION A: SLO SENSITIVITY
    ablation_a = [
        {"name": "A_slo_600", "slo": 600, "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, "data": Config.DEFAULT_DATASET, "mu": Config.DEFAULT_MU, "edge_slowdown": Config.DEFAULT_SLOWDOWN}, # Relaxed
        {"name": "A_slo_100", "slo": 100, "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, "data": Config.DEFAULT_DATASET, "mu": Config.DEFAULT_MU, "edge_slowdown": Config.DEFAULT_SLOWDOWN},
        {"name": "A_slo_040", "slo": 40 , "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, "data": Config.DEFAULT_DATASET, "mu": Config.DEFAULT_MU, "edge_slowdown": Config.DEFAULT_SLOWDOWN}, # Stress Test
    ]

    # ABLATION B: NETWORK CONDITIONS
    ablation_b = [
        {"name": "B_net_fast",       "slo": Config.DEFAULT_SLO, "rtt": 10,  "bw": 50, "data": Config.DEFAULT_DATASET, "mu": Config.DEFAULT_MU, "edge_slowdown": Config.DEFAULT_SLOWDOWN}, # 5G
        {"name": "B_net_slow",       "slo": Config.DEFAULT_SLO, "rtt": 150, "bw": 5,  "data": Config.DEFAULT_DATASET, "mu": Config.DEFAULT_MU, "edge_slowdown": Config.DEFAULT_SLOWDOWN}, # 3G
        {"name": "B_net_impossible", "slo": Config.DEFAULT_SLO, "rtt": 500, "bw": 1,  "data": Config.DEFAULT_DATASET, "mu": Config.DEFAULT_MU, "edge_slowdown": Config.DEFAULT_SLOWDOWN}, # Satellite
    ]

    # ABLATION C: DATASET DIFFICULTY
    ablation_c = [
        {"name": "C_data_mnist", "slo": Config.DEFAULT_SLO, "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, "data": "mnist",   "mu": Config.DEFAULT_MU, "edge_slowdown": Config.DEFAULT_SLOWDOWN},
        {"name": "C_data_cifar10", "slo": Config.DEFAULT_SLO, "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, "data": "cifar10", "mu": Config.DEFAULT_MU, "edge_slowdown": Config.DEFAULT_SLOWDOWN},
    ]

    # ABLATION D: EDGE SLOWDOWN SENSITIVITY
    ablation_d = [
        {"name": "D_edge_003", "slo": Config.DEFAULT_SLO, "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, "data": Config.DEFAULT_DATASET, "mu": Config.DEFAULT_MU, "edge_slowdown": 3.0},
        {"name": "D_edge_010", "slo": Config.DEFAULT_SLO, "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, "data": Config.DEFAULT_DATASET, "mu": Config.DEFAULT_MU, "edge_slowdown": 10.0},
        {"name": "D_edge_100", "slo": Config.DEFAULT_SLO, "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, "data": Config.DEFAULT_DATASET, "mu": Config.DEFAULT_MU, "edge_slowdown": 100.0},
    ]

    # Combine all lists
    all_experiments = default_experiment + ablation_a + ablation_b + ablation_c + ablation_d


    # --- EXECUTION LOOP ---
    for i, exp in enumerate(all_experiments):
        # Default profile file
        profile_file = "latency_profile.json"

        # If this experiment is part of ablation D, re-profile with a custom slowdown
        edge_slow = exp["edge_slowdown"]
        if edge_slow != Config.DEFAULT_SLOWDOWN:
            edge_slow = exp["edge_slowdown"]
            profile_file = f"latency_profile_edge_{edge_slow}.json"            
        
        cmd = [
            "run_experiment.py",
            "--exp_name", exp["name"],
            "--dataset", exp["data"],
            "--slo_ms", str(exp["slo"]),
            "--rtt_ms", str(exp["rtt"]),
            "--bw_mbps", str(exp["bw"]),
            "--mu_slo", str(exp["mu"]), 
            "--epochs", str(Config.DEFAULT_EPOCHS), # Standardized on 30 epochs for convergence
            "--edge_slowdown", str(edge_slow),
            "--profile_file", profile_file,
            "--attempt_num", str(attempt_num),
        ]
        
        step_desc = f"Experiment {i+1}/{len(all_experiments)}: {exp['name']}"
        run_command(cmd, step_desc)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED.")
    print("Results are stored in the 'experiments/' directory.")

if __name__ == "__main__":
    main(attempt_num=8)