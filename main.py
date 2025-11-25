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

def main():
    print("### AUTOMATED ABLATION RUNNER ###")
    print("This script will run Profiling + 3 Ablation Studies (A, B, C)")
    
    # 1. Hardware Profiling (Run once)
    run_command(["profile_env.py"], "Hardware Profiling (Universal 3-Channel)")

    # --- EXPERIMENT CONFIGURATIONS ---
    
    # Defaults
    # These match your specified defaults: slo=100, rtt=50, bw=15, dataset=fmnist
    
    # ABLATION A: SLO SENSITIVITY
    # Varying SLO target while keeping network (4G) and dataset (FMNIST) constant.
    ablation_a = [
        {"name": "A_slo_200", "slo": 200, "rtt": 50, "bw": 15, "data": "fmnist"},
        {"name": "default", "slo": 100, "rtt": 50, "bw": 15, "data": "fmnist"}, # Baseline
        {"name": "A_slo_75",  "slo": 75,  "rtt": 50, "bw": 15, "data": "fmnist"},
        {"name": "A_slo_50",  "slo": 50,  "rtt": 50, "bw": 15, "data": "fmnist"}, # Stress Test
    ]

    # ABLATION B: NETWORK CONDITIONS
    # Varying Network (RTT/BW) while keeping SLO (100ms) and dataset (FMNIST) constant.
    ablation_b = [
        {"name": "B_net_fast",   "slo": 100, "rtt": 10,  "bw": 50, "data": "fmnist"}, # 5G/WiFi
        # {"name": "B_net_default","slo": 100, "rtt": 50,  "bw": 15, "data": "fmnist"}, # 4G (Repeat of A_slo_100)
        {"name": "B_net_slow",   "slo": 100, "rtt": 75,  "bw": 3,  "data": "fmnist"}, # 3G
        {"name": "B_net_impossible",    "slo": 100, "rtt": 500, "bw": 1,  "data": "fmnist"}, # Satellite
    ]

    # ABLATION C: DATASET DIFFICULTY
    # Varying Dataset while keeping SLO (100ms) and Network (4G) constant.
    ablation_c = [
        {"name": "C_data_mnist", "slo": 100, "rtt": 50, "bw": 15, "data": "mnist"},
        # {"name": "C_data_fmnist","slo": 100, "rtt": 50, "bw": 15, "data": "fmnist"}, # Repeat of B_net_default
        {"name": "C_data_cifar", "slo": 100, "rtt": 50, "bw": 15, "data": "cifar10"},
    ]

    # Combine all lists
    all_experiments = ablation_a + ablation_b + ablation_c

    # --- EXECUTION LOOP ---
    for i, exp in enumerate(all_experiments):
        cmd = [
            "run_experiment.py",
            "--exp_name", exp["name"],
            "--dataset", exp["data"],
            "--slo_ms", str(exp["slo"]),
            "--rtt_ms", str(exp["rtt"]),
            "--bw_mbps", str(exp["bw"]),
            "--epochs", "30" 
        ]
        
        step_desc = f"Experiment {i+1}/{len(all_experiments)}: {exp['name']}"
        run_command(cmd, step_desc)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED.")
    print("Results are stored in the 'experiments/' directory.")
    print("="*60)

if __name__ == "__main__":
    main()