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

    default_experiment = [
        {"name": "default", "slo": Config.DEFAULT_SLO, "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, 
         "data": Config.DEFAULT_DATASET, "edge_slowdown": Config.DEFAULT_SLOWDOWN}, # Baseline
    ]

    # ABLATION A: SLO SENSITIVITY
    ablation_a = []
    for slo_ablation in Config.SLO_ABLATIONS:
        ablation_a.append({
            "name": f"A_slo_{int(slo_ablation):02d}", "slo": slo_ablation, "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, 
            "data": Config.DEFAULT_DATASET, "edge_slowdown": Config.DEFAULT_SLOWDOWN})

    # ABLATION B: NETWORK CONDITIONS
    ablation_b = []
    for net_name, net_ablation in Config.NET_ABLATIONS.items():
        rtt_ablation, bw_ablation = net_ablation
        ablation_b.append({
            "name": f"B_net_{net_name}", "slo": Config.DEFAULT_SLO, "rtt": rtt_ablation, "bw": bw_ablation, 
            "data": Config.DEFAULT_DATASET, "edge_slowdown": Config.DEFAULT_SLOWDOWN})

    # ABLATION C: DATASET DIFFICULTY
    ablation_c = []
    for data_ablation in Config.DATA_ABLATIONS:
        ablation_c.append({
            "name": f"C_data_{data_ablation}", "slo": Config.DEFAULT_SLO, "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, 
            "data": data_ablation, "edge_slowdown": Config.DEFAULT_SLOWDOWN})

    # ABLATION D: EDGE SLOWDOWN SENSITIVITY
    ablation_d = []
    for edge_ablation in Config.EDGE_SLOWDOWN_ABLATIONS:
        ablation_d.append({
            "name": f"D_edge_{int(edge_ablation):02d}", "slo": Config.DEFAULT_SLO, "rtt": Config.DEFAULT_RTT, "bw": Config.DEFAULT_BW, 
            "data": Config.DEFAULT_DATASET, "edge_slowdown": edge_ablation})
        
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