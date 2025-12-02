# import argparse
# import os
# import json
# import torch
# import sys

# # Import from src
# import src.config as Config
# from src.resnet_split import SplittableResNet18
# from src.deployment_model import DeploymentAwareResNet
# from src.network_sim import NetworkSimulator
# from src.train_engine import train_model
# from src.evaluate import evaluate_model

# def main():
#     parser = argparse.ArgumentParser(description="Run Deployment-Aware Edge/Cloud Experiment")
    
#     # Dataset Choice
#     parser.add_argument("--dataset", type=str, default=Config.DEFAULT_DATASET, choices=["mnist", "fmnist", "cifar10"])
    
#     # Experiment Settings
#     parser.add_argument("--exp_name", type=str, default="default_run")
#     parser.add_argument("--epochs", type=int, default=Config.DEFAULT_EPOCHS)
#     parser.add_argument("--batch_size", type=int, default=Config.DEFAULT_BATCH_SIZE)
#     parser.add_argument("--lr", type=float, default=Config.DEFAULT_LR)
    
#     # SLO & Network Settings
#     parser.add_argument("--slo_ms", type=float, default=Config.DEFAULT_SLO)
#     parser.add_argument("--rtt_ms", type=float, default=Config.DEFAULT_RTT)
#     parser.add_argument("--bw_mbps", type=float, default=Config.DEFAULT_BW)
    
#     # Loss Weights
#     parser.add_argument("--lambda_lat", type=float, default=Config.DEFAULT_LAMBDA_LAT)
#     parser.add_argument("--mu_slo", type=float, default=Config.DEFAULT_MU)

#     # Hardware profile / edge slowdown (for ablation E / logging)
#     parser.add_argument("--profile_file", type=str, default="latency_profile.json")
#     parser.add_argument("--edge_slowdown", type=float, default=Config.DEFAULT_SLOWDOWN)

#     parser.add_argument("--attempt_num", type=str, default=Config.DEFAULT_ATTEMPT_NUM)
    
#     args = parser.parse_args()

#     # 1. Load Universal Profile
#     profile_file = args.profile_file
    
#     if not os.path.exists(profile_file):
#         print(f"Error: {profile_file} not found. Run profile_env.py first or pass correct --profile_file.")
#         sys.exit(1)

#     print(f"Loading Universal Profile: {profile_file}")
#     with open(profile_file, "r") as f:
#         profiles = json.load(f)
        
#     profiles['dataset_used'] = args.dataset

#     # 2. Setup Directory
#     run_dir = os.path.join("experiments", f"attempt_{args.attempt_num}", f"{args.exp_name}")
#     os.makedirs(run_dir, exist_ok=True)
#     print(f"Experiment Directory: {run_dir}")
    
#     # Config
#     config = {
#         "dataset": args.dataset,
#         "epochs": args.epochs,
#         "batch_size": args.batch_size,
#         "lr": args.lr,
#         "slo_target": args.slo_ms / 1000.0,
#         "avg_rtt": args.rtt_ms,
#         "avg_bw": args.bw_mbps,
#         "lambda_lat": args.lambda_lat,
#         "mu_slo": args.mu_slo,
#         "temp_start": 5.0,
#         "temp_end": 0.1,
#         "edge_slowdown": args.edge_slowdown,
#     }
    
#     with open(os.path.join(run_dir, "config.json"), "w") as f:
#         json.dump(config, f, indent=4)

#     # 3. Init Resources
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net_sim = NetworkSimulator(avg_bw_mbps=args.bw_mbps, avg_rtt_ms=args.rtt_ms)

#     # Init Model
#     # ALWAYS use input_channels=3 because we duplicate grayscale channels
#     backbone = SplittableResNet18(num_classes=10, input_channels=3, pretrained=True)
#     model = DeploymentAwareResNet(backbone, num_classes=10)
    
#     config['model_instance'] = model

#     # 4. Run Training
#     print("--- Phase 1: Training ---")
#     train_history = train_model(config, profiles, net_sim, device, run_dir)
    
#     with open(os.path.join(run_dir, "train_log.json"), "w") as f:
#         json.dump(train_history, f, indent=4)

#     # 5. Run Evaluation
#     print("\n--- Phase 2: Evaluation ---")
#     test_results = evaluate_model(model, net_sim, profiles, device, dataset_name=args.dataset)
    
#     with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
#         json.dump(test_results, f, indent=4)

#     print(f"\nExperiment Complete. Results saved in {run_dir}")

# if __name__ == "__main__":
#     main()

import argparse
import os
import json
import torch
import sys

# Import from src
import src.config as Config
from src.resnet_split import SplittableResNet18
from src.deployment_model import DeploymentAwareResNet
from src.network_sim import NetworkSimulator
from src.train_engine import train_model
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Run Deployment-Aware Edge/Cloud Experiment")
    
    # Dataset Choice
    parser.add_argument("--dataset", type=str, default=Config.DEFAULT_DATASET, 
                        choices=["mnist", "fmnist", "cifar10"])
    
    # Experiment Settings
    parser.add_argument("--exp_name", type=str, default="default_run")
    parser.add_argument("--epochs", type=int, default=Config.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.DEFAULT_LR)
    
    # SLO & Network Settings
    parser.add_argument("--slo_ms", type=float, default=Config.DEFAULT_SLO)
    parser.add_argument("--rtt_ms", type=float, default=Config.DEFAULT_RTT)
    parser.add_argument("--bw_mbps", type=float, default=Config.DEFAULT_BW)
    
    # Loss Weights (with new defaults)
    parser.add_argument("--lambda_lat", type=float, default=Config.DEFAULT_LAMBDA_LAT)
    parser.add_argument("--mu_slo", type=float, default=Config.DEFAULT_MU)

    # Hardware profile / edge slowdown
    parser.add_argument("--profile_file", type=str, default="latency_profile.json")
    parser.add_argument("--edge_slowdown", type=float, default=Config.DEFAULT_SLOWDOWN)
    
    # NEW: Exit point configuration
    parser.add_argument("--exit_points", type=int, nargs='+', 
                        default=Config.DEFAULT_EXIT_POINTS,
                        help="Block indices for exit placement (e.g., 2 3 4)")

    parser.add_argument("--attempt_num", type=str, default=Config.DEFAULT_ATTEMPT_NUM)
    
    args = parser.parse_args()

    # 1. Load Profile
    profile_file = args.profile_file
    
    if not os.path.exists(profile_file):
        print(f"Error: {profile_file} not found. Run profile_env.py first.")
        sys.exit(1)

    print(f"Loading Profile: {profile_file}")
    with open(profile_file, "r") as f:
        profiles = json.load(f)
        
    profiles['dataset_used'] = args.dataset

    # 2. Setup Directory
    run_dir = os.path.join("experiments", f"attempt_{args.attempt_num}", f"{args.exp_name}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Experiment Directory: {run_dir}")
    
    # Config
    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "slo_target": args.slo_ms / 1000.0,
        "avg_rtt": args.rtt_ms,
        "avg_bw": args.bw_mbps,
        "lambda_lat": args.lambda_lat,
        "mu_slo": args.mu_slo,
        "temp_start": 5.0,
        "temp_end": 0.1,
        "edge_slowdown": args.edge_slowdown,
        "exit_points": args.exit_points,
    }
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 3. Init Resources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_sim = NetworkSimulator(avg_bw_mbps=args.bw_mbps, avg_rtt_ms=args.rtt_ms)

    # Init Model with better exit placement
    backbone = SplittableResNet18(num_classes=10, input_channels=3, pretrained=True)
    
    print(f"\nInitializing model with exits at blocks: {args.exit_points}")
    print(f"  Block 0: conv1 + bn + relu + maxpool")
    print(f"  Block 1: layer1 (64 channels)")
    print(f"  Block 2: layer2 (128 channels)")
    print(f"  Block 3: layer3 (256 channels)")
    print(f"  Block 4: layer4 (512 channels)")
    print(f"\nRecommendation: Use [2, 3, 4] for better feature depth before exiting")
    
    model = DeploymentAwareResNet(
        backbone, 
        num_classes=10, 
        exit_points=args.exit_points  # Now configurable!
    )
    
    config['model_instance'] = model

    # 4. Run Training
    print("\n" + "="*80)
    print("STARTING EXPERIMENT")
    print("="*80)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"SLO Target: {args.slo_ms}ms")
    print(f"Network: RTT={args.rtt_ms}ms, BW={args.bw_mbps}Mbps")
    print(f"Edge Slowdown: {args.edge_slowdown}x")
    print(f"Loss Weights: λ_lat={args.lambda_lat}, μ_slo={args.mu_slo}")
    print("="*80)
    
    train_history = train_model(config, profiles, net_sim, device, run_dir)
    
    with open(os.path.join(run_dir, "train_log.json"), "w") as f:
        json.dump(train_history, f, indent=4)

    # 5. Run Evaluation
    print("\n" + "="*80)
    print("EVALUATION ON TEST SET")
    print("="*80)
    test_results = evaluate_model(
        model, net_sim, profiles, device, 
        dataset_name=args.dataset
    )
    
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Test Accuracy: {test_results['test_accuracy']:.2f}%")
    print(f"Avg Latency: {test_results['avg_latency_ms']:.2f}ms (Target: {args.slo_ms}ms)")
    print(f"P95 Latency: {test_results['p95_latency_ms']:.2f}ms")
    print(f"Exit Distribution: {test_results['exit_distribution']}")
    print(f"Split Block: {test_results['split_block']}")
    print(f"{'='*80}")
    print(f"\nExperiment Complete. Results saved in {run_dir}")

if __name__ == "__main__":
    main()