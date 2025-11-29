import argparse
import os
import json
import torch
import sys

# Import from src
from src.resnet_split import SplittableResNet18
from src.deployment_model import DeploymentAwareResNet
from src.network_sim import NetworkSimulator
from src.train_engine import train_model
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Run Deployment-Aware Edge/Cloud Experiment")
    
    # Dataset Choice
    parser.add_argument("--dataset", type=str, default="fmnist", choices=["mnist", "fmnist", "cifar10"])
    
    # Experiment Settings
    parser.add_argument("--exp_name", type=str, default="default_run")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    
    # SLO & Network Settings
    parser.add_argument("--slo_ms", type=float, default=70.0)
    parser.add_argument("--rtt_ms", type=float, default=50.0)
    parser.add_argument("--bw_mbps", type=float, default=15.0)
    
    # Loss Weights
    parser.add_argument("--lambda_lat", type=float, default=3.0)
    parser.add_argument("--mu_slo", type=float, default=10.0)

    # Hardware profile / edge slowdown (for ablation E / logging)
    parser.add_argument("--profile_file", type=str, default="latency_profile.json")
    parser.add_argument("--edge_slowdown", type=float, default=20.0)
    
    args = parser.parse_args()

    # 1. Load Universal Profile
    profile_file = args.profile_file
    
    if not os.path.exists(profile_file):
        print(f"Error: {profile_file} not found. Run profile_env.py first or pass correct --profile_file.")
        sys.exit(1)

    print(f"Loading Universal Profile: {profile_file}")
    with open(profile_file, "r") as f:
        profiles = json.load(f)
        
    profiles['dataset_used'] = args.dataset

    # 2. Setup Directory
    run_dir = os.path.join("experiments", f"{args.exp_name}")
    os.makedirs(run_dir, exist_ok=True)
    
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
    }
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 3. Init Resources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_sim = NetworkSimulator(avg_bw_mbps=args.bw_mbps, avg_rtt_ms=args.rtt_ms)

    # Init Model
    # ALWAYS use input_channels=3 because we duplicate grayscale channels
    backbone = SplittableResNet18(num_classes=10, input_channels=3, pretrained=True)
    model = DeploymentAwareResNet(backbone, num_classes=10)
    
    config['model_instance'] = model

    # 4. Run Training
    print("--- Phase 1: Training ---")
    train_history = train_model(config, profiles, net_sim, device, run_dir)
    
    with open(os.path.join(run_dir, "train_log.json"), "w") as f:
        json.dump(train_history, f, indent=4)

    # 5. Run Evaluation
    print("\n--- Phase 2: Evaluation ---")
    test_results = evaluate_model(model, net_sim, profiles, device, dataset_name=args.dataset)
    
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    print(f"\nExperiment Complete. Results saved in {run_dir}")

if __name__ == "__main__":
    main()