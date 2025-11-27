import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def format_table(data, headers):
    """
    Format data as a simple ASCII table.
    
    Args:
        data: List of dictionaries with data for each row
        headers: List of column headers
        
    Returns:
        Formatted table as a string
    """
    # Calculate column widths
    col_widths = {}
    for header in headers:
        col_widths[header] = len(str(header))
        for row in data:
            col_widths[header] = max(col_widths[header], len(str(row[header])))
    
    # Build table
    lines = []
    
    # Header row
    header_row = " | ".join(f"{h:{col_widths[h]}}" for h in headers)
    lines.append(header_row)
    lines.append("-" * len(header_row))
    
    # Data rows
    for row in data:
        row_line = " | ".join(f"{str(row[h]):{col_widths[h]}}" for h in headers)
        lines.append(row_line)
    
    return "\n".join(lines)


def summarize_attempt(attempt_dir):
    """
    Summarize all experiments in the given attempt directory.
    
    Args:
        attempt_dir: Path to the attempt directory (e.g., experiments/attempt_1)
    """
    attempt_path = Path(attempt_dir)
    
    # Collect data from all experiment subdirectories
    experiments_data = []
    raw_data = {}  # Store raw numeric data for plotting
    
    # Get all subdirectories in the attempt folder
    for exp_dir in sorted(attempt_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        
        # Construct paths to the required files
        config_file = exp_dir / "config.json"
        train_log_file = exp_dir / "train_log.json"
        test_metrics_file = exp_dir / "test_metrics.json"
        
        # Check if all required files exist
        if not (config_file.exists() and train_log_file.exists() and test_metrics_file.exists()):
            print(f"Warning: Skipping {exp_name} - missing required files")
            continue
        
        try:
            # Load config.json
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Load train_log.json
            with open(train_log_file, 'r') as f:
                train_log = json.load(f)
            
            # Load test_metrics.json
            with open(test_metrics_file, 'r') as f:
                test_metrics = json.load(f)
            
            # Extract data from config.json
            dataset = config.get("dataset", "N/A")
            slo_target = config.get("slo_target", "N/A")
            avg_rtt = config.get("avg_rtt", "N/A")
            avg_bw = config.get("avg_bw", "N/A")
            
            # Convert slo_target to ms
            slo_target_ms = slo_target * 1000 if isinstance(slo_target, (int, float)) else "N/A"
            
            # Extract data from train_log.json (last epoch)
            last_epoch = train_log[-1] if train_log else {}
            train_accuracy = last_epoch.get("accuracy", "N/A")
            train_latency_ms = last_epoch.get("latency_ms", "N/A")
            exit_probs = last_epoch.get("exit_probs", [])
            exit_prob_0 = exit_probs[0] if len(exit_probs) > 0 else "N/A"
            exit_prob_1 = exit_probs[1] if len(exit_probs) > 1 else "N/A"
            loss = last_epoch.get("loss", "N/A")
            
            # Extract data from test_metrics.json
            test_accuracy = test_metrics.get("test_accuracy", "N/A")
            avg_latency = test_metrics.get("avg_latency_ms", "N/A")
            p95_latency = test_metrics.get("p95_latency_ms", "N/A")
            split_point = test_metrics.get("split_point", "N/A")
            
            # Add to results
            experiments_data.append({
                "Experiment": exp_name,
                "Dataset": dataset,
                "SLO Target (ms)": f"{slo_target_ms:.1f}" if isinstance(slo_target_ms, float) else slo_target_ms,
                "Avg RTT": avg_rtt,
                "Avg BW": avg_bw,
                "Train Accuracy": f"{train_accuracy:.2f}" if isinstance(train_accuracy, (int, float)) else train_accuracy,
                "Train Latency (ms)": f"{train_latency_ms:.2f}" if isinstance(train_latency_ms, (int, float)) else train_latency_ms,
                "Exit Prob [0]": f"{exit_prob_0:.6f}" if isinstance(exit_prob_0, (int, float)) else exit_prob_0,
                "Exit Prob [1]": f"{exit_prob_1:.6f}" if isinstance(exit_prob_1, (int, float)) else exit_prob_1,
                "Loss": f"{loss:.6f}" if isinstance(loss, (int, float)) else loss,
                "Test Accuracy": f"{test_accuracy:.2f}" if isinstance(test_accuracy, (int, float)) else test_accuracy,
                "Avg Latency (ms)": f"{avg_latency:.2f}" if isinstance(avg_latency, (int, float)) else avg_latency,
                "P95 Latency (ms)": f"{p95_latency:.2f}" if isinstance(p95_latency, (int, float)) else p95_latency,
                "Split Point": split_point,
            })
            
            # Store raw numeric data for plotting
            raw_data[exp_name] = {
                "dataset": dataset,
                "slo_target": float(slo_target) if isinstance(slo_target, (int, float)) else None,
                "avg_rtt": float(avg_rtt) if isinstance(avg_rtt, (int, float)) else None,
                "avg_bw": float(avg_bw) if isinstance(avg_bw, (int, float)) else None,
                "train_accuracy": float(train_accuracy) if isinstance(train_accuracy, (int, float)) else None,
                "train_latency_ms": float(train_latency_ms) if isinstance(train_latency_ms, (int, float)) else None,
                "exit_prob_0": float(exit_prob_0) if isinstance(exit_prob_0, (int, float)) else None,
                "exit_prob_1": float(exit_prob_1) if isinstance(exit_prob_1, (int, float)) else None,
                "loss": float(loss) if isinstance(loss, (int, float)) else None,
                "test_accuracy": float(test_accuracy) if isinstance(test_accuracy, (int, float)) else None,
                "avg_latency": float(avg_latency) if isinstance(avg_latency, (int, float)) else None,
                "p95_latency": float(p95_latency) if isinstance(p95_latency, (int, float)) else None,
                "split_point": int(split_point) if isinstance(split_point, (int, float)) else None,
                "train_log": train_log,
            }
        
        except Exception as e:
            print(f"Error processing {exp_name}: {e}")
            continue
    
    # Generate table using manual formatting
    if experiments_data:
        # Get headers from the first entry
        headers = list(experiments_data[0].keys())
        
        # Format table
        table = format_table(experiments_data, headers)
        
        # Write to summary file
        summary_file = attempt_path / "summary_1.txt"
        with open(summary_file, 'w') as f:
            f.write(table)
        
        print(f"Summary saved to {summary_file}")
        print("\nSummary Table:")
        print(table)
        
        # Generate figures
        generate_figures(attempt_path, raw_data)
    else:
        print("No valid experiments found to summarize")


def get_ablation_set(exp_name):
    """Extract ablation set letter from experiment name."""
    if exp_name.startswith('A_'):
        return 'A', exp_name
    elif exp_name.startswith('B_'):
        return 'B', exp_name
    elif exp_name.startswith('C_'):
        return 'C', exp_name
    elif exp_name == 'default':
        return None, exp_name
    return None, exp_name


def generate_figures(attempt_path, raw_data):
    """
    Generate comparison figures for each ablation set and across all experiments.
    
    Args:
        attempt_path: Path to the attempt directory
        raw_data: Dictionary of experiment data
    """
    # Create figures directory
    figures_dir = attempt_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    print(f"\nGenerating figures in {figures_dir}...")
    
    # Rename default experiment based on context and organize by ablation set
    renamed_data = {}
    for exp_name, data in raw_data.items():
        if exp_name == 'default':
            # default will be added to each ablation set based on its dataset
            continue
        renamed_data[exp_name] = data
    
    # Create ablation sets with renamed default experiments
    ablation_sets = {}
    
    # Add A set experiments
    ablation_sets['A'] = {}
    for exp_name, data in raw_data.items():
        if exp_name.startswith('A_'):
            ablation_sets['A'][exp_name] = data
    # Add default as A_slo_100
    if 'default' in raw_data:
        ablation_sets['A']['A_slo_100'] = raw_data['default']
    
    # Add B set experiments
    ablation_sets['B'] = {}
    for exp_name, data in raw_data.items():
        if exp_name.startswith('B_'):
            ablation_sets['B'][exp_name] = data
    # Add default as B_net_moderate
    if 'default' in raw_data:
        ablation_sets['B']['B_net_moderate'] = raw_data['default']
    
    # Add C set experiments
    ablation_sets['C'] = {}
    for exp_name, data in raw_data.items():
        if exp_name.startswith('C_'):
            ablation_sets['C'][exp_name] = data
    # Add default as C_data_fmnist
    if 'default' in raw_data:
        ablation_sets['C']['C_data_fmnist'] = raw_data['default']
    
    # Generate figures for each ablation set
    for ablation in ['A', 'B', 'C']:
        if ablation in ablation_sets and ablation_sets[ablation]:
            exps = ablation_sets[ablation]
            
            # Figure 1: Accuracy vs Latency Trade-off
            plot_accuracy_latency_tradeoff(figures_dir, ablation, exps)
            
            # Figure 2: Test Accuracy Comparison
            plot_test_accuracy_comparison(figures_dir, ablation, exps)
            
            # Figure 3: Latency Metrics Comparison (Avg and P95)
            plot_latency_metrics_comparison(figures_dir, ablation, exps)
            
            # Figure 4: Training Metrics Evolution (loss and accuracy across epochs)
            plot_training_evolution(figures_dir, ablation, exps)
    
    # Generate comparative figures across all ablation sets
    all_experiments = {}
    all_experiments.update(ablation_sets.get('A', {}))
    all_experiments.update(ablation_sets.get('B', {}))
    all_experiments.update(ablation_sets.get('C', {}))
    
    # Figure 5: All experiments - Accuracy vs Latency
    plot_all_accuracy_latency_comparison(figures_dir, all_experiments)
    
    # Figure 6: All experiments - Test Accuracy by Set
    plot_all_test_accuracy_by_set(figures_dir, ablation_sets)
    
    # Figure 7: All experiments - Latency Comparison
    plot_all_latency_comparison(figures_dir, ablation_sets)
    
    print("Figure generation complete!")


def plot_accuracy_latency_tradeoff(figures_dir, ablation, experiments):
    """Plot accuracy vs latency trade-off for an ablation set."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exp_names = []
    test_accs = []
    avg_lats = []
    colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))
    
    for i, (exp_name, data) in enumerate(sorted(experiments.items())):
        if data['test_accuracy'] is not None and data['avg_latency'] is not None:
            exp_names.append(exp_name)
            test_accs.append(data['test_accuracy'])
            avg_lats.append(data['avg_latency'])
    
    scatter = ax.scatter(avg_lats, test_accs, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels to points
    for i, exp_name in enumerate(exp_names):
        ax.annotate(exp_name, (avg_lats[i], test_accs[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Ablation Set {ablation}: Accuracy vs Latency Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'set_{ablation}_01_accuracy_vs_latency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: set_{ablation}_01_accuracy_vs_latency.png")


def plot_test_accuracy_comparison(figures_dir, ablation, experiments):
    """Plot test accuracy comparison for an ablation set."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exp_names = []
    test_accs = []
    
    for exp_name, data in sorted(experiments.items()):
        if data['test_accuracy'] is not None:
            exp_names.append(exp_name)
            test_accs.append(data['test_accuracy'])
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
    bars = ax.bar(range(len(exp_names)), test_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, test_accs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Ablation Set {ablation}: Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'set_{ablation}_02_test_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: set_{ablation}_02_test_accuracy_comparison.png")


def plot_latency_metrics_comparison(figures_dir, ablation, experiments):
    """Plot average and P95 latency comparison for an ablation set."""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    exp_names = []
    avg_lats = []
    p95_lats = []
    
    for exp_name, data in sorted(experiments.items()):
        if data['avg_latency'] is not None and data['p95_latency'] is not None:
            exp_names.append(exp_name)
            avg_lats.append(data['avg_latency'])
            p95_lats.append(data['p95_latency'])
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, avg_lats, width, label='Average Latency', 
                   color='skyblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, p95_lats, width, label='P95 Latency',
                   color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Experiments', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'Ablation Set {ablation}: Latency Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'set_{ablation}_03_latency_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: set_{ablation}_03_latency_metrics.png")


def plot_training_evolution(figures_dir, ablation, experiments):
    """Plot training loss and accuracy evolution across epochs for an ablation set."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for idx, (exp_name, data) in enumerate(sorted(experiments.items())):
        train_log = data['train_log']
        if train_log:
            epochs = [entry['epoch'] for entry in train_log]
            losses = [entry['loss'] for entry in train_log]
            accuracies = [entry['accuracy'] for entry in train_log]
            
            ax1.plot(epochs, losses, marker='o', linewidth=2, markersize=4, 
                    label=exp_name, color=colors[idx], alpha=0.8)
            ax2.plot(epochs, accuracies, marker='s', linewidth=2, markersize=4,
                    label=exp_name, color=colors[idx], alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title(f'Ablation Set {ablation}: Training Loss Evolution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Ablation Set {ablation}: Training Accuracy Evolution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'set_{ablation}_04_training_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: set_{ablation}_04_training_evolution.png")


def plot_all_accuracy_latency_comparison(figures_dir, all_experiments):
    """Plot accuracy vs latency for all experiments colored by ablation set."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define colors for each ablation set
    set_colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
    
    for exp_name, data in sorted(all_experiments.items()):
        if data['test_accuracy'] is not None and data['avg_latency'] is not None:
            # Determine ablation set from experiment name
            if exp_name.startswith('A_') or exp_name == 'A_slo_100':
                color = set_colors['A']
                set_label = 'Set A (SLO Target)'
            elif exp_name.startswith('B_') or exp_name == 'B_net_moderate':
                color = set_colors['B']
                set_label = 'Set B (Network)'
            elif exp_name.startswith('C_') or exp_name == 'C_data_fmnist':
                color = set_colors['C']
                set_label = 'Set C (Dataset)'
            else:
                color = '#808080'
                set_label = 'Other'
            
            # Map set label to marker
            markers = {'Set A (SLO Target)': 'o', 'Set B (Network)': 's', 'Set C (Dataset)': '^'}
            marker = markers.get(set_label, 'o')
            
            ax.scatter(data['avg_latency'], data['test_accuracy'], s=250, c=color, 
                      marker=marker, alpha=0.7, edgecolors='black', linewidth=2, label=set_label)
            ax.annotate(exp_name, (data['avg_latency'], data['test_accuracy']),
                       xytext=(7, 7), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('All Experiments: Accuracy vs Latency Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='Set A (SLO Target)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Set B (Network)'),
        Patch(facecolor='#2ca02c', edgecolor='black', label='Set C (Dataset)')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'all_01_accuracy_vs_latency_all_sets.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_01_accuracy_vs_latency_all_sets.png")


def plot_all_test_accuracy_by_set(figures_dir, ablation_sets):
    """Plot test accuracy for all experiments grouped by ablation set."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    set_colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
    x_pos = 0
    tick_positions = []
    tick_labels = []
    
    for set_letter in ['A', 'B', 'C']:
        if set_letter in ablation_sets and ablation_sets[set_letter]:
            exps = sorted(ablation_sets[set_letter].items())
            for exp_name, data in exps:
                if data['test_accuracy'] is not None:
                    ax.bar(x_pos, data['test_accuracy'], color=set_colors[set_letter], 
                          alpha=0.8, edgecolor='black', linewidth=1.5, width=0.8)
                    ax.text(x_pos, data['test_accuracy'] + 0.5, f"{data['test_accuracy']:.2f}%",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
                    tick_positions.append(x_pos)
                    tick_labels.append(exp_name)
                    x_pos += 1
            x_pos += 0.5  # Add spacing between sets
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('All Experiments: Test Accuracy Comparison by Ablation Set', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add set separators
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='Set A (SLO Target)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Set B (Network)'),
        Patch(facecolor='#2ca02c', edgecolor='black', label='Set C (Dataset)')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'all_02_test_accuracy_all_experiments.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_02_test_accuracy_all_experiments.png")


def plot_all_latency_comparison(figures_dir, ablation_sets):
    """Plot average and P95 latency for all experiments grouped by ablation set."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    set_colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
    x_pos = 0
    tick_positions = []
    tick_labels = []
    width = 0.35
    
    all_avg_lats = []
    all_p95_lats = []
    all_x_pos_avg = []
    all_x_pos_p95 = []
    
    for set_letter in ['A', 'B', 'C']:
        if set_letter in ablation_sets and ablation_sets[set_letter]:
            exps = sorted(ablation_sets[set_letter].items())
            for exp_name, data in exps:
                if data['avg_latency'] is not None and data['p95_latency'] is not None:
                    all_avg_lats.append(data['avg_latency'])
                    all_p95_lats.append(data['p95_latency'])
                    all_x_pos_avg.append(x_pos - width/2)
                    all_x_pos_p95.append(x_pos + width/2)
                    
                    tick_positions.append(x_pos)
                    tick_labels.append(exp_name)
                    x_pos += 1
            x_pos += 0.5  # Add spacing between sets
    
    # Create bars
    bars1 = ax.bar(all_x_pos_avg, all_avg_lats, width, label='Average Latency',
                   color='skyblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(all_x_pos_p95, all_p95_lats, width, label='P95 Latency',
                   color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('All Experiments: Latency Metrics Comparison by Ablation Set', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'all_03_latency_all_experiments.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_03_latency_all_experiments.png")




if __name__ == "__main__":
    # Path to attempt_1 directory
    attempt_1_path = Path(__file__).parent / "experiments" / "attempt_1"
    summarize_attempt(attempt_1_path)
