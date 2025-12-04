from pathlib import Path

# Root of the repo (one level up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------
EXPERIMENT_ROOTS = [
    PROJECT_ROOT / "experiments" / "attempt_12",
    PROJECT_ROOT / "experiments" / "baseline_models",
]

# --------------------------------------------------------------------
# Where to save ALL figures
# --------------------------------------------------------------------
FIG_ROOT = PROJECT_ROOT / "figures"

OVERVIEW_FIG_DIR = FIG_ROOT / "overview"
ABLATION_FIG_DIR = FIG_ROOT / "ablations"

for d in [FIG_ROOT, OVERVIEW_FIG_DIR, ABLATION_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Colour palettes 
# --------------------------------------------------------------------
FAMILY_COLOURS = {
    "default":  "#BFD7FF",  # light blue
    "slo":      "#D8CCFF",  # light lavender
    "net":      "#FFC9D9",  # light pink
    "edge":     "#C0DAFC",  # light mint
    "dataset":  "#FFE8B8",  # light yellow
    "baseline": "#D7D7D7",  # light grey
    "other":    "#AAAAAA",  # darker grey
}


EXIT_COLOURS = {
    "1": "#A2BFFF",   # blue
    "2": "#C3A6FF",   # lavender
    "3": "#FF9EBB",   # pink
    "final": "#C0C0C0",
}

# ablation colours
COLOR_LAT_AVG = "#8FAFFF"        # blue for avg latency
COLOR_LAT_P95 = "#B499FF"        # lavender for p95
COLOR_ACC_BAR =  "#C9F1D4"        # pink for accuracy bars
COLOR_ACC_BASELINE = "#C0C0C0"   # grey baseline
COLOR_ACC_OURS = "#8FAFFF"       # blue ours

# dataset overview bars
COLOR_DATASET_LAT = "#B499FF"    # lavender
COLOR_DATASET_ACC = "#8FAFFF"    # blue

# --------------------------------------------------------------------
# Baseline mapping: 
# --------------------------------------------------------------------
DATASET_BASELINES = {
    "fmnist": "fmnist",
    "mnist": "mnist",
    "cifar10": "cifar10",
    "cifar100": "cifar100"
}
