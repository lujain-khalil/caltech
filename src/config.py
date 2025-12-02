# # --- EXPERIMENT CONFIGURATIONS ---
# DEFAULT_ATTEMPT_NUM = 0

# DEFAULT_SLO = 20
# DEFAULT_RTT = 50
# DEFAULT_BW = 15
# DEFAULT_DATASET = "fmnist"
# DEFAULT_SLOWDOWN = 10.0

# # Default experiment parameters
# DEFAULT_EPOCHS = 5
# DEFAULT_BATCH_SIZE = 64
# DEFAULT_LR = 0.001
# DEFAULT_LAMBDA_LAT = 0.3
# DEFAULT_MU = 0.5

# # Ablations
# SLO_ABLATIONS = [80, 30, 5]  # in ms
# NET_ABLATIONS = {"fast": (10, 50), "slow": (150, 5), "impossible": (500, 1)}  # (rtt in ms, bw in Mbps)
# DATA_ABLATIONS = ["mnist", "cifar10"]
# EDGE_SLOWDOWN_ABLATIONS = [3.0, 30.0, 80.0]

# --- EXPERIMENT CONFIGURATIONS ---
DEFAULT_ATTEMPT_NUM = 0

DEFAULT_SLO = 20
DEFAULT_RTT = 50
DEFAULT_BW = 15
DEFAULT_DATASET = "fmnist"
DEFAULT_SLOWDOWN = 10.0

# Default experiment parameters
DEFAULT_EPOCHS = 30  # INCREASED: Need more epochs for 3-stage training
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 0.001

# CRITICAL FIX: Much lower latency penalties
DEFAULT_LAMBDA_LAT = 0.3  # Down from 2.0 - don't overwhelm accuracy
DEFAULT_MU = 0.5          # Down from 10.0 - SLO matters but not at all costs

# BETTER EXIT PLACEMENT: Move exits deeper in network
# Old: [1, 2, 3] - exits too early
# New: [2, 3, 4] - let features develop before exiting
DEFAULT_EXIT_POINTS = [1, 2, 3]  # After blocks 2, 3, 4

# Ablations
SLO_ABLATIONS = [80, 30, 5]  # in ms
NET_ABLATIONS = {"fast": (10, 50), "slow": (150, 5), "impossible": (500, 1)}  # (rtt in ms, bw in Mbps)
DATA_ABLATIONS = ["mnist", "cifar10"]
EDGE_SLOWDOWN_ABLATIONS = [3.0, 30.0, 80.0]

# NEW: Training stage configuration
STAGE1_RATIO = 0.25  # 25% for backbone training
STAGE2_RATIO = 0.25  # 25% for exit head training
STAGE3_RATIO = 0.50  # 50% for joint fine-tuning

# NEW: Exit training order
TRAIN_EXITS_DEEPEST_FIRST = True  # Branch-wise training

# NEW: Knowledge distillation weight for Stage 2
KD_ALPHA = 0.7  # Weight for CE loss vs KD loss