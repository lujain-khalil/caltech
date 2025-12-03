# --- EXPERIMENT CONFIGURATIONS ---
DEFAULT_ATTEMPT_NUM = 0

DEFAULT_SLO = 80
DEFAULT_RTT = 50
DEFAULT_BW = 15
DEFAULT_DATASET = "fmnist"
DEFAULT_SLOWDOWN = 20.0

# Default experiment parameters
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 0.001

# CRITICAL FIX: Much lower latency penalties
DEFAULT_LAMBDA_LAT = 0.3  
DEFAULT_MU = 0.5        

# Move exits deeper in network
DEFAULT_EXIT_POINTS = [1, 2, 3]  # After blocks 1, 2, 3

# Ablations
SLO_ABLATIONS = [200, 100, 60]  # in ms
NET_ABLATIONS = {"fast": (10, 50), "slow": (150, 5), "impossible": (500, 1)}  # (rtt in ms, bw in Mbps)
DATA_ABLATIONS = ["mnist", "cifar10", "cifar100"]
EDGE_SLOWDOWN_ABLATIONS = [5.0, 40.0, 100.0]

# Training stage configuration
STAGE1_RATIO = 0.25  # 25% for backbone training
STAGE2_RATIO = 0.25  # 25% for exit head training
STAGE3_RATIO = 0.50  # 50% for joint fine-tuning

# Exit training order
TRAIN_EXITS_DEEPEST_FIRST = True  # Branch-wise training

# Knowledge distillation weight for Stage 2
KD_ALPHA = 0.7  # Weight for CE loss vs KD loss