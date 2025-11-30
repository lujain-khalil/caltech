import torch
import torch.nn as nn
import torch.nn.functional as F

class DeploymentAwareResNet(nn.Module):
    def __init__(self, backbone, num_classes=10, exit_points=[1, 2, 3]):
        """
        backbone: The SplittableResNet18 instance
        exit_points: Indices of blocks after which we attach an early exit.
                     For ResNet18 (blocks 0-4), 1 and 3 are good standard spots.
        """
        super().__init__()
        self.backbone = backbone
        self.exit_points = set(exit_points)
        self.num_blocks = len(backbone.blocks)
        
        # 1. EARLY EXIT HEADS
        # We need to know the input size for the exit heads. 
        # For ResNet18: Block 1 -> 64 channels, Block 3 -> 256 channels.
        # We create a dictionary to hold these distinct heads.
        self.exit_heads = nn.ModuleDict()
        
        # Simple conv-based exit heads (Classifier + Confidence estimator)
        for idx in exit_points:
            in_channels = self._get_channels_for_block(idx)
            self.exit_heads[str(idx)] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_channels, num_classes)
            )
            
        # 2. SPLIT POINT LOGITS
        # We have K candidate split points (between blocks).
        # Logits z_1 ... z_K learn the preference for splitting at each point.
        self.split_candidates = [0, 1, 2, 3]
        self.split_logits = nn.Parameter(torch.zeros(len(self.split_candidates)))
        
        # 3. EXIT THRESHOLD SCALING (Learnable gamma from paper)
        # Gamma controls the sharpness of the sigmoid for exit probability
        self.exit_scale = nn.Parameter(torch.ones(len(exit_points)))
        self.exit_threshold = nn.Parameter(torch.empty(len(exit_points)).uniform_(0, 1))

    def _get_channels_for_block(self, idx):
        # Helper to hardcode channel sizes for standard ResNet18 blocks
        # Block 0: 64, Block 1: 64, Block 2: 128, Block 3: 256, Block 4: 512
        sizes = {0: 64, 1: 64, 2: 128, 3: 256, 4: 512}
        return sizes.get(idx, 512)

    def forward(self, x, temperature=1.0):
        """
        Returns:
            final_pred: Prediction from the very end of the network
            exit_preds: List of predictions from early exits
            split_probs: Softmax probabilities of splitting at each layer
            exit_probs: Probabilities of taking each early exit
        """
        
        # 1. Calculate Split Probabilities (Gumbel-Softmax) [cite: 133]
        # This makes the discrete choice "differentiable"
        split_probs = F.gumbel_softmax(self.split_logits, tau=temperature, hard=False, dim=0)
        
        exit_preds = {}
        exit_confidences = {}
        current_exit_idx = 0
        
        # 2. Forward Pass through Blocks
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            
            # If this block has an early exit attached
            if i in self.exit_points:
                # Run the early exit head
                out_exit = self.exit_heads[str(i)](x)
                exit_preds[i] = out_exit
                
                # Calculate Confidence (Max Logit) [cite: 138]
                # We use max of softmax as a proxy for confidence
                softmax_out = F.softmax(out_exit, dim=1)
                confidence, _ = torch.max(softmax_out, dim=1)
                
                # Calculate Exit Probability p_i [cite: 140]
                # p_i = sigmoid( (Confidence - Threshold) * Scale )
                gamma = self.exit_scale[current_exit_idx]
                tau = self.exit_threshold[current_exit_idx]
                
                p_exit = torch.sigmoid((confidence - tau) * gamma)
                exit_confidences[i] = p_exit
                current_exit_idx += 1
        
        # Final prediction (Cloud)
        x_final = self.backbone.avgpool(x)
        x_final = torch.flatten(x_final, 1)
        final_pred = self.backbone.fc(x_final)
        
        return final_pred, exit_preds, split_probs, exit_confidences