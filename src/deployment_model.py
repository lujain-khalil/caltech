import torch
import torch.nn as nn
import torch.nn.functional as F

class DeploymentAwareResNet(nn.Module):
    def __init__(self, backbone, num_classes=10, exit_points=[1, 2, 3]):
        """
        Key improvements:
        - Better exit head architecture with batch norm
        - Separate confidence estimation heads
        - More robust threshold learning
        """
        super().__init__()
        self.backbone = backbone
        self.exit_points = set(exit_points)
        self.num_blocks = len(backbone.blocks)
        
        # 1. IMPROVED EARLY EXIT HEADS
        self.exit_heads = nn.ModuleDict()
        self.confidence_heads = nn.ModuleDict()  # NEW: Separate confidence estimation
        
        for idx in exit_points:
            in_channels = self._get_channels_for_block(idx)
            
            # Classifier head (improved with batch norm)
            self.exit_heads[str(idx)] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_channels, in_channels // 2),
                nn.BatchNorm1d(in_channels // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(in_channels // 2, num_classes)
            )
            
        # 2. SPLIT POINT LOGITS (unchanged)
        self.split_candidates = [0, 1, 2, 3]
        self.split_logits = nn.Parameter(torch.zeros(len(self.split_candidates)))
        
        # 3. EXIT THRESHOLD & SCALE (tau, gamma)
        num_exits = len(exit_points)
        init_thresholds = torch.linspace(0.3, 0.7, steps=num_exits)
        self.raw_exit_threshold = nn.Parameter(init_thresholds)
        self.raw_exit_scale = nn.Parameter(torch.ones(num_exits))

    def _get_channels_for_block(self, idx):
        sizes = {0: 64, 1: 64, 2: 128, 3: 256, 4: 512}
        return sizes.get(idx, 512)
    
    @property
    def exit_scale(self):
        return F.softplus(self.raw_exit_scale) + 1.0  # Ensure minimum scale of 1
    
    @property
    def exit_threshold(self):
        return torch.sigmoid(self.raw_exit_threshold)

    def forward(self, x, temperature=1.0):
        """
        Returns:
            final_pred: Prediction from the very end
            exit_preds: Dict of predictions from early exits
            split_probs: Softmax probabilities of splitting at each layer
            exit_confidences: Dict of exit probabilities (per-sample)
        """
        
        # Split probabilities (Gumbel-Softmax)
        split_probs = F.gumbel_softmax(self.split_logits, tau=temperature, hard=False, dim=0)
        
        exit_preds = {}
        exit_confidences = {}
        
        sorted_exits = sorted(self.exit_points)
        exit_index_map = {b: i for i, b in enumerate(sorted_exits)}
        
        # Forward Pass through Blocks
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            
            if i in self.exit_points:
                # Run classifier head
                out_exit = self.exit_heads[str(i)](x)
                exit_preds[i] = out_exit
                
                # Confidence = max softmax probability
                softmax_out = F.softmax(out_exit, dim=1)
                confidence, _ = torch.max(softmax_out, dim=1)  # (B,)
                
                # Look up this head's tau/gamma
                ex_idx = exit_index_map[i]
                gamma = self.exit_scale[ex_idx]
                tau = self.exit_threshold[ex_idx]

                # Exit probability: p_exit = sigmoid((conf - tau) * gamma)
                p_exit = torch.sigmoid((confidence - tau) * gamma)
                exit_confidences[i] = p_exit
        
        # Final prediction (Cloud)
        x_final = self.backbone.avgpool(x)
        x_final = torch.flatten(x_final, 1)
        final_pred = self.backbone.fc(x_final)
        
        return final_pred, exit_preds, split_probs, exit_confidences