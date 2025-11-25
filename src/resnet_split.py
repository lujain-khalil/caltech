import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SplittableResNet18(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes (e.g., 10 for FashionMNIST).
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            pretrained (bool): If True, loads ImageNet weights.
        """
        super(SplittableResNet18, self).__init__()
        
        # 1. Load the base model with or without pretrained weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_model = resnet18(weights=weights)
        
        # 2. Handle Input Channel Mismatch (e.g., FashionMNIST is 1 channel, ResNet expects 3)
        if input_channels != 3:
            old_conv = base_model.conv1
            
            # Create a new conv layer with correct input channels
            # We keep the same kernel size (7x7), stride (2), etc.
            new_conv = nn.Conv2d(
                in_channels=input_channels, 
                out_channels=old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, 
                stride=old_conv.stride, 
                padding=old_conv.padding, 
                bias=old_conv.bias
            )
            
            if pretrained:
                # SMART INIT: Sum the weights across the RGB dimension to preserve 
                # learned patterns (edge detectors) instead of starting from scratch.
                # Shape is [Out, In, k, k]. Summing over dim 1 (Input channels).
                with torch.no_grad():
                    new_conv.weight[:] = torch.sum(old_conv.weight, dim=1, keepdim=True)
            
            base_model.conv1 = new_conv

        # 3. Handle Output Class Mismatch
        if base_model.fc.out_features != num_classes:
            base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)

        # 4. Create the Blocks (same split logic as before)
        self.block0 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        
        self.block1 = base_model.layer1
        self.block2 = base_model.layer2
        self.block3 = base_model.layer3
        self.block4 = base_model.layer4
        
        self.avgpool = base_model.avgpool
        self.fc = base_model.fc

        self.blocks = nn.ModuleList([
            self.block0, 
            self.block1, 
            self.block2, 
            self.block3, 
            self.block4
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_block_output_size(self, block_index, input_shape):
        """
        Helper to get the size of the tensor output by a specific block.
        Calculates dynamically based on the input_shape provided.
        """
        # Create a dummy input of the correct size/channels to trace dimensions
        dummy_input = torch.zeros(input_shape)
        
        with torch.no_grad():
            x = dummy_input
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i == block_index:
                    # Return size in Bytes (float32 = 4 bytes)
                    return x.numel() * 4 
        return 0