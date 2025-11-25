import torch
import time
import json
import numpy as np
from src.resnet_split import SplittableResNet18

def profile_hardware(filename="latency_profile.json"):
    # Since we are duplicating channels for MNIST/FMNIST, 
    # we ONLY profile for 3 channels (RGB) configuration.
    channels = 3
    input_shape = (1, channels, 32, 32)
    
    print(f"--- Profiling Hardware for 3-Channel Input (Universal) ---")
    
    # Init model with standard 3 channels
    model = SplittableResNet18(num_classes=10, input_channels=channels, pretrained=True)
    model.eval()
    
    dummy_input = torch.randn(input_shape)
    profiling_data = {}
    
    with torch.no_grad():
        # Warmup
        _ = model(dummy_input)
        
        x = dummy_input
        for i, block in enumerate(model.blocks):
            times = []
            for _ in range(50): # 50 runs for stability
                start = time.perf_counter()
                out = block(x)
                end = time.perf_counter()
                times.append(end - start)
            
            x = out
            median_time = np.median(times)
            output_size_bytes = model.get_block_output_size(i, input_shape)
            
            profiling_data[f"block_{i}"] = {
                "cloud_time_sec": median_time,
                "edge_time_sec": median_time * 5.0, # Simulate 5x slower edge
                "output_bytes": output_size_bytes
            }
            print(f"Block {i}: {median_time*1000:.3f}ms (Cloud) | Out: {output_size_bytes} B")

    with open(filename, "w") as f:
        json.dump(profiling_data, f, indent=4)
    print(f"Saved {filename}\n")

if __name__ == "__main__":
    profile_hardware()