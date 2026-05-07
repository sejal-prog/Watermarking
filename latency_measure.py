import torch
import time

def measure_latency(model, input_tensor, n_warmup=50, n_runs=200):
    """Measure average forward pass latency in milliseconds."""
    model.eval()
    
    # Warmup runs to stabilise GPU clock
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
    
    # Timed runs
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(input_tensor)
            torch.cuda.synchronize()
    
    end = time.perf_counter()
    avg_ms = (end - start) / n_runs * 1000
    return avg_ms

# Input: single 256x256 RGB frame, same as evaluation
device = torch.device("cuda")
dummy_input = torch.randn(1, 3, 256, 256).to(device)
print


# Load your models here
from videoseal.modules.convnext import ConvNeXtV2
from videoseal.modules.g_convnext import GConvNeXtExtractor

models = {
    "ConvNeXt-Tiny (Baseline)": ConvNeXtV2(depths=[3,3,9,3], dims=[96,192,384,768]).to(device),
    "G-ConvNeXt C4":            GConvNeXtExtractor(group_type="C4", nbits=32).to(device),
    "G-ConvNeXt C8":            GConvNeXtExtractor(group_type="C8", nbits=32).to(device),
}
for name, model in models.items():
    latency = measure_latency(model, dummy_input)
    print(f"{name}: {latency:.2f} ms")