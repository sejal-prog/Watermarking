"""
Diagnostic script to understand why extractor outputs are too similar
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from videoseal.models.g_extractor import build_g_extractor
from escnn import nn as enn


def diagnose_extractor():
    print("=" * 70)
    print(" Extractor Diagnostic")
    print("=" * 70)
    
    extractor = build_g_extractor(
        nbits=32,
        depths=[2, 2, 2],
        dims=[32, 64, 128],
        group_type="C4"
    )
    extractor.eval()
    
    g_convnext = extractor.g_convnext
    
    # Two very different images
    img1 = torch.zeros(1, 3, 128, 128)  # All black
    img2 = torch.ones(1, 3, 128, 128)   # All white
    img3 = torch.rand(1, 3, 128, 128)   # Random
    
    print("\n[Input Images]")
    print(f"  img1 (black): mean={img1.mean():.4f}")
    print(f"  img2 (white): mean={img2.mean():.4f}")
    print(f"  img3 (random): mean={img3.mean():.4f}")
    
    with torch.no_grad():
        # Preprocess
        x1 = extractor.preprocess(img1)
        x2 = extractor.preprocess(img2)
        x3 = extractor.preprocess(img3)
        
        print(f"\n[After Preprocess]")
        print(f"  x1: mean={x1.mean():.4f}, std={x1.std():.4f}")
        print(f"  x2: mean={x2.mean():.4f}, std={x2.std():.4f}")
        print(f"  x3: mean={x3.mean():.4f}, std={x3.std():.4f}")
        print(f"  diff(x1,x2): {(x1-x2).abs().mean():.4f}")
        
        # Lift
        x1 = enn.GeometricTensor(x1, g_convnext.in_type)
        x2 = enn.GeometricTensor(x2, g_convnext.in_type)
        x3 = enn.GeometricTensor(x3, g_convnext.in_type)
        
        x1 = g_convnext.stem(x1)
        x2 = g_convnext.stem(x2)
        x3 = g_convnext.stem(x3)
        
        print(f"\n[After Stem]")
        print(f"  x1: mean={x1.tensor.mean():.4f}, std={x1.tensor.std():.4f}, shape={x1.tensor.shape}")
        print(f"  x2: mean={x2.tensor.mean():.4f}, std={x2.tensor.std():.4f}")
        print(f"  x3: mean={x3.tensor.mean():.4f}, std={x3.tensor.std():.4f}")
        print(f"  diff(x1,x2): {(x1.tensor-x2.tensor).abs().mean():.4f}")
        
        # Check for dead ReLU
        print(f"  x1 zeros ratio: {(x1.tensor == 0).float().mean():.4f}")
        print(f"  x2 zeros ratio: {(x2.tensor == 0).float().mean():.4f}")
        
        # Stages
        for i, stage in enumerate(g_convnext.stages):
            x1 = stage(x1)
            x2 = stage(x2)
            x3 = stage(x3)
            
            print(f"\n[After Stage {i}]")
            print(f"  x1: mean={x1.tensor.mean():.4f}, std={x1.tensor.std():.4f}, shape={x1.tensor.shape}")
            print(f"  x2: mean={x2.tensor.mean():.4f}, std={x2.tensor.std():.4f}")
            print(f"  x3: mean={x3.tensor.mean():.4f}, std={x3.tensor.std():.4f}")
            print(f"  diff(x1,x2): {(x1.tensor-x2.tensor).abs().mean():.4f}")
            print(f"  diff(x1,x3): {(x1.tensor-x3.tensor).abs().mean():.4f}")
            print(f"  x1 zeros ratio: {(x1.tensor == 0).float().mean():.4f}")
        
        # Group pooling
        x1 = g_convnext.group_pool(x1)
        x2 = g_convnext.group_pool(x2)
        x3 = g_convnext.group_pool(x3)
        
        print(f"\n[After Group Pool]")
        print(f"  x1: mean={x1.tensor.mean():.4f}, std={x1.tensor.std():.4f}, shape={x1.tensor.shape}")
        print(f"  x2: mean={x2.tensor.mean():.4f}, std={x2.tensor.std():.4f}")
        print(f"  x3: mean={x3.tensor.mean():.4f}, std={x3.tensor.std():.4f}")
        print(f"  diff(x1,x2): {(x1.tensor-x2.tensor).abs().mean():.4f}")
        
        # Spatial pooling
        f1 = x1.tensor.mean(dim=[-2, -1])
        f2 = x2.tensor.mean(dim=[-2, -1])
        f3 = x3.tensor.mean(dim=[-2, -1])
        
        print(f"\n[After Spatial Pool]")
        print(f"  f1: mean={f1.mean():.4f}, std={f1.std():.4f}, shape={f1.shape}")
        print(f"  f2: mean={f2.mean():.4f}, std={f2.std():.4f}")
        print(f"  f3: mean={f3.mean():.4f}, std={f3.std():.4f}")
        print(f"  diff(f1,f2): {(f1-f2).abs().mean():.4f}")
        print(f"  diff(f1,f3): {(f1-f3).abs().mean():.4f}")
        
        # FC head
        logits1 = g_convnext.fc(f1)
        logits2 = g_convnext.fc(f2)
        logits3 = g_convnext.fc(f3)
        
        print(f"\n[After FC (Logits)]")
        print(f"  logits1: mean={logits1.mean():.4f}, std={logits1.std():.4f}")
        print(f"  logits2: mean={logits2.mean():.4f}, std={logits2.std():.4f}")
        print(f"  logits3: mean={logits3.mean():.4f}, std={logits3.std():.4f}")
        print(f"  diff(logits1,logits2): {(logits1-logits2).abs().mean():.4f}")
        print(f"  diff(logits1,logits3): {(logits1-logits3).abs().mean():.4f}")
        
        print(f"\n[Decoded Bits]")
        bits1 = (torch.sigmoid(logits1) > 0.5).float()
        bits2 = (torch.sigmoid(logits2) > 0.5).float()
        bits3 = (torch.sigmoid(logits3) > 0.5).float()
        print(f"  bits1: {bits1[0, :10].tolist()}")
        print(f"  bits2: {bits2[0, :10].tolist()}")
        print(f"  bits3: {bits3[0, :10].tolist()}")
        print(f"  bits1 vs bits2: {(bits1 == bits2).float().mean():.4f} same")
        print(f"  bits1 vs bits3: {(bits1 == bits3).float().mean():.4f} same")


if __name__ == "__main__":
    diagnose_extractor()