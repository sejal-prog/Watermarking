"""
Test with Larger Extractor Capacity

The current extractor has dims=[32, 64, 128] = 128 final channels
This may not be enough to decode 32 independent bits.

This test uses dims=[64, 128, 256, 512] = 512 final channels
and a deeper FC head.

Run from videoseal root:
    python test_larger_extractor.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from videoseal.models.g_embedder import build_g_embedder
from videoseal.models.g_extractor import build_g_extractor


def test_larger_extractor():
    print("=" * 70)
    print(" Test with Larger Extractor")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    nbits = 32
    batch_size = 16
    img_size = 64
    
    embedder = build_g_embedder(
        nbits=nbits,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    ).to(device)
    
    # LARGER extractor
    extractor = build_g_extractor(
        nbits=nbits,
        depths=[2, 2, 4, 2],        # Deeper: 4 stages instead of 3
        dims=[64, 128, 256, 512],   # Wider: up to 512 channels
        group_type="C4"
    ).to(device)
    
    print(f"\n  Embedder params: {sum(p.numel() for p in embedder.parameters()):,}")
    print(f"  Extractor params: {sum(p.numel() for p in extractor.parameters()):,}")
    print(f"  Extractor dims: [64, 128, 256, 512]")
    
    # Freeze embedder
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    
    extractor.train()
    
    optimizer = Adam(extractor.parameters(), lr=1e-3)
    num_iterations = 3000
    scheduler = CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=1e-5)
    
    print(f"\n  Training for {num_iterations} iterations...")
    print("-" * 70)
    
    best_acc = 0
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
        msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
        
        with torch.no_grad():
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
        
        logits = extractor(imgs_w_norm)
        loss = F.binary_cross_entropy_with_logits(logits, msgs)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
        
        if accuracy > best_acc:
            best_acc = accuracy
        
        if (iteration + 1) % 300 == 0 or iteration == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Iter {iteration+1:5d} | Loss: {loss.item():.4f} | "
                  f"Bit Acc: {accuracy*100:.1f}% | Best: {best_acc*100:.1f}%")
    
    # Final evaluation
    print("\n  Final Evaluation:")
    extractor.eval()
    
    test_accs = []
    with torch.no_grad():
        for _ in range(20):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            logits = extractor(imgs_w_norm)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            test_accs.append(accuracy)
    
    final_avg = sum(test_accs) / len(test_accs)
    print(f"\n  Final Test Accuracy: {final_avg*100:.1f}%")
    print(f"  Best during training: {best_acc*100:.1f}%")
    
    # Per-bit
    print("\n  Per-Bit Accuracy:")
    bit_correct = torch.zeros(nbits)
    bit_total = torch.zeros(nbits)
    
    with torch.no_grad():
        for _ in range(100):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            logits = extractor(imgs_w_norm)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            correct = (predicted == msgs).float().cpu()
            bit_correct += correct.sum(dim=0)
            bit_total += batch_size
    
    bit_accuracy = bit_correct / bit_total
    print(f"  Mean: {bit_accuracy.mean()*100:.1f}%")
    print(f"  Min:  {bit_accuracy.min()*100:.1f}%")
    print(f"  Max:  {bit_accuracy.max()*100:.1f}%")
    
    return final_avg


def test_simple_cnn_extractor_32bits():
    """Test with simple CNN (non-equivariant) for comparison"""
    print("\n" + "=" * 70)
    print(" Comparison: Simple CNN Extractor (non-equivariant)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nbits = 32
    batch_size = 16
    img_size = 64
    
    embedder = build_g_embedder(
        nbits=nbits,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    ).to(device)
    
    # Simple CNN extractor (like in test_simple_extractor.py)
    class SimpleCNN(nn.Module):
        def __init__(self, nbits):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, 3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, nbits),
            )
        
        def forward(self, x):
            x = x * 2 - 1
            x = self.features(x)
            x = x.mean(dim=[-2, -1])
            return self.fc(x)
    
    extractor = SimpleCNN(nbits).to(device)
    
    print(f"\n  Embedder: G-CNN (equivariant)")
    print(f"  Extractor: Simple CNN (NOT equivariant)")
    print(f"  Extractor params: {sum(p.numel() for p in extractor.parameters()):,}")
    
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    
    extractor.train()
    optimizer = Adam(extractor.parameters(), lr=1e-3)
    
    num_iterations = 3000
    print(f"\n  Training for {num_iterations} iterations...")
    print("-" * 70)
    
    best_acc = 0
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
        msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
        
        with torch.no_grad():
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
        
        logits = extractor(imgs_w_norm)
        loss = F.binary_cross_entropy_with_logits(logits, msgs)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
        
        if accuracy > best_acc:
            best_acc = accuracy
        
        if (iteration + 1) % 300 == 0 or iteration == 0:
            print(f"  Iter {iteration+1:5d} | Loss: {loss.item():.4f} | "
                  f"Bit Acc: {accuracy*100:.1f}% | Best: {best_acc*100:.1f}%")
    
    # Final evaluation
    extractor.eval()
    test_accs = []
    with torch.no_grad():
        for _ in range(20):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            logits = extractor(imgs_w_norm)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            test_accs.append(accuracy)
    
    final_avg = sum(test_accs) / len(test_accs)
    print(f"\n  Final Test Accuracy: {final_avg*100:.1f}%")
    print(f"  Best during training: {best_acc*100:.1f}%")
    
    return final_avg


if __name__ == "__main__":
    # Test 1: Larger G-CNN extractor
    acc_gcnn = test_larger_extractor()
    
    # Test 2: Simple CNN for comparison
    acc_simple = test_simple_cnn_extractor_32bits()
    
    print("\n" + "=" * 70)
    print(" Comparison Summary")
    print("=" * 70)
    print(f"\n  G-CNN Extractor (larger): {acc_gcnn*100:.1f}%")
    print(f"  Simple CNN Extractor:     {acc_simple*100:.1f}%")
    
    if acc_simple > acc_gcnn + 0.1:
        print("\n  → Simple CNN significantly better. G-CNN architecture may need work.")
    elif acc_gcnn > 0.8:
        print("\n  → G-CNN works well with larger capacity!")
    else:
        print("\n  → Both struggle. May need different approach.")
    
    print("=" * 70)