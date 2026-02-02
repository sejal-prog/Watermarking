"""
Comprehensive Training Test

This script tests multiple approaches to find what works:
1. Longer training with frozen embedder
2. Smaller image size (easier to learn)
3. Per-bit analysis to see which bits are learnable

Run from videoseal root:
    python test_comprehensive.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from videoseal.models.g_embedder import build_g_embedder
from videoseal.models.g_extractor import build_g_extractor


def test_long_training_frozen_embedder():
    """Train extractor for much longer with frozen embedder"""
    print("=" * 70)
    print(" Test 1: Long Training with Frozen Embedder")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nbits = 32
    batch_size = 16
    img_size = 64  # Smaller for faster learning
    
    embedder = build_g_embedder(
        nbits=nbits,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    ).to(device)
    
    extractor = build_g_extractor(
        nbits=nbits,
        depths=[2, 2, 2],
        dims=[32, 64, 128],
        group_type="C4"
    ).to(device)
    
    # Freeze embedder
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    
    extractor.train()
    optimizer = Adam(extractor.parameters(), lr=1e-3)
    
    num_iterations = 2000  # Much longer!
    
    print(f"\n  Image size: {img_size}x{img_size}")
    print(f"  Training for {num_iterations} iterations...")
    print("-" * 70)
    
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
        
        if (iteration + 1) % 200 == 0 or iteration == 0:
            with torch.no_grad():
                predicted = (torch.sigmoid(logits) > 0.5).float()
                accuracy = (predicted == msgs).float().mean().item()
            print(f"  Iter {iteration+1:5d} | Loss: {loss.item():.4f} | Bit Acc: {accuracy*100:.1f}%")
    
    # Evaluate
    extractor.eval()
    test_accs = []
    with torch.no_grad():
        for _ in range(10):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            logits = extractor(imgs_w_norm)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            test_accs.append(accuracy)
    
    avg_acc = sum(test_accs) / len(test_accs)
    print(f"\n  Final Test Accuracy: {avg_acc*100:.1f}%")
    
    return avg_acc


def test_per_bit_accuracy():
    """Analyze which bits are easier/harder to extract"""
    print("\n" + "=" * 70)
    print(" Test 2: Per-Bit Accuracy Analysis")
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
    
    extractor = build_g_extractor(
        nbits=nbits,
        depths=[2, 2, 2],
        dims=[32, 64, 128],
        group_type="C4"
    ).to(device)
    
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    
    extractor.train()
    optimizer = Adam(extractor.parameters(), lr=1e-3)
    
    # Train for a while
    for iteration in range(1000):
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
    
    # Evaluate per-bit accuracy
    extractor.eval()
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
    
    print(f"\n  Per-bit accuracy after 1000 iterations:")
    print(f"  Bits 0-7:   {[f'{a*100:.0f}%' for a in bit_accuracy[:8].tolist()]}")
    print(f"  Bits 8-15:  {[f'{a*100:.0f}%' for a in bit_accuracy[8:16].tolist()]}")
    print(f"  Bits 16-23: {[f'{a*100:.0f}%' for a in bit_accuracy[16:24].tolist()]}")
    print(f"  Bits 24-31: {[f'{a*100:.0f}%' for a in bit_accuracy[24:32].tolist()]}")
    
    print(f"\n  Mean: {bit_accuracy.mean()*100:.1f}%")
    print(f"  Best bit: {bit_accuracy.max()*100:.1f}% (bit {bit_accuracy.argmax().item()})")
    print(f"  Worst bit: {bit_accuracy.min()*100:.1f}% (bit {bit_accuracy.argmin().item()})")


def test_fewer_bits():
    """Test with fewer bits (easier problem)"""
    print("\n" + "=" * 70)
    print(" Test 3: Fewer Bits (8 bits instead of 32)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nbits = 8  # Much fewer bits!
    batch_size = 16
    img_size = 64
    
    embedder = build_g_embedder(
        nbits=nbits,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    ).to(device)
    
    extractor = build_g_extractor(
        nbits=nbits,
        depths=[2, 2, 2],
        dims=[32, 64, 128],
        group_type="C4"
    ).to(device)
    
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    
    extractor.train()
    optimizer = Adam(extractor.parameters(), lr=1e-3)
    
    num_iterations = 1000
    print(f"\n  Using only {nbits} bits")
    print(f"  Training for {num_iterations} iterations...")
    print("-" * 70)
    
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
        
        if (iteration + 1) % 200 == 0 or iteration == 0:
            with torch.no_grad():
                predicted = (torch.sigmoid(logits) > 0.5).float()
                accuracy = (predicted == msgs).float().mean().item()
            print(f"  Iter {iteration+1:5d} | Loss: {loss.item():.4f} | Bit Acc: {accuracy*100:.1f}%")
    
    # Evaluate
    extractor.eval()
    test_accs = []
    with torch.no_grad():
        for _ in range(10):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            logits = extractor(imgs_w_norm)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            test_accs.append(accuracy)
    
    avg_acc = sum(test_accs) / len(test_accs)
    print(f"\n  Final Test Accuracy (8 bits): {avg_acc*100:.1f}%")
    
    return avg_acc


def test_single_bit():
    """Test with just 1 bit (should definitely work)"""
    print("\n" + "=" * 70)
    print(" Test 4: Single Bit (simplest case)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nbits = 1  # Just 1 bit!
    batch_size = 16
    img_size = 64
    
    embedder = build_g_embedder(
        nbits=nbits,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    ).to(device)
    
    extractor = build_g_extractor(
        nbits=nbits,
        depths=[2, 2, 2],
        dims=[32, 64, 128],
        group_type="C4"
    ).to(device)
    
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    
    extractor.train()
    optimizer = Adam(extractor.parameters(), lr=1e-3)
    
    num_iterations = 500
    print(f"\n  Using only {nbits} bit")
    print(f"  Training for {num_iterations} iterations...")
    print("-" * 70)
    
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
        
        if (iteration + 1) % 100 == 0 or iteration == 0:
            with torch.no_grad():
                predicted = (torch.sigmoid(logits) > 0.5).float()
                accuracy = (predicted == msgs).float().mean().item()
            print(f"  Iter {iteration+1:4d} | Loss: {loss.item():.4f} | Bit Acc: {accuracy*100:.1f}%")
    
    # Evaluate
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
    
    avg_acc = sum(test_accs) / len(test_accs)
    print(f"\n  Final Test Accuracy (1 bit): {avg_acc*100:.1f}%")
    
    if avg_acc > 0.9:
        print("  ✓ Single bit works!")
    else:
        print("  ✗ Even single bit fails - fundamental issue!")
    
    return avg_acc


if __name__ == "__main__":
    print("=" * 70)
    print(" Comprehensive G-CNN Watermarking Tests")
    print("=" * 70)
    
    # Test 4 first - if single bit fails, we have fundamental issues
    acc_1bit = test_single_bit()
    
    # Test 3 - fewer bits
    acc_8bit = test_fewer_bits()
    
    # Test 2 - per-bit analysis
    test_per_bit_accuracy()
    
    # Test 1 - long training (optional, takes time)
    # acc_32bit = test_long_training_frozen_embedder()
    
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"\n  1 bit:  {acc_1bit*100:.1f}%")
    print(f"  8 bits: {acc_8bit*100:.1f}%")
    # print(f"  32 bits (long): {acc_32bit*100:.1f}%")
    
    if acc_1bit > 0.9:
        print("\n  → Single bit works. Problem scales with number of bits.")
        print("  → Try: fewer bits, longer training, or different architecture")
    else:
        print("\n  → Even single bit fails. Fundamental architecture issue.")