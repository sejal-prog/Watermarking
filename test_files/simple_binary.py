"""
Simplified 2-Class Test - Binary message classification

This test simplifies the problem to verify the architecture can learn:
1. Use only 2 messages: all zeros and all ones
2. Train to classify between them
3. If this doesn't work, there's a fundamental issue

Run from videoseal root:
    python test_simple_binary.py
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


def simple_binary_test():
    print("=" * 70)
    print(" Simple Binary Classification Test")
    print("=" * 70)
    print("\n  This test uses only 2 messages (all zeros vs all ones)")
    print("  and trains the extractor to classify between them.")
    print("  If this doesn't work, there's a fundamental architecture issue.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    
    # Build models with small architecture
    nbits = 32
    
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
    
    # Freeze embedder - only train extractor
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    
    extractor.train()
    
    print(f"\n  Embedder: FROZEN (not training)")
    print(f"  Extractor: Training")
    
    # Optimizer - only extractor
    optimizer = Adam(extractor.parameters(), lr=1e-3)
    
    # Training
    num_iterations = 300
    batch_size = 8
    img_size = 64  # Smaller for speed
    
    # Fixed messages
    msg_zeros = torch.zeros(batch_size, nbits).to(device)
    msg_ones = torch.ones(batch_size, nbits).to(device)
    
    print(f"\n  Training for {num_iterations} iterations...")
    print("-" * 70)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Random images
        imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
        
        # Randomly choose message type
        use_ones = torch.rand(batch_size) > 0.5
        msgs = torch.where(use_ones.unsqueeze(1).to(device), msg_ones, msg_zeros)
        
        # Embed (no gradient needed)
        with torch.no_grad():
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
        
        # Extract
        logits = extractor(imgs_w_norm)
        
        # Loss
        loss = F.binary_cross_entropy_with_logits(logits, msgs)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
        
        if (iteration + 1) % 30 == 0 or iteration == 0:
            print(f"  Iter {iteration+1:4d} | Loss: {loss.item():.4f} | Bit Acc: {accuracy*100:.1f}%")
    
    print("-" * 70)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print(" Final Evaluation")
    print("=" * 70)
    
    extractor.eval()
    
    test_accs = []
    with torch.no_grad():
        for _ in range(10):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            use_ones = torch.rand(batch_size) > 0.5
            msgs = torch.where(use_ones.unsqueeze(1).to(device), msg_ones, msg_zeros)
            
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            logits = extractor(imgs_w_norm)
            
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            test_accs.append(accuracy)
    
    avg_acc = sum(test_accs) / len(test_accs)
    
    print(f"\n  Test Accuracy: {avg_acc*100:.1f}%")
    
    if avg_acc > 0.8:
        print(f"\n  ✓ SUCCESS: Extractor can learn from frozen embedder!")
        print(f"    The architecture works. Issue is likely in joint training dynamics.")
    elif avg_acc > 0.6:
        print(f"\n  ○ PARTIAL: Some learning, but not enough.")
        print(f"    May need architecture adjustments.")
    else:
        print(f"\n  ✗ FAIL: Extractor cannot learn from embedder output.")
        print(f"    There's a fundamental architecture issue.")
    
    # Additional test: Check if embedder actually produces different outputs
    print("\n" + "=" * 70)
    print(" Embedder Output Analysis")
    print("=" * 70)
    
    with torch.no_grad():
        img = torch.rand(1, 3, img_size, img_size).to(device)
        
        emb_zeros = embedder(img, msg_zeros[:1])
        emb_ones = embedder(img, msg_ones[:1])
        
        diff = (emb_zeros - emb_ones).abs()
        
        print(f"\n  Same image, different messages:")
        print(f"    Mean difference: {diff.mean().item():.4f}")
        print(f"    Max difference: {diff.max().item():.4f}")
        print(f"    Std difference: {diff.std().item():.4f}")
        
        if diff.mean().item() > 0.05:
            print(f"\n  ✓ Embedder produces different outputs for different messages")
        else:
            print(f"\n  ✗ Embedder outputs are too similar!")


def test_joint_training_with_frozen_extractor():
    """Train embedder with frozen extractor"""
    print("\n" + "=" * 70)
    print(" Reverse Test: Train Embedder with Frozen Extractor")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nbits = 32
    
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
    
    # Freeze extractor - only train embedder
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False
    
    embedder.train()
    
    print(f"\n  Embedder: Training")
    print(f"  Extractor: FROZEN (not training)")
    
    # Optimizer - only embedder
    optimizer = Adam(embedder.parameters(), lr=1e-3)
    
    # Training
    num_iterations = 300
    batch_size = 8
    img_size = 64
    
    msg_zeros = torch.zeros(batch_size, nbits).to(device)
    msg_ones = torch.ones(batch_size, nbits).to(device)
    
    print(f"\n  Training for {num_iterations} iterations...")
    print("-" * 70)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
        
        use_ones = torch.rand(batch_size) > 0.5
        msgs = torch.where(use_ones.unsqueeze(1).to(device), msg_ones, msg_zeros)
        
        # Embed
        imgs_w = embedder(imgs, msgs)
        imgs_w_norm = (imgs_w + 1) / 2
        
        # Extract (no gradient for extractor)
        with torch.no_grad():
            logits = extractor(imgs_w_norm)
        
        # We can't backprop through frozen extractor, so use a proxy loss
        # This won't work well, but let's see
        
        # Actually, we need gradients through extractor for embedder to learn
        # Let me change this test
        
        # For this test, let's maximize the difference in extractor output
        # when given different messages
        
        imgs_w_zeros = embedder(imgs, msg_zeros)
        imgs_w_ones = embedder(imgs, msg_ones)
        
        with torch.no_grad():
            logits_zeros = extractor((imgs_w_zeros + 1) / 2)
            logits_ones = extractor((imgs_w_ones + 1) / 2)
        
        # This won't actually train anything meaningful since gradients are blocked
        # Let's skip this test
        break
    
    print("  Skipped - need gradients through extractor to train embedder")


if __name__ == "__main__":
    simple_binary_test()