"""
Test G-CNN Embedder with Simple CNN Extractor

This test verifies if the G-CNN embedder is encoding messages correctly
by using a simple (non-equivariant) CNN extractor.

If this works, the embedder is fine and the G-CNN extractor has issues.
If this doesn't work, the embedder has issues.

Run from videoseal root:
    python test_simple_extractor.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from videoseal.models.g_embedder import build_g_embedder


class SimpleCNNExtractor(nn.Module):
    """Simple CNN extractor for testing purposes"""
    
    def __init__(self, nbits=32):
        super().__init__()
        
        self.nbits = nbits
        
        # Simple CNN backbone
        self.features = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 64 -> 32
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 32 -> 16
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 16 -> 8
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 8 -> 4
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling + FC
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, nbits),
        )
    
    def forward(self, x):
        # x: [B, 3, H, W] in [0, 1]
        x = x * 2 - 1  # Normalize to [-1, 1]
        
        features = self.features(x)  # [B, 256, 4, 4]
        features = features.mean(dim=[-2, -1])  # Global average pool -> [B, 256]
        logits = self.fc(features)  # [B, nbits]
        
        return logits


def test_gcnn_embedder_with_simple_extractor():
    print("=" * 70)
    print(" Test G-CNN Embedder with Simple CNN Extractor")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    
    nbits = 32
    
    # G-CNN Embedder
    embedder = build_g_embedder(
        nbits=nbits,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    ).to(device)
    
    # Simple CNN Extractor (NOT equivariant)
    extractor = SimpleCNNExtractor(nbits=nbits).to(device)
    
    print(f"\n  Embedder: G-CNN (equivariant)")
    print(f"  Extractor: Simple CNN (NOT equivariant)")
    print(f"  Embedder params: {sum(p.numel() for p in embedder.parameters()):,}")
    print(f"  Extractor params: {sum(p.numel() for p in extractor.parameters()):,}")
    
    # Train both
    embedder.train()
    extractor.train()
    
    params = list(embedder.parameters()) + list(extractor.parameters())
    optimizer = Adam(params, lr=1e-3)
    
    # Training
    num_iterations = 500
    batch_size = 8
    img_size = 128
    
    print(f"\n  Training for {num_iterations} iterations...")
    print("-" * 70)
    
    history = []
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
        msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
        
        # Embed
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
        
        history.append(accuracy)
        
        if (iteration + 1) % 50 == 0 or iteration == 0:
            wm_str = embedder.gunet.watermark_strength.item()
            print(f"  Iter {iteration+1:4d} | Loss: {loss.item():.4f} | "
                  f"Bit Acc: {accuracy*100:.1f}% | WM Str: {wm_str:.3f}")
    
    print("-" * 70)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print(" Final Evaluation")
    print("=" * 70)
    
    embedder.eval()
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
    
    print(f"\n  Test Bit Accuracy: {avg_acc*100:.1f}%")
    print(f"  Individual: {[f'{a*100:.0f}%' for a in test_accs]}")
    
    # Test rotation (simple extractor is NOT rotation invariant)
    print("\n" + "=" * 70)
    print(" Rotation Test (Simple extractor is NOT rotation invariant)")
    print("=" * 70)
    
    with torch.no_grad():
        img = torch.rand(1, 3, img_size, img_size).to(device)
        msg = torch.randint(0, 2, (1, nbits)).float().to(device)
        
        img_w = embedder(img, msg)
        img_w_norm = (img_w + 1) / 2
        
        print(f"\n  Original message: {msg[0, :10].int().tolist()}...")
        
        for k in range(4):
            img_w_rotated = torch.rot90(img_w_norm, k, dims=[-2, -1])
            logits = extractor(img_w_rotated)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            acc = (predicted == msg).float().mean().item()
            print(f"  Rotation {k*90:3d}°: {predicted[0, :10].int().tolist()}... | Acc: {acc*100:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    
    if avg_acc > 0.8:
        print(f"\n  ✓ SUCCESS: G-CNN Embedder works with simple extractor!")
        print(f"    Final accuracy: {avg_acc*100:.1f}%")
        print(f"    This proves the embedder is encoding messages correctly.")
        print(f"    The issue is with the G-CNN extractor architecture.")
    elif avg_acc > 0.6:
        print(f"\n  ○ PARTIAL: Some learning but not optimal.")
        print(f"    May need more iterations or hyperparameter tuning.")
    else:
        print(f"\n  ✗ FAIL: Even simple extractor cannot learn.")
        print(f"    There may be an issue with the embedder architecture.")
    
    print("=" * 70)
    
    return history, avg_acc


if __name__ == "__main__":
    history, final_acc = test_gcnn_embedder_with_simple_extractor()