"""
Staged Training for G-CNN Watermarking

This approach works because:
- Stage 1: Freeze embedder, train extractor to recognize the initial watermark
- Stage 2: Fine-tune both with lower learning rate

Run from videoseal root:
    python test_staged_training.py
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


def staged_training():
    print("=" * 70)
    print(" Staged Training - G-CNN Watermarking")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    nbits = 32
    batch_size = 8
    img_size = 128
    
    # Build models
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
    
    print(f"\n  Embedder params: {sum(p.numel() for p in embedder.parameters()):,}")
    print(f"  Extractor params: {sum(p.numel() for p in extractor.parameters()):,}")
    
    # =========================================================================
    # STAGE 1: Freeze Embedder, Train Extractor
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STAGE 1: Freeze Embedder, Train Extractor")
    print("=" * 70)
    
    # Freeze embedder
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    
    extractor.train()
    
    optimizer_ext = Adam(extractor.parameters(), lr=1e-3)
    
    stage1_iters = 200
    print(f"\n  Training extractor for {stage1_iters} iterations (embedder frozen)...")
    print("-" * 70)
    
    for iteration in range(stage1_iters):
        optimizer_ext.zero_grad()
        
        imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
        msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
        
        # Embed (no gradient)
        with torch.no_grad():
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
        
        # Extract
        logits = extractor(imgs_w_norm)
        
        # Loss
        loss = F.binary_cross_entropy_with_logits(logits, msgs)
        
        # Backward
        loss.backward()
        optimizer_ext.step()
        
        # Accuracy
        with torch.no_grad():
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
        
        if (iteration + 1) % 40 == 0 or iteration == 0:
            print(f"  Iter {iteration+1:4d} | Loss: {loss.item():.4f} | Bit Acc: {accuracy*100:.1f}%")
    
    # Evaluate Stage 1
    print("\n  Stage 1 Evaluation:")
    extractor.eval()
    
    stage1_accs = []
    with torch.no_grad():
        for _ in range(10):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            logits = extractor(imgs_w_norm)
            
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            stage1_accs.append(accuracy)
    
    stage1_avg = sum(stage1_accs) / len(stage1_accs)
    print(f"  Stage 1 Test Accuracy: {stage1_avg*100:.1f}%")
    
    # =========================================================================
    # STAGE 2: Fine-tune Both (lower learning rate)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STAGE 2: Fine-tune Both (lower LR)")
    print("=" * 70)
    
    # Unfreeze embedder
    embedder.train()
    for param in embedder.parameters():
        param.requires_grad = True
    
    extractor.train()
    
    # Lower learning rate for joint training
    optimizer_both = Adam([
        {'params': embedder.parameters(), 'lr': 1e-4},  # Lower LR for embedder
        {'params': extractor.parameters(), 'lr': 5e-4},  # Higher LR for extractor
    ])
    
    stage2_iters = 300
    print(f"\n  Fine-tuning both for {stage2_iters} iterations...")
    print(f"  Embedder LR: 1e-4, Extractor LR: 5e-4")
    print("-" * 70)
    
    for iteration in range(stage2_iters):
        optimizer_both.zero_grad()
        
        imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
        msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
        
        # Embed
        imgs_w = embedder(imgs, msgs)
        imgs_w_norm = (imgs_w + 1) / 2
        
        # Extract
        logits = extractor(imgs_w_norm)
        
        # Loss: message loss + small image loss for regularization
        msg_loss = F.binary_cross_entropy_with_logits(logits, msgs)
        img_loss = F.mse_loss(imgs_w, imgs * 2 - 1)
        
        loss = msg_loss + 0.01 * img_loss
        
        # Backward
        loss.backward()
        optimizer_both.step()
        
        # Accuracy
        with torch.no_grad():
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
        
        if (iteration + 1) % 50 == 0 or iteration == 0:
            print(f"  Iter {iteration+1:4d} | Loss: {loss.item():.4f} | "
                  f"Msg: {msg_loss.item():.4f} | Bit Acc: {accuracy*100:.1f}%")
    
    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Final Evaluation")
    print("=" * 70)
    
    embedder.eval()
    extractor.eval()
    
    final_accs = []
    with torch.no_grad():
        for _ in range(10):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            logits = extractor(imgs_w_norm)
            
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            final_accs.append(accuracy)
    
    final_avg = sum(final_accs) / len(final_accs)
    print(f"\n  Final Test Accuracy: {final_avg*100:.1f}%")
    print(f"  Individual: {[f'{a*100:.0f}%' for a in final_accs]}")
    
    # =========================================================================
    # ROTATION INVARIANCE TEST
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Rotation Invariance Test")
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
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    
    print(f"\n  Stage 1 (frozen embedder): {stage1_avg*100:.1f}%")
    print(f"  Stage 2 (fine-tune both):  {final_avg*100:.1f}%")
    
    if final_avg > 0.85:
        print(f"\n  ✓ SUCCESS: Staged training works! Final accuracy: {final_avg*100:.1f}%")
    elif final_avg > 0.7:
        print(f"\n  ○ GOOD: Significant improvement. May need more iterations.")
    elif final_avg > 0.6:
        print(f"\n  ○ PARTIAL: Some learning. Architecture works but needs tuning.")
    else:
        print(f"\n  ✗ Need more debugging.")
    
    print("=" * 70)
    
    return stage1_avg, final_avg


if __name__ == "__main__":
    staged_training()