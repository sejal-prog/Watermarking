"""
Test Staged Training Locally

This mimics the real VideoSeal training setup to verify
staged training works before running full training.

Run from videoseal root:
    python test_staged_real.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from videoseal.models.g_embedder import build_g_embedder, GUnetEmbedder
from videoseal.models.g_extractor import build_g_extractor
from videoseal.modules.g_unet import GUNetMsg
from videoseal.modules.g_msg_processor import GMsgProcessor
from videoseal.modules.g_common import get_gspace
import omegaconf


def build_models(nbits=8):
    """Build G-CNN models matching config"""
    
    # Embedder config (matches unet_gcnn in embedder.yaml)
    gspace = get_gspace('C4')
    group_order = gspace.fibergroup.order()
    hidden_size = nbits * 2
    
    msg_processor = GMsgProcessor(
        nbits=nbits,
        hidden_size=hidden_size,
        group_order=group_order,
        msg_processor_type='binary+concat',
    )
    
    gunet = GUNetMsg(
        msg_processor=msg_processor,
        in_channels=3,
        out_channels=3,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type='C4',
    )
    
    embedder = GUnetEmbedder(gunet, msg_processor)
    
    # Extractor (matches g_convnext in extractor.yaml)
    extractor = build_g_extractor(
        nbits=nbits,
        depths=[2, 2, 2],
        dims=[32, 64, 128],
        group_type='C4',
    )
    
    return embedder, extractor


def test_joint_training(embedder, extractor, device, num_iters=2000):
    """Test joint training (both models train together)"""
    print("\n" + "=" * 60)
    print(" Test 1: JOINT Training (baseline)")
    print("=" * 60)
    
    embedder = embedder.to(device)
    extractor = extractor.to(device)
    embedder.train()
    extractor.train()
    
    # Reset weights
    for m in embedder.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    for m in extractor.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    
    optimizer = AdamW(
        list(embedder.parameters()) + list(extractor.parameters()),
        lr=1e-3
    )
    
    nbits = 8
    batch_size = 8
    img_size = 128
    
    best_acc = 0
    
    for it in range(num_iters):
        optimizer.zero_grad()
        
        # Random images (simulating real data)
        imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
        msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
        
        # Forward
        imgs_w = embedder(imgs, msgs)  # Output in [-1, 1]
        imgs_w_norm = (imgs_w + 1) / 2  # Convert to [0, 1] for extractor
        
        preds = extractor(imgs_w_norm)  # [B, 1+nbits]
        bit_preds = preds[:, 1:]  # Skip detection channel
        
        # Losses (similar to VideoSeal)
        decode_loss = F.binary_cross_entropy_with_logits(bit_preds, msgs)
        img_loss = F.mse_loss(imgs_w, imgs * 2 - 1)  # Compare in [-1, 1]
        
        loss = decode_loss + 0.1 * img_loss
        loss.backward()
        optimizer.step()
        
        # Accuracy
        with torch.no_grad():
            predicted = (torch.sigmoid(bit_preds) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            if accuracy > best_acc:
                best_acc = accuracy
        
        if (it + 1) % 500 == 0:
            print(f"  Iter {it+1:5d} | Loss: {loss.item():.4f} | "
                  f"Bit Acc: {accuracy*100:.1f}% | Best: {best_acc*100:.1f}%")
    
    # Final eval
    embedder.eval()
    extractor.eval()
    
    test_accs = []
    with torch.no_grad():
        for _ in range(20):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            preds = extractor(imgs_w_norm)
            bit_preds = preds[:, 1:]
            predicted = (torch.sigmoid(bit_preds) > 0.5).float()
            acc = (predicted == msgs).float().mean().item()
            test_accs.append(acc)
    
    final_acc = sum(test_accs) / len(test_accs)
    print(f"\n  Final Accuracy: {final_acc*100:.1f}%")
    
    return final_acc


def test_staged_training(embedder, extractor, device, freeze_iters=1000, total_iters=3000):
    """Test staged training (freeze embedder first, then unfreeze)"""
    print("\n" + "=" * 60)
    print(" Test 2: STAGED Training (freeze embedder first)")
    print("=" * 60)
    print(f"  Stage 1: Freeze embedder for {freeze_iters} iters")
    print(f"  Stage 2: Unfreeze and fine-tune for {total_iters - freeze_iters} iters")
    
    embedder = embedder.to(device)
    extractor = extractor.to(device)
    
    # Reset weights
    for m in embedder.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    for m in extractor.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    
    nbits = 8
    batch_size = 8
    img_size = 128
    
    # Stage 1: Freeze embedder
    print("\n  --- Stage 1: Training extractor only ---")
    embedder.eval()
    for p in embedder.parameters():
        p.requires_grad = False
    
    extractor.train()
    optimizer = AdamW(extractor.parameters(), lr=1e-3)
    
    best_acc = 0
    
    for it in range(freeze_iters):
        optimizer.zero_grad()
        
        imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
        msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
        
        with torch.no_grad():
            imgs_w = embedder(imgs, msgs)
        imgs_w_norm = (imgs_w + 1) / 2
        
        preds = extractor(imgs_w_norm)
        bit_preds = preds[:, 1:]
        
        loss = F.binary_cross_entropy_with_logits(bit_preds, msgs)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            predicted = (torch.sigmoid(bit_preds) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            if accuracy > best_acc:
                best_acc = accuracy
        
        if (it + 1) % 500 == 0:
            print(f"  Iter {it+1:5d} | Loss: {loss.item():.4f} | "
                  f"Bit Acc: {accuracy*100:.1f}% | Best: {best_acc*100:.1f}%")
    
    stage1_acc = best_acc
    print(f"\n  Stage 1 Complete: {stage1_acc*100:.1f}%")
    
    # Stage 2: Unfreeze embedder
    print("\n  --- Stage 2: Fine-tuning both ---")
    embedder.train()
    for p in embedder.parameters():
        p.requires_grad = True
    
    # New optimizer with lower LR for embedder
    optimizer = AdamW([
        {'params': embedder.parameters(), 'lr': 1e-4},  # Lower LR
        {'params': extractor.parameters(), 'lr': 5e-4},
    ])
    
    for it in range(freeze_iters, total_iters):
        optimizer.zero_grad()
        
        imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
        msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
        
        imgs_w = embedder(imgs, msgs)
        imgs_w_norm = (imgs_w + 1) / 2
        
        preds = extractor(imgs_w_norm)
        bit_preds = preds[:, 1:]
        
        decode_loss = F.binary_cross_entropy_with_logits(bit_preds, msgs)
        img_loss = F.mse_loss(imgs_w, imgs * 2 - 1)
        
        loss = decode_loss + 0.1 * img_loss
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            predicted = (torch.sigmoid(bit_preds) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            if accuracy > best_acc:
                best_acc = accuracy
        
        if (it + 1) % 500 == 0:
            print(f"  Iter {it+1:5d} | Loss: {loss.item():.4f} | "
                  f"Bit Acc: {accuracy*100:.1f}% | Best: {best_acc*100:.1f}%")
    
    # Final eval
    embedder.eval()
    extractor.eval()
    
    test_accs = []
    with torch.no_grad():
        for _ in range(20):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            preds = extractor(imgs_w_norm)
            bit_preds = preds[:, 1:]
            predicted = (torch.sigmoid(bit_preds) > 0.5).float()
            acc = (predicted == msgs).float().mean().item()
            test_accs.append(acc)
    
    final_acc = sum(test_accs) / len(test_accs)
    print(f"\n  Final Accuracy: {final_acc*100:.1f}%")
    
    return final_acc, stage1_acc


def main():
    print("=" * 60)
    print(" Staged Training Test for G-CNN")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Build fresh models for each test
    print("\nBuilding models...")
    embedder1, extractor1 = build_models(nbits=8)
    embedder2, extractor2 = build_models(nbits=8)
    
    print(f"  Embedder params: {sum(p.numel() for p in embedder1.parameters()):,}")
    print(f"  Extractor params: {sum(p.numel() for p in extractor1.parameters()):,}")
    
    # Test 1: Joint training
    joint_acc = test_joint_training(embedder1, extractor1, device, num_iters=2000)
    
    # Test 2: Staged training
    staged_acc, stage1_acc = test_staged_training(
        embedder2, extractor2, device, 
        freeze_iters=1500, total_iters=3000
    )
    
    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"\n  Joint Training:  {joint_acc*100:.1f}%")
    print(f"  Staged Training: {staged_acc*100:.1f}% (Stage 1: {stage1_acc*100:.1f}%)")
    
    if staged_acc > joint_acc + 0.1:
        print(f"\n  ✓ Staged training is {(staged_acc-joint_acc)*100:.1f}% better!")
        print("  → Implement staged training in train.py")
    elif staged_acc > joint_acc:
        print(f"\n  ○ Staged training slightly better (+{(staged_acc-joint_acc)*100:.1f}%)")
    else:
        print(f"\n  ✗ Staged training not better. Need different approach.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()