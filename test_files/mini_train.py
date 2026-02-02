"""
Mini Training Script for G-CNN Watermarking

This script does a quick training test to verify:
1. Embedder and Extractor can learn to work together
2. Bit accuracy improves above 50%
3. Losses decrease

Run from videoseal root:
    python test_mini_train.py
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


def compute_bit_accuracy(logits: torch.Tensor, target_msg: torch.Tensor) -> float:
    """Compute bit accuracy between predicted logits and target message."""
    predicted_bits = (torch.sigmoid(logits) > 0.5).float()
    accuracy = (predicted_bits == target_msg.float()).float().mean().item()
    return accuracy


def mini_train():
    print("=" * 70)
    print(" Mini Training Test - G-CNN Watermarking")
    print("=" * 70)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Build models
    print("\nBuilding models...")
    embedder = build_g_embedder(
        nbits=32,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    ).to(device)
    
    extractor = build_g_extractor(
        nbits=32,
        depths=[2, 2, 2],
        dims=[32, 64, 128],
        group_type="C4"
    ).to(device)
    
    print(f"  Embedder params: {sum(p.numel() for p in embedder.parameters()):,}")
    print(f"  Extractor params: {sum(p.numel() for p in extractor.parameters()):,}")
    
    # Optimizer - use higher learning rate
    params = list(embedder.parameters()) + list(extractor.parameters())
    optimizer = Adam(params, lr=1e-2)  # Increased from 1e-3
    
    # Training settings
    num_iterations = 500
    batch_size = 4
    img_size = 128
    nbits = 32
    
    # Loss weights
    # With residual connection and watermark_strength, image loss is automatically low
    # Focus entirely on message loss
    lambda_msg = 1.0      # Message loss weight
    lambda_img = 0.0      # Disable image loss - residual handles this
    
    print(f"\nTraining for {num_iterations} iterations...")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Message bits: {nbits}")
    print(f"  Initial watermark strength: {embedder.gunet.watermark_strength.item():.3f}")
    print("-" * 70)
    
    # Training loop
    embedder.train()
    extractor.train()
    
    history = {
        'loss': [],
        'msg_loss': [],
        'img_loss': [],
        'bit_accuracy': []
    }
    
    for iteration in range(num_iterations):
        # Generate random batch
        imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
        msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Embed watermark
        imgs_w = embedder(imgs, msgs)  # Output in [-1, 1]
        
        # Normalize for extractor: [-1, 1] -> [0, 1]
        imgs_w_norm = (imgs_w + 1) / 2
        
        # Extract message
        logits = extractor(imgs_w_norm)
        
        # Losses
        # 1. Message loss (BCE)
        msg_loss = F.binary_cross_entropy_with_logits(logits, msgs)
        
        # 2. Image reconstruction loss (keep watermark invisible)
        # Normalize original imgs to [-1, 1] for comparison
        imgs_normalized = imgs * 2 - 1
        img_loss = F.mse_loss(imgs_w, imgs_normalized)
        
        # Total loss
        loss = lambda_msg * msg_loss + lambda_img * img_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            bit_acc = compute_bit_accuracy(logits, msgs)
        
        # Store history
        history['loss'].append(loss.item())
        history['msg_loss'].append(msg_loss.item())
        history['img_loss'].append(img_loss.item())
        history['bit_accuracy'].append(bit_acc)
        
        # Print progress
        if (iteration + 1) % 50 == 0 or iteration == 0:
            # Get watermark strength (now a buffer, not parameter)
            wm_strength = embedder.gunet.watermark_strength.item()
            print(f"  Iter {iteration+1:4d} | Loss: {loss.item():.4f} | "
                  f"Msg: {msg_loss.item():.4f} | Img: {img_loss.item():.4f} | "
                  f"Bit Acc: {bit_acc*100:.1f}% | WM Str: {wm_strength:.3f} (fixed)")
    
    print("-" * 70)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print(" Final Evaluation")
    print("=" * 70)
    
    embedder.eval()
    extractor.eval()
    
    # Test on fresh data
    test_accuracies = []
    
    with torch.no_grad():
        for _ in range(10):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            logits = extractor(imgs_w_norm)
            
            acc = compute_bit_accuracy(logits, msgs)
            test_accuracies.append(acc)
    
    final_acc = sum(test_accuracies) / len(test_accuracies)
    
    print(f"\n  Test Bit Accuracy (10 batches): {final_acc*100:.1f}%")
    print(f"  Individual: {[f'{a*100:.0f}%' for a in test_accuracies]}")
    
    # Check improvement
    initial_acc = history['bit_accuracy'][0]
    final_train_acc = history['bit_accuracy'][-1]
    
    print(f"\n  Training Progress:")
    print(f"    Initial accuracy: {initial_acc*100:.1f}%")
    print(f"    Final accuracy:   {final_train_acc*100:.1f}%")
    print(f"    Improvement:      {(final_train_acc - initial_acc)*100:+.1f}%")
    
    # Test rotation invariance of extractor
    print("\n" + "=" * 70)
    print(" Rotation Invariance Test (After Training)")
    print("=" * 70)
    
    with torch.no_grad():
        img = torch.rand(1, 3, img_size, img_size).to(device)
        msg = torch.randint(0, 2, (1, nbits)).float().to(device)
        
        # Embed
        img_w = embedder(img, msg)
        img_w_norm = (img_w + 1) / 2
        
        print(f"\n  Original message: {msg[0, :10].int().tolist()}...")
        
        for k in range(4):
            # Rotate watermarked image
            img_w_rotated = torch.rot90(img_w_norm, k, dims=[-2, -1])
            
            # Extract
            logits = extractor(img_w_rotated)
            extracted = (torch.sigmoid(logits) > 0.5).float()
            
            acc = (extracted == msg).float().mean().item()
            print(f"  Rotation {k*90:3d}°: {extracted[0, :10].int().tolist()}... | Accuracy: {acc*100:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    
    if final_acc > 0.7:
        print(f"\n  ✓ SUCCESS: Bit accuracy improved to {final_acc*100:.1f}%")
        print(f"    The architecture is working correctly!")
        print(f"    With longer training, expect 95%+ accuracy.")
    elif final_acc > 0.55:
        print(f"\n  ○ PARTIAL: Bit accuracy improved to {final_acc*100:.1f}%")
        print(f"    Learning is happening, but may need more iterations or tuning.")
    else:
        print(f"\n  ✗ ISSUE: Bit accuracy stayed at {final_acc*100:.1f}%")
        print(f"    May need to investigate architecture or hyperparameters.")
    
    print("\n" + "=" * 70)
    
    return history, final_acc


def plot_history(history):
    """Plot training history if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(history['loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Iteration')
        
        axes[0, 1].plot(history['msg_loss'])
        axes[0, 1].set_title('Message Loss (BCE)')
        axes[0, 1].set_xlabel('Iteration')
        
        axes[1, 0].plot(history['img_loss'])
        axes[1, 0].set_title('Image Loss (MSE)')
        axes[1, 0].set_xlabel('Iteration')
        
        axes[1, 1].plot([a * 100 for a in history['bit_accuracy']])
        axes[1, 1].axhline(y=50, color='r', linestyle='--', label='Random (50%)')
        axes[1, 1].set_title('Bit Accuracy (%)')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylim([0, 100])
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('mini_train_history.png', dpi=100)
        print(f"\n  Training plot saved to: mini_train_history.png")
        plt.show()
        
    except ImportError:
        print("\n  (matplotlib not available, skipping plot)")


if __name__ == "__main__":
    history, final_acc = mini_train()
    plot_history(history)