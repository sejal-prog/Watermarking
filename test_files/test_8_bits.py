"""
Long Training Test - 8 Bits

8 bits showed 71% at 1000 iterations.
Let's train for 10,000 iterations and see if we can hit 90%+.

If this works, we can proceed with actual training on real data.

Run from videoseal root:
    python test_8bits_long.py
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


def test_8bits_long():
    print("=" * 70)
    print(" Long Training Test - 8 Bits, 10000 Iterations")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    nbits = 8
    batch_size = 16
    img_size = 128  # Larger image
    
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
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Bits: {nbits}")
    
    # Freeze embedder
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    
    extractor.train()
    
    # Optimizer with cosine schedule
    optimizer = Adam(extractor.parameters(), lr=1e-3)
    num_iterations = 10000
    scheduler = CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=1e-5)
    
    print(f"\n  Training for {num_iterations} iterations...")
    print(f"  Learning rate: 1e-3 → 1e-5 (cosine)")
    print("-" * 70)
    
    best_acc = 0
    history = []
    
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
        
        history.append(accuracy)
        
        if accuracy > best_acc:
            best_acc = accuracy
        
        if (iteration + 1) % 1000 == 0 or iteration == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Iter {iteration+1:5d} | Loss: {loss.item():.4f} | "
                  f"Bit Acc: {accuracy*100:.1f}% | Best: {best_acc*100:.1f}% | LR: {lr:.2e}")
    
    print("-" * 70)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print(" Final Evaluation")
    print("=" * 70)
    
    extractor.eval()
    
    test_accs = []
    with torch.no_grad():
        for _ in range(50):  # More test samples
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            logits = extractor(imgs_w_norm)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            test_accs.append(accuracy)
    
    final_avg = sum(test_accs) / len(test_accs)
    final_std = (sum((a - final_avg)**2 for a in test_accs) / len(test_accs)) ** 0.5
    
    print(f"\n  Final Test Accuracy: {final_avg*100:.1f}% ± {final_std*100:.1f}%")
    print(f"  Best during training: {best_acc*100:.1f}%")
    
    # Per-bit accuracy
    print("\n" + "=" * 70)
    print(" Per-Bit Accuracy")
    print("=" * 70)
    
    bit_correct = torch.zeros(nbits)
    bit_total = torch.zeros(nbits)
    
    with torch.no_grad():
        for _ in range(200):
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
    
    print(f"\n  Per-bit: {[f'{a*100:.0f}%' for a in bit_accuracy.tolist()]}")
    print(f"  Mean: {bit_accuracy.mean()*100:.1f}%")
    print(f"  Min:  {bit_accuracy.min()*100:.1f}% (bit {bit_accuracy.argmin().item()})")
    print(f"  Max:  {bit_accuracy.max()*100:.1f}% (bit {bit_accuracy.argmax().item()})")
    
    # Rotation invariance
    print("\n" + "=" * 70)
    print(" Rotation Invariance Test")
    print("=" * 70)
    
    rotation_accs = []
    with torch.no_grad():
        for _ in range(20):
            img = torch.rand(1, 3, img_size, img_size).to(device)
            msg = torch.randint(0, 2, (1, nbits)).float().to(device)
            
            img_w = embedder(img, msg)
            img_w_norm = (img_w + 1) / 2
            
            for k in range(4):
                img_rotated = torch.rot90(img_w_norm, k, dims=[-2, -1])
                logits = extractor(img_rotated)
                predicted = (torch.sigmoid(logits) > 0.5).float()
                acc = (predicted == msg).float().mean().item()
                rotation_accs.append((k * 90, acc))
    
    # Average by rotation
    for rot in [0, 90, 180, 270]:
        accs = [a for r, a in rotation_accs if r == rot]
        avg = sum(accs) / len(accs)
        print(f"  Rotation {rot:3d}°: {avg*100:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    
    if final_avg > 0.90:
        print(f"\n  ✓ EXCELLENT! {final_avg*100:.1f}% - Ready for real training!")
        status = "PASS"
    elif final_avg > 0.80:
        print(f"\n  ✓ GOOD! {final_avg*100:.1f}% - Should work with real data")
        status = "PASS"
    elif final_avg > 0.70:
        print(f"\n  ○ OK: {final_avg*100:.1f}% - May need more iterations or tuning")
        status = "MARGINAL"
    else:
        print(f"\n  ✗ {final_avg*100:.1f}% - Need to investigate further")
        status = "FAIL"
    
    print("=" * 70)
    
    # Save model if good
    if status == "PASS":
        save_path = "gcnn_8bit_extractor.pt"
        torch.save({
            'extractor_state_dict': extractor.state_dict(),
            'final_accuracy': final_avg,
            'best_accuracy': best_acc,
            'nbits': nbits,
        }, save_path)
        print(f"\n  Model saved to: {save_path}")
    
    return final_avg, best_acc, history


if __name__ == "__main__":
    final_acc, best_acc, history = test_8bits_long()
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        # Smooth the history
        window = 100
        smoothed = [sum(history[max(0,i-window):i+1])/min(i+1, window) for i in range(len(history))]
        
        plt.figure(figsize=(12, 5))
        plt.plot(smoothed, label='Bit Accuracy (smoothed)')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Random (50%)')
        plt.axhline(y=0.9, color='g', linestyle='--', label='Target (90%)')
        plt.xlabel('Iteration')
        plt.ylabel('Bit Accuracy')
        plt.title('8-Bit G-CNN Watermarking Training (Frozen Embedder)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('8bit_long_training.png', dpi=100)
        print(f"\n  Plot saved to: 8bit_long_training.png")
        plt.show()
    except Exception as e:
        print(f"\n  Could not save plot: {e}")