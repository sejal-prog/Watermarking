"""
Gradient Flow Diagnostic - Check if gradients are actually flowing
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from videoseal.models.g_embedder import build_g_embedder
from videoseal.models.g_extractor import build_g_extractor


def check_gradient_flow():
    print("=" * 70)
    print(" Gradient Flow Diagnostic")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    embedder.train()
    extractor.train()
    
    # Forward pass
    imgs = torch.rand(2, 3, 128, 128, requires_grad=True).to(device)
    msgs = torch.randint(0, 2, (2, 32)).float().to(device)
    
    print(f"\n[Forward Pass]")
    
    imgs_w = embedder(imgs, msgs)
    print(f"  imgs_w: min={imgs_w.min().item():.3f}, max={imgs_w.max().item():.3f}, mean={imgs_w.mean().item():.3f}")
    
    # Check watermark magnitude
    imgs_normalized = imgs * 2 - 1
    watermark = imgs_w - imgs_normalized
    print(f"  watermark: min={watermark.min().item():.3f}, max={watermark.max().item():.3f}, mean={watermark.abs().mean().item():.3f}")
    
    imgs_w_norm = (imgs_w + 1) / 2
    logits = extractor(imgs_w_norm)
    print(f"  logits: min={logits.min().item():.3f}, max={logits.max().item():.3f}, std={logits.std().item():.3f}")
    
    # Loss
    loss = F.binary_cross_entropy_with_logits(logits, msgs)
    print(f"\n[Loss]")
    print(f"  BCE Loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    print(f"\n[Extractor Gradients]")
    ext_grad_norms = []
    for name, param in extractor.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            ext_grad_norms.append(norm)
            if 'fc' in name or 'outc' in name or len(ext_grad_norms) <= 5:
                print(f"  {name}: grad_norm = {norm:.6f}")
    print(f"  Total params with grad: {len(ext_grad_norms)}")
    print(f"  Grad norm range: [{min(ext_grad_norms):.6f}, {max(ext_grad_norms):.6f}]")
    print(f"  Grad norm mean: {sum(ext_grad_norms)/len(ext_grad_norms):.6f}")
    
    print(f"\n[Embedder Gradients]")
    emb_grad_norms = []
    for name, param in embedder.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            emb_grad_norms.append(norm)
            if 'msg' in name or 'outc' in name or len(emb_grad_norms) <= 5:
                print(f"  {name}: grad_norm = {norm:.6f}")
    print(f"  Total params with grad: {len(emb_grad_norms)}")
    print(f"  Grad norm range: [{min(emb_grad_norms):.6f}, {max(emb_grad_norms):.6f}]")
    print(f"  Grad norm mean: {sum(emb_grad_norms)/len(emb_grad_norms):.6f}")
    
    # Check if gradients are too small
    print(f"\n[Diagnosis]")
    if max(emb_grad_norms) < 1e-5:
        print(f"  ✗ Embedder gradients are too small!")
    else:
        print(f"  ✓ Embedder has reasonable gradients")
    
    if max(ext_grad_norms) < 1e-5:
        print(f"  ✗ Extractor gradients are too small!")
    else:
        print(f"  ✓ Extractor has reasonable gradients")


def check_watermark_magnitude():
    print("\n" + "=" * 70)
    print(" Watermark Magnitude Check")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    embedder = build_g_embedder(
        nbits=32,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    ).to(device)
    
    embedder.eval()
    
    with torch.no_grad():
        img = torch.rand(1, 3, 128, 128).to(device)
        msg = torch.randint(0, 2, (1, 32)).float().to(device)
        
        img_w = embedder(img, msg)
        
        # The input is preprocessed to [-1, 1] inside embedder
        img_preprocessed = img * 2 - 1
        
        # Watermark is the difference
        watermark = img_w - img_preprocessed
        
        print(f"\n  Input range: [{img_preprocessed.min():.3f}, {img_preprocessed.max():.3f}]")
        print(f"  Output range: [{img_w.min():.3f}, {img_w.max():.3f}]")
        print(f"  Watermark range: [{watermark.min():.3f}, {watermark.max():.3f}]")
        print(f"  Watermark mean abs: {watermark.abs().mean():.3f}")
        print(f"  Watermark std: {watermark.std():.3f}")
        
        # Check how much of the image is clipped
        clipped_low = (img_w <= -0.99).float().mean().item()
        clipped_high = (img_w >= 0.99).float().mean().item()
        print(f"  Clipped pixels: {(clipped_low + clipped_high) * 100:.1f}%")
        
        if watermark.abs().mean() > 0.5:
            print(f"\n  ✗ Watermark is too large! Network outputs random noise, not a small perturbation.")
            print(f"    This drowns out the message signal.")
        else:
            print(f"\n  ✓ Watermark magnitude seems reasonable")


def check_message_sensitivity():
    print("\n" + "=" * 70)
    print(" Message Sensitivity Check")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    embedder.eval()
    extractor.eval()
    
    with torch.no_grad():
        img = torch.rand(1, 3, 128, 128).to(device)
        
        msg0 = torch.zeros(1, 32).to(device)
        msg1 = torch.ones(1, 32).to(device)
        
        img_w0 = embedder(img, msg0)
        img_w1 = embedder(img, msg1)
        
        watermark_diff = (img_w0 - img_w1).abs().mean().item()
        
        print(f"\n  Same image, different messages (zeros vs ones):")
        print(f"  Watermarked image difference: {watermark_diff:.4f}")
        
        # Extract from both
        logits0 = extractor((img_w0 + 1) / 2)
        logits1 = extractor((img_w1 + 1) / 2)
        
        logits_diff = (logits0 - logits1).abs().mean().item()
        print(f"  Extracted logits difference: {logits_diff:.4f}")
        
        if watermark_diff < 0.01:
            print(f"\n  ✗ Different messages produce same watermark! Message is not embedded.")
        elif watermark_diff > 0.5:
            print(f"\n  ✓ Different messages produce very different watermarks (diff={watermark_diff:.2f})")
        else:
            print(f"\n  ✓ Messages affect watermark moderately")


if __name__ == "__main__":
    check_gradient_flow()
    check_watermark_magnitude()
    check_message_sensitivity()