"""
Test Staged Training with REAL Images from SA-1B

Run from videoseal root:
    python test_staged_real_images.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image
import sys
import os
import glob
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from videoseal.models.g_embedder import GUnetEmbedder
from videoseal.models.g_extractor import build_g_extractor
from videoseal.modules.g_unet import GUNetMsg
from videoseal.modules.g_msg_processor import GMsgProcessor
from videoseal.modules.g_common import get_gspace


def build_models(nbits=8):
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
    
    extractor = build_g_extractor(
        nbits=nbits,
        depths=[2, 2, 2],
        dims=[32, 64, 128],
        group_type='C4',
    )
    
    return embedder, extractor


class RealImageLoader:
    """Load real images from SA-1B dataset"""
    
    def __init__(self, data_dir, img_size=128):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG']:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        
        print(f"  Found {len(self.image_paths)} images in {data_dir}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
    
    def get_batch(self, batch_size):
        """Get a batch of real images"""
        imgs = []
        paths = random.sample(self.image_paths, min(batch_size, len(self.image_paths)))
        
        for path in paths:
            try:
                img = Image.open(path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
            except Exception as e:
                # Skip corrupted images
                continue
        
        if len(imgs) < batch_size:
            # Pad with duplicates if needed
            while len(imgs) < batch_size:
                imgs.append(imgs[0].clone())
        
        return torch.stack(imgs[:batch_size])


def test_with_real_images(data_dir, num_iters=3000):
    """Test staged training with real images"""
    
    print("=" * 60)
    print(" Testing G-CNN with REAL Images")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load real images
    print("\nLoading images...")
    try:
        loader = RealImageLoader(data_dir, img_size=128)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please provide correct path to images")
        return
    
    # Build models
    print("\nBuilding models...")
    embedder, extractor = build_models(nbits=8)
    embedder = embedder.to(device)
    extractor = extractor.to(device)
    
    print(f"  Embedder params: {sum(p.numel() for p in embedder.parameters()):,}")
    print(f"  Extractor params: {sum(p.numel() for p in extractor.parameters()):,}")
    
    nbits = 8
    batch_size = 8
    freeze_iters = num_iters // 2
    
    # Stage 1: Freeze embedder
    print(f"\n--- Stage 1: Freeze embedder ({freeze_iters} iters) ---")
    embedder.eval()
    for p in embedder.parameters():
        p.requires_grad = False
    
    extractor.train()
    optimizer = AdamW(extractor.parameters(), lr=1e-3)
    
    best_acc = 0
    
    for it in range(num_iters):
        # Switch to Stage 2
        if it == freeze_iters:
            print(f"\n--- Stage 2: Unfreeze embedder ({num_iters - freeze_iters} iters) ---")
            embedder.train()
            for p in embedder.parameters():
                p.requires_grad = True
            optimizer = AdamW([
                {'params': embedder.parameters(), 'lr': 1e-4},
                {'params': extractor.parameters(), 'lr': 5e-4},
            ])
        
        optimizer.zero_grad()
        
        # Get real images
        imgs = loader.get_batch(batch_size).to(device)
        msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
        
        # Embed
        if it < freeze_iters:
            with torch.no_grad():
                imgs_w = embedder(imgs, msgs)
        else:
            imgs_w = embedder(imgs, msgs)
        
        imgs_w_norm = (imgs_w + 1) / 2  # [-1,1] -> [0,1]
        
        # Extract (no augmentation for this test)
        preds = extractor(imgs_w_norm)
        bit_preds = preds[:, 1:]
        
        # Loss
        decode_loss = F.binary_cross_entropy_with_logits(bit_preds, msgs)
        
        if it >= freeze_iters:
            img_loss = F.mse_loss(imgs_w, imgs * 2 - 1)
            loss = decode_loss + 0.1 * img_loss
        else:
            loss = decode_loss
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            predicted = (torch.sigmoid(bit_preds) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            if accuracy > best_acc:
                best_acc = accuracy
        
        if (it + 1) % 500 == 0:
            psnr_val = 10 * torch.log10(1 / F.mse_loss(imgs_w_norm, imgs)).item() if it >= freeze_iters else 0
            print(f"  Iter {it+1:5d} | Loss: {loss.item():.4f} | "
                  f"Acc: {accuracy*100:.1f}% | Best: {best_acc*100:.1f}% | PSNR: {psnr_val:.1f}")
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    embedder.eval()
    extractor.eval()
    
    test_accs = []
    test_psnrs = []
    
    with torch.no_grad():
        for _ in range(20):
            imgs = loader.get_batch(batch_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            
            # Accuracy
            preds = extractor(imgs_w_norm)
            bit_preds = preds[:, 1:]
            predicted = (torch.sigmoid(bit_preds) > 0.5).float()
            acc = (predicted == msgs).float().mean().item()
            test_accs.append(acc)
            
            # PSNR
            psnr_val = 10 * torch.log10(1 / F.mse_loss(imgs_w_norm, imgs)).item()
            test_psnrs.append(psnr_val)
    
    final_acc = sum(test_accs) / len(test_accs)
    final_psnr = sum(test_psnrs) / len(test_psnrs)
    
    print(f"\n  Final Bit Accuracy: {final_acc*100:.1f}%")
    print(f"  Final PSNR: {final_psnr:.1f} dB")
    print(f"  Best during training: {best_acc*100:.1f}%")
    
    # Rotation invariance test
    print("\n--- Rotation Invariance Test ---")
    with torch.no_grad():
        imgs = loader.get_batch(1).to(device)
        msgs = torch.randint(0, 2, (1, nbits)).float().to(device)
        imgs_w = embedder(imgs, msgs)
        imgs_w_norm = (imgs_w + 1) / 2
        
        for k in range(4):
            rotated = torch.rot90(imgs_w_norm, k, dims=[-2, -1])
            preds = extractor(rotated)
            bit_preds = preds[:, 1:]
            predicted = (torch.sigmoid(bit_preds) > 0.5).float()
            acc = (predicted == msgs).float().mean().item()
            print(f"  Rotation {k*90:3d}°: {acc*100:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    
    if final_acc > 0.85:
        print(f"\n  ✓ WORKS with real images! {final_acc*100:.1f}%")
        print("  → Proceed with full training using staged approach")
    elif final_acc > 0.7:
        print(f"\n  ○ Decent with real images: {final_acc*100:.1f}%")
        print("  → May need more iterations or tuning")
    else:
        print(f"\n  ✗ Struggles with real images: {final_acc*100:.1f}%")
        print("  → Need to investigate further")
    
    print("=" * 60)
    
    return final_acc


if __name__ == "__main__":
    # Path to your SA-1B images
    data_dir = "/home/sejal/Documents/Thesis/videoseal/large_experiments/omniseal/sa-1b/train"
    
    # Check if path exists
    if not os.path.exists(data_dir):
        print(f"Path not found: {data_dir}")
        print("Trying alternative paths...")
        
        alternatives = [
            "large_experiments/omniseal/sa-1b/train",
            "../large_experiments/omniseal/sa-1b/train",
            "data/sa-1b/train",
        ]
        
        for alt in alternatives:
            if os.path.exists(alt):
                data_dir = alt
                print(f"Found: {data_dir}")
                break
        else:
            print("Please provide correct path to images")
            sys.exit(1)
    
    test_with_real_images(data_dir, num_iters=3000)