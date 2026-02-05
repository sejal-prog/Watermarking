"""
Test: Is the watermark signal present on real images?

Compare watermark visibility on random noise vs real images.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys, os, glob, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from videoseal.models.g_embedder import GUnetEmbedder
from videoseal.modules.g_unet import GUNetMsg
from videoseal.modules.g_msg_processor import GMsgProcessor
from videoseal.modules.g_common import get_gspace


def build_embedder(nbits=8):
    gspace = get_gspace('C4')
    msg_processor = GMsgProcessor(nbits=nbits, hidden_size=nbits*2, group_order=4, msg_processor_type='binary+concat')
    gunet = GUNetMsg(msg_processor=msg_processor, in_channels=3, out_channels=3, z_channels=32, z_channels_mults=(1,2), num_blocks=1, group_type='C4')
    return GUnetEmbedder(gunet, msg_processor)


def load_real_images(image_dir, n=10, size=128):
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    paths = glob.glob(os.path.join(image_dir, '**/*.jpg'), recursive=True)[:100]
    imgs = []
    for p in random.sample(paths, min(n, len(paths))):
        try:
            imgs.append(transform(Image.open(p).convert('RGB')))
        except:
            pass
    return torch.stack(imgs)


def main():
    print("=" * 60)
    print(" Watermark Signal Analysis")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    embedder = build_embedder(8).to(device)
    embedder.eval()
    
    # Test 1: Random noise
    print("\n1. Random Noise Images:")
    noise_imgs = torch.rand(10, 3, 128, 128).to(device)
    msg0 = torch.zeros(10, 8).to(device)
    msg1 = torch.ones(10, 8).to(device)
    
    with torch.no_grad():
        w0 = embedder(noise_imgs, msg0)
        w1 = embedder(noise_imgs, msg1)
        
        diff_msg = (w0 - w1).abs().mean().item()
        diff_orig = (w0 - (noise_imgs * 2 - 1)).abs().mean().item()
        
        print(f"   Watermark diff (msg=0 vs msg=1): {diff_msg:.4f}")
        print(f"   Image change (orig vs watermarked): {diff_orig:.4f}")
        psnr = 10 * torch.log10(1 / F.mse_loss((w0+1)/2, noise_imgs)).item()
        print(f"   PSNR: {psnr:.1f} dB")
    
    # Test 2: Real images
    print("\n2. Real Images:")
    real_imgs = load_real_images("large_experiments/omniseal/sa-1b/train", n=10).to(device)
    
    with torch.no_grad():
        w0 = embedder(real_imgs, msg0)
        w1 = embedder(real_imgs, msg1)
        
        diff_msg = (w0 - w1).abs().mean().item()
        diff_orig = (w0 - (real_imgs * 2 - 1)).abs().mean().item()
        
        print(f"   Watermark diff (msg=0 vs msg=1): {diff_msg:.4f}")
        print(f"   Image change (orig vs watermarked): {diff_orig:.4f}")
        psnr = 10 * torch.log10(1 / F.mse_loss((w0+1)/2, real_imgs)).item()
        print(f"   PSNR: {psnr:.1f} dB")
    
    # Test 3: Check per-channel difference
    print("\n3. Per-channel watermark difference (msg=0 vs msg=1):")
    with torch.no_grad():
        w0_noise = embedder(noise_imgs, msg0)
        w1_noise = embedder(noise_imgs, msg1)
        w0_real = embedder(real_imgs, msg0)
        w1_real = embedder(real_imgs, msg1)
        
        for c, name in enumerate(['R', 'G', 'B']):
            diff_noise = (w0_noise[:, c] - w1_noise[:, c]).abs().mean().item()
            diff_real = (w0_real[:, c] - w1_real[:, c]).abs().mean().item()
            print(f"   {name}: Noise={diff_noise:.4f}, Real={diff_real:.4f}")
    
    # Test 4: Check watermark std across spatial locations
    print("\n4. Watermark spatial variance:")
    with torch.no_grad():
        watermark_noise = w0_noise - (noise_imgs * 2 - 1)
        watermark_real = w0_real - (real_imgs * 2 - 1)
        
        print(f"   Noise images - watermark std: {watermark_noise.std().item():.4f}")
        print(f"   Real images  - watermark std: {watermark_real.std().item():.4f}")
        print(f"   Noise images - watermark mean: {watermark_noise.abs().mean().item():.4f}")
        print(f"   Real images  - watermark mean: {watermark_real.abs().mean().item():.4f}")
    
    # Analysis
    print("\n" + "=" * 60)
    print(" ANALYSIS")
    print("=" * 60)
    
    with torch.no_grad():
        diff_noise = (embedder(noise_imgs, msg0) - embedder(noise_imgs, msg1)).abs().mean().item()
        diff_real = (embedder(real_imgs, msg0) - embedder(real_imgs, msg1)).abs().mean().item()
    
    if diff_real < diff_noise * 0.8:
        print(f"\n  ✗ Watermark signal is WEAKER on real images!")
        print(f"    Noise: {diff_noise:.4f}")
        print(f"    Real:  {diff_real:.4f} ({diff_real/diff_noise*100:.0f}% of noise)")
        print(f"\n  → Problem: Embedder produces weaker signal on structured images")
        print(f"  → Solution: Train embedder on real images (don't freeze)")
    else:
        print(f"\n  ✓ Watermark signal similar on both")
        print(f"    Noise: {diff_noise:.4f}")
        print(f"    Real:  {diff_real:.4f}")
        print(f"\n  → Problem is NOT the embedder signal")
        print(f"  → Problem might be extractor struggling with real image features")


if __name__ == "__main__":
    main()