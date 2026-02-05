"""
DIRECT COMPARISON: VideoSeal Original vs G-CNN

Find exactly which component fails:
1. VideoSeal embedder + VideoSeal extractor (baseline - should work)
2. VideoSeal embedder + G-CNN extractor (test extractor)
3. G-CNN embedder + VideoSeal extractor (test embedder)  
4. G-CNN embedder + G-CNN extractor (our full system)

Run: python test_compare.py
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image
import sys, os, glob, random
import omegaconf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# G-CNN imports
from videoseal.models.g_embedder import GUnetEmbedder
from videoseal.models.g_extractor import build_g_extractor
from videoseal.modules.g_unet import GUNetMsg
from videoseal.modules.g_msg_processor import GMsgProcessor
from videoseal.modules.g_common import get_gspace

# VideoSeal original imports
from videoseal.models.embedder import build_embedder
from videoseal.models.extractor import build_extractor


def build_gcnn_embedder(nbits=8):
    gspace = get_gspace('C4')
    msg_processor = GMsgProcessor(nbits=nbits, hidden_size=nbits*2, group_order=4, msg_processor_type='binary+concat')
    gunet = GUNetMsg(msg_processor=msg_processor, in_channels=3, out_channels=3, z_channels=32, z_channels_mults=(1,2), num_blocks=1, group_type='C4')
    return GUnetEmbedder(gunet, msg_processor)


def build_gcnn_extractor(nbits=8):
    return build_g_extractor(nbits=nbits, depths=[2,2,2], dims=[32,64,128], group_type='C4')


def build_videoseal_embedder(nbits=8):
    """Build original VideoSeal UNet embedder"""
    cfg = omegaconf.OmegaConf.load('configs/embedder.yaml')
    model_cfg = cfg['unet_small2_quant']
    return build_embedder('unet_small2_quant', model_cfg, nbits, hidden_size_multiplier=2)


def build_videoseal_extractor(nbits=8):
    """Build original VideoSeal ConvNeXt extractor"""
    cfg = omegaconf.OmegaConf.load('configs/extractor.yaml')
    model_cfg = cfg['convnext_tiny']
    return build_extractor('convnext_tiny', model_cfg, img_size=128, nbits=nbits)


class ImageLoader:
    def __init__(self, image_dir, img_size=128):
        self.transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        self.paths = glob.glob(os.path.join(image_dir, '**/*.jpg'), recursive=True)[:500]
        print(f"  Loaded {len(self.paths)} images")
    
    def get_batch(self, bs, device):
        imgs = []
        for p in random.sample(self.paths, bs):
            try:
                imgs.append(self.transform(Image.open(p).convert('RGB')))
            except:
                imgs.append(torch.rand(3, 128, 128))
        return torch.stack(imgs).to(device)


def train_and_eval(embedder, extractor, loader, device, name, num_iters=3000):
    """Train embedder+extractor pair and return final accuracy"""
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")
    
    embedder = embedder.to(device)
    extractor = extractor.to(device)
    
    embedder_params = sum(p.numel() for p in embedder.parameters())
    extractor_params = sum(p.numel() for p in extractor.parameters())
    print(f"  Embedder: {embedder_params:,} params")
    print(f"  Extractor: {extractor_params:,} params")
    
    embedder.train()
    extractor.train()
    
    optimizer = AdamW([
        {'params': embedder.parameters(), 'lr': 5e-4},
        {'params': extractor.parameters(), 'lr': 1e-3},
    ])
    
    nbits = 8
    best_acc = 0
    
    for it in range(num_iters):
        optimizer.zero_grad()
        
        imgs = loader.get_batch(8, device)
        msgs = embedder.get_random_msg(8).float().to(device)
        
        # Forward through embedder
        imgs_w = embedder(imgs, msgs)
        
        # Normalize to [0,1] for extractor if needed
        if imgs_w.min() < 0:  # Output is [-1,1]
            imgs_w_norm = (imgs_w + 1) / 2
        else:
            imgs_w_norm = imgs_w
        imgs_w_norm = torch.clamp(imgs_w_norm, 0, 1)
        
        # Forward through extractor
        preds = extractor(imgs_w_norm)
        
        # Handle different output formats
        if preds.dim() == 4:  # [B, C, H, W] - pixelwise
            preds = preds.mean(dim=[-2, -1])  # [B, C]
        
        if preds.shape[1] == nbits + 1:  # Has detection channel
            bit_preds = preds[:, 1:]
        else:
            bit_preds = preds
        
        # Loss
        decode_loss = F.binary_cross_entropy_with_logits(bit_preds, msgs)
        img_loss = F.mse_loss(imgs_w_norm, imgs)
        loss = decode_loss + 0.1 * img_loss
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            acc = ((torch.sigmoid(bit_preds) > 0.5).float() == msgs).float().mean().item()
            best_acc = max(best_acc, acc)
        
        if (it + 1) % 500 == 0:
            print(f"  Iter {it+1:5d} | Loss: {loss.item():.4f} | Acc: {acc*100:.1f}% | Best: {best_acc*100:.1f}%")
    
    # Final eval
    embedder.eval()
    extractor.eval()
    
    accs = []
    with torch.no_grad():
        for _ in range(30):
            imgs = loader.get_batch(8, device)
            msgs = embedder.get_random_msg(8).float().to(device)
            imgs_w = embedder(imgs, msgs)
            if imgs_w.min() < 0:
                imgs_w_norm = (imgs_w + 1) / 2
            else:
                imgs_w_norm = imgs_w
            imgs_w_norm = torch.clamp(imgs_w_norm, 0, 1)
            preds = extractor(imgs_w_norm)
            if preds.dim() == 4:
                preds = preds.mean(dim=[-2, -1])
            if preds.shape[1] == nbits + 1:
                bit_preds = preds[:, 1:]
            else:
                bit_preds = preds
            accs.append(((torch.sigmoid(bit_preds) > 0.5).float() == msgs).float().mean().item())
    
    final = sum(accs) / len(accs)
    print(f"\n  FINAL: {final*100:.1f}% | Best: {best_acc*100:.1f}%")
    
    return final, best_acc


def main():
    print("=" * 60)
    print(" COMPARISON: VideoSeal vs G-CNN Components")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    loader = ImageLoader("large_experiments/omniseal/sa-1b/train")
    
    results = {}
    
    # Test 1: VideoSeal + VideoSeal (baseline)
    try:
        emb = build_videoseal_embedder(8)
        ext = build_videoseal_extractor(8)
        final, best = train_and_eval(emb, ext, loader, device, "VideoSeal Embedder + VideoSeal Extractor", 3000)
        results["VS+VS"] = (final, best)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["VS+VS"] = (0, 0)
    
    # Test 2: VideoSeal embedder + G-CNN extractor
    try:
        emb = build_videoseal_embedder(8)
        ext = build_gcnn_extractor(8)
        final, best = train_and_eval(emb, ext, loader, device, "VideoSeal Embedder + G-CNN Extractor", 3000)
        results["VS+GCNN"] = (final, best)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["VS+GCNN"] = (0, 0)
    
    # Test 3: G-CNN embedder + VideoSeal extractor
    try:
        emb = build_gcnn_embedder(8)
        ext = build_videoseal_extractor(8)
        final, best = train_and_eval(emb, ext, loader, device, "G-CNN Embedder + VideoSeal Extractor", 3000)
        results["GCNN+VS"] = (final, best)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["GCNN+VS"] = (0, 0)
    
    # Test 4: G-CNN + G-CNN
    try:
        emb = build_gcnn_embedder(8)
        ext = build_gcnn_extractor(8)
        final, best = train_and_eval(emb, ext, loader, device, "G-CNN Embedder + G-CNN Extractor", 3000)
        results["GCNN+GCNN"] = (final, best)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["GCNN+GCNN"] = (0, 0)
    
    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"\n  {'Combination':<35} {'Final':<10} {'Best':<10}")
    print("-" * 55)
    for name, (final, best) in results.items():
        status = "✓" if final > 0.85 else "○" if final > 0.70 else "✗"
        print(f"  {name:<35} {final*100:>6.1f}%    {best*100:>6.1f}%  {status}")
    
    # Analysis
    print("\n" + "=" * 60)
    print(" ANALYSIS")
    print("=" * 60)
    
    vs_vs = results.get("VS+VS", (0,0))[0]
    vs_gcnn = results.get("VS+GCNN", (0,0))[0]
    gcnn_vs = results.get("GCNN+VS", (0,0))[0]
    gcnn_gcnn = results.get("GCNN+GCNN", (0,0))[0]
    
    if vs_vs > 0.85:
        print(f"\n  ✓ VideoSeal baseline works: {vs_vs*100:.1f}%")
    
    if vs_gcnn > 0.85:
        print(f"  ✓ G-CNN extractor works with VS embedder: {vs_gcnn*100:.1f}%")
        print("    → G-CNN extractor is OK")
    elif vs_gcnn < gcnn_gcnn:
        print(f"  ✗ G-CNN extractor fails with VS embedder: {vs_gcnn*100:.1f}%")
        print("    → G-CNN EXTRACTOR is the problem")
    
    if gcnn_vs > 0.85:
        print(f"  ✓ G-CNN embedder works with VS extractor: {gcnn_vs*100:.1f}%")
        print("    → G-CNN embedder is OK")
    elif gcnn_vs < gcnn_gcnn:
        print(f"  ✗ G-CNN embedder fails with VS extractor: {gcnn_vs*100:.1f}%")
        print("    → G-CNN EMBEDDER is the problem")
    
    if gcnn_gcnn < 0.70:
        if gcnn_vs > gcnn_gcnn and vs_gcnn > gcnn_gcnn:
            print(f"\n  → BOTH G-CNN components have issues")
        elif gcnn_vs < vs_vs * 0.8:
            print(f"\n  → G-CNN EMBEDDER is the main problem")
        elif vs_gcnn < vs_vs * 0.8:
            print(f"\n  → G-CNN EXTRACTOR is the main problem")
    
    print("=" * 60)


if __name__ == "__main__":
    main()