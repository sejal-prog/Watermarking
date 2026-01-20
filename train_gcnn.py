"""
Training script for GCNN VideoSeal model with 128-bit messages.
"""

import torch
from videoseal.models.embedder import build_embedder
from videoseal.models.extractor import build_extractor
from videoseal.models.videoseal import Videoseal
from videoseal.augmentation.augmenter import Augmenter
from omegaconf import OmegaConf


def main():
    print("=" * 60)
    print("GCNN VIDEOSEAL TRAINING SETUP (128 bits)")
    print("=" * 60)
    
    # Load config
    cfg = OmegaConf.load('configs/unet_gcnn_128bits.yaml')
    
    print(f"\n📋 Configuration:")
    print(f"   Model: {cfg.name}")
    print(f"   Message bits: {cfg.nbits}")
    print(f"   Hidden size: {cfg.msg_processor.hidden_size}")
    print(f"   Base channels: {cfg.unet.z_channels}")
    print(f"   Batch size: {cfg.training.batch_size}")
    print(f"   Total epochs: {cfg.training.epochs}")
    
    # Build embedder (GCNN)
    print("\n🔧 Building GCNN embedder...")
    embedder = build_embedder('unet_gcnn', cfg, nbits=cfg.nbits)
    embedder_params = sum(p.numel() for p in embedder.parameters())
    print(f"   ✓ Parameters: {embedder_params:,}")
    print(f"   ✓ Memory: ~{embedder_params * 4 / 1024**2:.1f} MB (fp32)")
    
    # Build extractor (regular ViT)
    print("\n🔧 Building extractor (ViT)...")
    
    # Example extractor config
    extractor_cfg = OmegaConf.create({
        'encoder': {
            'img_size': 256,
            'patch_size': 16,
            'in_chans': 3,
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'mlp_ratio': 4.0,
            'out_chans': 384,
        },
        'pixel_decoder': {
            'nbits': cfg.nbits,
            'in_channels': 384,
            'hidden_channels': 256,
        }
    })
    
    extractor = build_extractor('sam', extractor_cfg, img_size=256, nbits=cfg.nbits)
    extractor_params = sum(p.numel() for p in extractor.parameters())
    print(f"   ✓ Parameters: {extractor_params:,}")
    print(f"   ✓ Memory: ~{extractor_params * 4 / 1024**2:.1f} MB (fp32)")
    
    # Build augmenter
    print("\n🔧 Building augmenter...")
    augmenter_cfg = OmegaConf.create({
        'geometric_p': 0.5,
        'valuemetric_p': 0.5,
        'compression_p': 0.5,
    })
    augmenter = Augmenter(**augmenter_cfg)
    print("   ✓ Augmenter ready")
    
    # Build complete model
    print("\n🔧 Building VideoSeal model...")
    model = Videoseal(
        embedder=embedder,
        detector=extractor,
        augmenter=augmenter,
        scaling_w=2.0,
        img_size=256,
        chunk_size=8,
        step_size=4,
    )
    
    total_params = embedder_params + extractor_params
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Total memory: ~{total_params * 4 / 1024**2:.1f} MB (fp32)")
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\n💻 Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Quick test
    print("\n🧪 Running quick test...")
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 256, 256).to(device)
    masks = torch.ones(batch_size, 1, 256, 256).to(device)
    msgs = torch.randint(0, 2, (batch_size, cfg.nbits)).to(device)
    
    print(f"   Input: {imgs.shape}")
    print(f"   Messages: {msgs.shape} ({cfg.nbits} bits)")
    
    with torch.no_grad():
        outputs = model(imgs, masks, msgs, is_video=False)
    
    print(f"\n   Output keys: {list(outputs.keys())}")
    print(f"   Watermarked: {outputs['imgs_w'].shape}")
    print(f"   Predictions: {outputs['preds'].shape}")
    
    # Check bit accuracy on clean images
    preds_bits = (outputs['preds'][:, 1:] > 0).float()  # Skip detection bit
    preds_bits_avg = preds_bits.mean(dim=(2, 3))  # Average spatially
    bit_accuracy = (preds_bits_avg == msgs.float()).float().mean()
    print(f"   Bit accuracy (clean): {bit_accuracy:.2%}")
    
    print("\n" + "=" * 60)
    print("✓ SETUP COMPLETE! READY FOR TRAINING")
    print("=" * 60)
    
    print("\n📌 Next steps:")
    print("   1. Prepare dataset (SA-1b images, SA-V videos)")
    print("   2. Configure optimizer (AdamW, lr=1e-5)")
    print("   3. Set up losses (BCE, MSE, Adversarial)")
    print("   4. Run training loop:")
    print("      - Epochs 0-800: Image pre-training")
    print("      - Epochs 800-1100: Video training")
    print("      - Epochs 1050+: Fine-tune extractor")
    print("   5. Evaluate rotation robustness")


if __name__ == "__main__":
    main()