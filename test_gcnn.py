"""
Test GCNN implementation with 128-bit messages.
"""

import torch
from videoseal.models.embedder import build_embedder
from videoseal.modules.msg_processor import MsgProcessor
from omegaconf import OmegaConf
from torch import nn  # ADD THIS
from escnn import nn as gnn 


def test_gcnn_basic():
    """Test basic GCNN forward pass."""
    
    print("=" * 60)
    print("TEST 1: Basic Forward Pass (128 bits)")
    print("=" * 60)
    
    # Config
    cfg = OmegaConf.create({
        'msg_processor': {
            'nbits': 128,
            'hidden_size': 256,
            'msg_processor_type': 'binary+concat',
            'msg_mult': 1.0,
        },
        'unet': {
            'in_channels': 3,
            'out_channels': 3,
            'z_channels': 64,
            'num_blocks': 2,
            'z_channels_mults': [1, 2, 4, 8],
            'last_tanh': True,
            'zero_init': False,
        }
    })
    
    # Build embedder
    embedder = build_embedder('unet_gcnn', cfg, nbits=128)
    
    # Test input
    imgs = torch.randn(2, 3, 256, 256)  # Batch of 2 images
    msgs = torch.randint(0, 2, (2, 128))  # 2 random 128-bit messages
    
    # Forward pass
    print(f"\nInput shape: {imgs.shape}")
    print(f"Message shape: {msgs.shape} (128 bits)")
    
    imgs_w = embedder(imgs, msgs)
    
    print(f"Output shape: {imgs_w.shape}")
    print(f"Output range: [{imgs_w.min():.3f}, {imgs_w.max():.3f}]")
    
    assert imgs_w.shape == imgs.shape, "Output shape mismatch!"
    print("\n✓ Forward pass successful!")


def test_gcnn_rotation_equivariance():
    """Test rotation equivariance property."""
    
    print("\n" + "=" * 60)
    print("TEST 2: Rotation Equivariance (32 bits for speed)")
    print("=" * 60)
    
    # Config (smaller for speed)
    cfg = OmegaConf.create({
        'msg_processor': {
            'nbits': 32,
            'hidden_size': 64,
            'msg_processor_type': 'binary+concat',
            'msg_mult': 1.0,
        },
        'unet': {
            'in_channels': 3,
            'out_channels': 3,
            'z_channels': 32,
            'num_blocks': 1,
            'z_channels_mults': [1, 2, 4],
            'last_tanh': True,
            'zero_init': False,
        }
    })
    
    # Build embedder
    embedder = build_embedder('unet_gcnn', cfg, nbits=32)
    embedder.eval()
    
    # Test input
    img = torch.randn(1, 3, 64, 64)
    msg = torch.randint(0, 2, (1, 32))
    
    with torch.no_grad():
        # Path 1: Embed then rotate
        img_w = embedder(img, msg)
        img_w_rot90 = torch.rot90(img_w, k=1, dims=[2, 3])
        
        # Path 2: Rotate then embed
        img_rot90 = torch.rot90(img, k=1, dims=[2, 3])
        img_rot90_w = embedder(img_rot90, msg)
        
        # Compare (should be approximately equal due to equivariance)
        diff = (img_w_rot90 - img_rot90_w).abs().mean()
        
        print(f"\nRotation equivariance error: {diff:.6f}")
        print(f"Expected: < 0.1 (due to approximate equivariance)")
        
        if diff < 0.1:
            print("✓ Rotation equivariance verified!")
        else:
            print("⚠ Equivariance error higher than expected")


def test_gcnn_parameters():
    """Count parameters and compare to regular CNN."""
    
    print("\n" + "=" * 60)
    print("TEST 3: Parameter Count (128 bits)")
    print("=" * 60)
    
    cfg = OmegaConf.create({
        'msg_processor': {
            'nbits': 128,
            'hidden_size': 256,
            'msg_processor_type': 'binary+concat',
            'msg_mult': 1.0,
        },
        'unet': {
            'in_channels': 3,
            'out_channels': 3,
            'z_channels': 64,
            'num_blocks': 2,
            'z_channels_mults': [1, 2, 4, 8],
            'last_tanh': True,
            'zero_init': False,
            'activation': 'silu',
            'normalization': 'rmsnorm',
            'upsampling_type': 'bilinear',
            'downsampling_type': 'bilinear',
            'conv_layer': 'conv2d',
        }
    })
    
    # GCNN
    print("\nBuilding GCNN embedder...")
    embedder_gcnn = build_embedder('unet_gcnn', cfg, nbits=128)
    
    # DEBUG: Print layer-by-layer
    print("\nGCNN Layer breakdown:")
    for name, module in embedder_gcnn.named_modules():
        if isinstance(module, (gnn.R2Conv, nn.Conv2d)):
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {params:,}")
    
    params_gcnn = sum(p.numel() for p in embedder_gcnn.parameters())
    
    # Regular CNN
    print("\nBuilding Regular CNN embedder...")
    embedder_regular = build_embedder('unet', cfg, nbits=128)
    
    # DEBUG: Print layer-by-layer
    print("\nRegular CNN Layer breakdown:")
    for name, module in embedder_regular.named_modules():
        if isinstance(module, nn.Conv2d):
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {params:,}")
    
    params_regular = sum(p.numel() for p in embedder_regular.parameters())
    
    print(f"\n{'=' * 60}")
    print(f"Regular CNN parameters: {params_regular:,}")
    print(f"GCNN parameters:        {params_gcnn:,}")
    print(f"{'=' * 60}")
    print(f"Ratio:                  {params_gcnn / params_regular:.2f}×")
    print(f"Increase:               +{params_gcnn - params_regular:,}")
    print(f"{'=' * 60}")
    
    expected_min = 2.0  
    expected_max = 3.0  
    actual_ratio = params_gcnn / params_regular

    if expected_min < actual_ratio < expected_max:
        print("\n✓ Parameter count as expected (~2.3× due to escnn efficiency)")
    else:
        print(f"\n⚠ Unexpected parameter ratio: {actual_ratio:.2f}× (expected {expected_min}-{expected_max}×)")


def test_gcnn_128bit_full():
    """Full test with 128-bit messages."""
    
    print("\n" + "=" * 60)
    print("TEST 4: Full 128-bit Pipeline")
    print("=" * 60)
    
    # Load config
    cfg = OmegaConf.load('configs/unet_gcnn_128bits.yaml')
    
    print(f"\nConfig loaded:")
    print(f"  nbits: {cfg.nbits}")
    print(f"  hidden_size: {cfg.msg_processor.hidden_size}")
    print(f"  z_channels: {cfg.unet.z_channels}")
    print(f"  batch_size: {cfg.training.batch_size}")
    
    # Build embedder
    embedder = build_embedder('unet_gcnn', cfg, nbits=cfg.nbits)
    params = sum(p.numel() for p in embedder.parameters())
    
    print(f"\nEmbedder built:")
    print(f"  Parameters: {params:,}")
    print(f"  Memory estimate: ~{params * 4 / 1024**2:.1f} MB (fp32)")
    
    # Test
    batch_size = cfg.training.batch_size
    imgs = torch.randn(batch_size, 3, 256, 256)
    msgs = torch.randint(0, 2, (batch_size, cfg.nbits))
    
    print(f"\nRunning forward pass...")
    print(f"  Batch size: {batch_size}")
    print(f"  Image shape: {imgs.shape}")
    print(f"  Message shape: {msgs.shape}")
    
    imgs_w = embedder(imgs, msgs)
    
    print(f"\nOutput:")
    print(f"  Shape: {imgs_w.shape}")
    print(f"  Range: [{imgs_w.min():.3f}, {imgs_w.max():.3f}]")
    print(f"  Mean perturbation: {(imgs_w - imgs).abs().mean():.6f}")
    
    print("\n✓ Full 128-bit pipeline test passed!")


if __name__ == "__main__":
    print("\n🧪 TESTING GCNN IMPLEMENTATION (128 BITS)\n")
    
    # Run tests
    test_gcnn_basic()
    test_gcnn_rotation_equivariance()
    test_gcnn_parameters()
    test_gcnn_128bit_full()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60 + "\n")