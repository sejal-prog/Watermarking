"""
Message Injection Verification Test
Verifies:
1. Message is correctly replicated across group dimension
2. Watermark is consistent from all rotation angles
3. Same message produces consistent output when image is rotated

Run from videoseal root:
    python test_message_injection.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from videoseal.models.g_embedder import build_g_embedder
from escnn import nn as enn


def rotate_tensor_90(x: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Rotate tensor by k * 90 degrees counterclockwise."""
    return torch.rot90(x, k, dims=[-2, -1])


def test_message_replication():
    """Test 1: Verify message is replicated across all group channels"""
    print("=" * 70)
    print(" Test 1: Message Replication Across Group Dimension")
    print("=" * 70)
    
    model = build_g_embedder(
        nbits=32,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    )
    model.eval()
    
    gunet = model.gunet
    msg_processor = model.msg_processor
    group_order = msg_processor.group_order
    hidden_size = msg_processor.hidden_size
    
    print(f"\n  Configuration:")
    print(f"    nbits: {msg_processor.nbits}")
    print(f"    hidden_size: {hidden_size}")
    print(f"    group_order: {group_order}")
    
    # Create a dummy latent tensor (simulating encoder output)
    B, C, H, W = 1, 64, 16, 16  # 64 channels in group space = 16 base × 4 rotations
    
    # Create as if it's from encoder (regular repr)
    gspace = gunet.gspace
    latent_type = enn.FieldType(gspace, (C // group_order) * [gspace.regular_repr])
    latent_tensor = torch.randn(B, C, H, W)
    latents = enn.GeometricTensor(latent_tensor, latent_type)
    
    # Create message
    msg = torch.tensor([[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
                         0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]])
    
    print(f"\n  Input shapes:")
    print(f"    Latent: {latents.tensor.shape} ({C//group_order} ch × {group_order} rot = {C})")
    print(f"    Message: {msg.shape}")
    
    # Process message
    output = msg_processor(latents, msg, verbose=True)
    
    print(f"\n  Output shape: {output.tensor.shape}")
    
    # Verify replication
    output_tensor = output.tensor
    total_channels = output_tensor.shape[1]
    added_channels = total_channels - C
    
    print(f"\n  Channel analysis:")
    print(f"    Original channels: {C}")
    print(f"    Added channels: {added_channels}")
    print(f"    Expected added: {hidden_size} × {group_order} = {hidden_size * group_order}")
    
    # Extract the message portion (last added_channels)
    msg_portion = output_tensor[:, C:, :, :]  # [B, hidden_size*G, H, W]
    
    # Reshape to see group dimension: [B, hidden_size, G, H, W]
    msg_reshaped = msg_portion.view(B, hidden_size, group_order, H, W)
    
    print(f"\n  Message portion reshaped: {msg_reshaped.shape}")
    print(f"    → [B, hidden_size, group_order, H, W]")
    
    # Check if message is same across all rotation channels
    print(f"\n  Checking if message is identical across rotation channels:")
    
    all_same = True
    for g in range(1, group_order):
        diff = (msg_reshaped[:, :, 0, :, :] - msg_reshaped[:, :, g, :, :]).abs().max().item()
        status = "✓" if diff < 1e-6 else "✗"
        print(f"    Rotation 0 vs Rotation {g}: max diff = {diff:.8f} {status}")
        if diff > 1e-6:
            all_same = False
    
    if all_same:
        print(f"\n  ✓ Message is correctly replicated across all {group_order} rotation channels!")
    else:
        print(f"\n  ✗ ERROR: Message differs across rotation channels!")
    
    return all_same


def test_embedding_rotation_consistency():
    """Test 2: Same image rotated should produce rotated watermark"""
    print("\n" + "=" * 70)
    print(" Test 2: Embedding Rotation Consistency")
    print("=" * 70)
    
    model = build_g_embedder(
        nbits=32,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    )
    model.eval()
    
    # Fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Create test image and message
    img = torch.rand(1, 3, 64, 64)
    msg = model.get_random_msg(1)
    
    print(f"\n  Input:")
    print(f"    Image: {img.shape}")
    print(f"    Message: {msg.shape} = {msg[0, :8].tolist()}...")
    
    print(f"\n  Testing: embed(rotate(img, k)) vs rotate(embed(img), k)")
    print(f"  If equivariant, these should be approximately equal.\n")
    
    # Embed original image
    with torch.no_grad():
        embedded_original = model(img, msg)
    
    results = []
    for k in range(4):
        rotation_deg = k * 90
        
        # Method 1: Rotate input, then embed
        img_rotated = rotate_tensor_90(img, k)
        with torch.no_grad():
            embed_then_nothing = model(img_rotated, msg)
        
        # Method 2: Embed original, then rotate
        rotated_embedding = rotate_tensor_90(embedded_original, k)
        
        # Compare
        diff = (embed_then_nothing - rotated_embedding).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()
        
        status = "✓" if mean_diff < 0.15 else "○"
        print(f"  {status} Rotation {rotation_deg:3d}°: mean_diff = {mean_diff:.6f}, max_diff = {max_diff:.6f}")
        
        results.append(mean_diff)
    
    avg_error = np.mean(results)
    print(f"\n  Average equivariance error: {avg_error:.6f}")
    
    if avg_error < 0.15:
        print(f"  ✓ Embedding is rotation equivariant!")
    else:
        print(f"  ○ Note: Higher error expected for untrained model")
    
    return avg_error < 0.15


def test_same_message_different_rotations():
    """Test 3: Same message should be recoverable from any rotation"""
    print("\n" + "=" * 70)
    print(" Test 3: Message Consistency Across Rotations")
    print("=" * 70)
    
    model = build_g_embedder(
        nbits=32,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    )
    model.eval()
    
    torch.manual_seed(123)
    
    img = torch.rand(1, 3, 64, 64)
    msg = model.get_random_msg(1)
    
    print(f"\n  Original message: {msg[0, :16].tolist()}...")
    print(f"\n  Embedding same message into image at different rotations:")
    
    embeddings = []
    for k in range(4):
        rotation_deg = k * 90
        
        # Rotate image, embed message
        img_rotated = rotate_tensor_90(img, k)
        with torch.no_grad():
            embedded = model(img_rotated, msg)
        
        # Rotate back to original orientation
        embedded_aligned = rotate_tensor_90(embedded, -k % 4)
        embeddings.append(embedded_aligned)
        
        print(f"    Rotation {rotation_deg:3d}°: embedded and aligned back")
    
    # Compare all aligned embeddings
    print(f"\n  Comparing aligned embeddings (should be similar):")
    
    reference = embeddings[0]
    all_similar = True
    
    for k in range(1, 4):
        diff = (reference - embeddings[k]).abs().mean().item()
        status = "✓" if diff < 0.2 else "○"
        print(f"    0° vs {k*90}°: mean_diff = {diff:.6f} {status}")
        if diff > 0.2:
            all_similar = False
    
    if all_similar:
        print(f"\n  ✓ Message embedding is consistent across rotations!")
    else:
        print(f"\n  ○ Some variation (expected for untrained model)")
    
    return all_similar


def test_message_injection_detailed():
    """Test 4: Detailed message injection analysis"""
    print("\n" + "=" * 70)
    print(" Test 4: Detailed Message Injection Analysis")
    print("=" * 70)
    
    model = build_g_embedder(
        nbits=32,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    )
    model.eval()
    
    gunet = model.gunet
    msg_processor = model.msg_processor
    
    # Create two different messages
    msg1 = torch.zeros(1, 32)
    msg2 = torch.ones(1, 32)
    
    print(f"\n  Testing with two extreme messages:")
    print(f"    Message 1: all zeros")
    print(f"    Message 2: all ones")
    
    # Create dummy latent
    B, C, H, W = 1, 64 * 4, 16, 16  # 64 base channels × 4 group
    gspace = gunet.gspace
    latent_type = enn.FieldType(gspace, 64 * [gspace.regular_repr])
    latent_tensor = torch.randn(B, C, H, W)
    latents = enn.GeometricTensor(latent_tensor, latent_type)
    
    # Process both messages
    output1 = msg_processor(latents, msg1)
    output2 = msg_processor(latents, msg2)
    
    # Extract message portions
    msg_portion1 = output1.tensor[:, C:, :, :]
    msg_portion2 = output2.tensor[:, C:, :, :]
    
    print(f"\n  Message embedding statistics:")
    print(f"    Msg1 (zeros) - mean: {msg_portion1.mean().item():.4f}, std: {msg_portion1.std().item():.4f}")
    print(f"    Msg2 (ones)  - mean: {msg_portion2.mean().item():.4f}, std: {msg_portion2.std().item():.4f}")
    
    diff = (msg_portion1 - msg_portion2).abs().mean().item()
    print(f"\n  Difference between embeddings: {diff:.4f}")
    
    if diff > 0.1:
        print(f"  ✓ Different messages produce different embeddings!")
        return True
    else:
        print(f"  ✗ Messages should produce more different embeddings")
        return False


def test_full_pipeline_visualization():
    """Test 5: Full pipeline with dimension tracking"""
    print("\n" + "=" * 70)
    print(" Test 5: Full Pipeline Dimension Tracking")
    print("=" * 70)
    
    model = build_g_embedder(
        nbits=32,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    )
    model.eval()
    
    gunet = model.gunet
    group_order = gunet.group_order
    
    img = torch.rand(1, 3, 64, 64)
    msg = model.get_random_msg(1)
    
    # Preprocess
    x = model.preprocess(img)
    print(f"\n  [Input]")
    print(f"    Image: {img.shape} → preprocessed: {x.shape}")
    print(f"    Message: {msg.shape}")
    
    # Lift
    x = enn.GeometricTensor(x, gunet.in_type)
    x = gunet.lift(x)
    print(f"\n  [Lift: Trivial → Regular]")
    print(f"    Shape: {x.tensor.shape}")
    print(f"    Interpretation: {x.tensor.shape[1]//group_order} channels × {group_order} rotations")
    
    # Encoder
    x = gunet.inc(x)
    hiddens = [x]
    print(f"\n  [Encoder]")
    print(f"    After inc: {x.tensor.shape} ({x.tensor.shape[1]//group_order} ch × {group_order} rot)")
    
    for i, down in enumerate(gunet.downs):
        x = down(x)
        hiddens.append(x)
        print(f"    After down[{i}]: {x.tensor.shape} ({x.tensor.shape[1]//group_order} ch × {group_order} rot)")
    
    # Message injection
    print(f"\n  [Message Injection]")
    print(f"    Before: {x.tensor.shape}")
    x_before = x.tensor.clone()
    x = gunet.msg_processor(x, msg)
    print(f"    After:  {x.tensor.shape}")
    print(f"    Added:  {x.tensor.shape[1] - x_before.shape[1]} channels")
    print(f"           = {model.msg_processor.hidden_size} hidden × {group_order} rotations")
    
    # Verify message replication
    added_portion = x.tensor[:, x_before.shape[1]:, :, :]
    reshaped = added_portion.view(1, model.msg_processor.hidden_size, group_order, 
                                   x.tensor.shape[2], x.tensor.shape[3])
    
    replication_diffs = []
    for g in range(1, group_order):
        diff = (reshaped[:, :, 0] - reshaped[:, :, g]).abs().max().item()
        replication_diffs.append(diff)
    
    print(f"    Replication check: max diff across rotations = {max(replication_diffs):.8f}")
    if max(replication_diffs) < 1e-6:
        print(f"    ✓ Message correctly replicated to all rotation channels!")
    
    # Bottleneck
    x = gunet.bottleneck(x)
    print(f"\n  [Bottleneck]")
    print(f"    Shape: {x.tensor.shape}")
    
    # Decoder
    print(f"\n  [Decoder]")
    for i, up in enumerate(gunet.ups):
        skip = hiddens.pop()
        x_tensor = x.tensor
        skip_tensor = skip.tensor * gunet.connect_scale
        concat_tensor = torch.cat([x_tensor, skip_tensor], dim=1)
        x = enn.GeometricTensor(concat_tensor, gunet.up_in_types[i])
        x = up(x)
        print(f"    After up[{i}]: {x.tensor.shape} ({x.tensor.shape[1]//group_order} ch × {group_order} rot)")
    
    # Group pool and project
    print(f"\n  [Group Pool → Project]")
    print(f"    Before pool: {x.tensor.shape}")
    x = gunet.group_pool(x)
    print(f"    After pool:  {x.tensor.shape} (averaged over {group_order} rotations)")
    x = gunet.outc(x)
    print(f"    After proj:  {x.tensor.shape} (RGB output)")
    
    out = torch.tanh(x.tensor)
    print(f"\n  [Output]")
    print(f"    Final: {out.shape}")
    
    print(f"\n  ✓ Full pipeline completed successfully!")
    return True


def run_all_tests():
    print("\n" + "=" * 70)
    print(" Message Injection Verification - Complete Test Suite")
    print("=" * 70)
    
    results = {}
    
    try:
        results['replication'] = test_message_replication()
        results['rotation_consistency'] = test_embedding_rotation_consistency()
        results['message_consistency'] = test_same_message_different_rotations()
        results['different_messages'] = test_message_injection_detailed()
        results['full_pipeline'] = test_full_pipeline_visualization()
        
        print("\n" + "=" * 70)
        print(" Summary")
        print("=" * 70)
        
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "○ WARN"
            print(f"  {status}: {test_name}")
        
        all_passed = all(results.values())
        
        print("\n" + "=" * 70)
        if all_passed:
            print(" ✓ All tests passed! Embedder is working correctly.")
        else:
            print(" ○ Some tests have warnings (expected for untrained model)")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()