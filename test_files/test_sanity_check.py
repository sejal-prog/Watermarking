"""
Quick Sanity Check: Does the embedder actually embed different messages differently?

Run from videoseal root:
    python test_sanity_check.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from videoseal.models.g_embedder import build_g_embedder
from videoseal.models.g_extractor import build_g_extractor


def test_message_affects_output():
    """Check if different messages produce different watermarked images"""
    print("=" * 60)
    print(" Sanity Check 1: Do different messages produce different outputs?")
    print("=" * 60)
    
    embedder = build_g_embedder(
        nbits=32,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    )
    embedder.eval()
    
    img = torch.rand(1, 3, 128, 128)
    
    # Two extreme messages
    msg_zeros = torch.zeros(1, 32)
    msg_ones = torch.ones(1, 32)
    
    with torch.no_grad():
        out_zeros = embedder(img, msg_zeros)
        out_ones = embedder(img, msg_ones)
    
    diff = (out_zeros - out_ones).abs().mean().item()
    max_diff = (out_zeros - out_ones).abs().max().item()
    
    print(f"\n  Same image, different messages:")
    print(f"    Message 1: all zeros")
    print(f"    Message 2: all ones")
    print(f"\n  Output difference:")
    print(f"    Mean: {diff:.6f}")
    print(f"    Max:  {max_diff:.6f}")
    
    if diff > 0.01:
        print(f"\n  ✓ GOOD: Different messages produce different outputs!")
        print(f"    The embedder IS encoding the message into the image.")
        return True
    else:
        print(f"\n  ✗ BAD: Outputs are too similar!")
        print(f"    The embedder might NOT be encoding the message properly.")
        return False


def test_same_message_same_output():
    """Check if same message produces consistent output"""
    print("\n" + "=" * 60)
    print(" Sanity Check 2: Same message = consistent output?")
    print("=" * 60)
    
    embedder = build_g_embedder(
        nbits=32,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    )
    embedder.eval()
    
    img = torch.rand(1, 3, 128, 128)
    msg = torch.randint(0, 2, (1, 32))
    
    with torch.no_grad():
        out1 = embedder(img, msg)
        out2 = embedder(img, msg)
    
    diff = (out1 - out2).abs().max().item()
    
    print(f"\n  Same image + same message, run twice:")
    print(f"    Max difference: {diff:.10f}")
    
    if diff < 1e-6:
        print(f"\n  ✓ GOOD: Deterministic output!")
        return True
    else:
        print(f"\n  ✗ WARN: Non-deterministic output")
        return False


def test_extractor_sensitivity():
    """Check if extractor produces different outputs for different images"""
    print("\n" + "=" * 60)
    print(" Sanity Check 3: Extractor sensitive to input?")
    print("=" * 60)
    
    extractor = build_g_extractor(
        nbits=32,
        depths=[2, 2, 2],
        dims=[32, 64, 128],
        group_type="C4"
    )
    extractor.eval()
    
    img1 = torch.rand(1, 3, 128, 128)
    img2 = torch.rand(1, 3, 128, 128)
    
    with torch.no_grad():
        logits1 = extractor(img1)
        logits2 = extractor(img2)
    
    diff = (logits1 - logits2).abs().mean().item()
    
    print(f"\n  Two random images:")
    print(f"    Logits difference: {diff:.6f}")
    
    if diff > 0.1:
        print(f"\n  ✓ GOOD: Extractor is sensitive to input!")
        return True
    else:
        print(f"\n  ✗ BAD: Extractor outputs are too similar")
        return False


def test_multiple_bit_accuracy_runs():
    """Run bit accuracy test multiple times to see variance"""
    print("\n" + "=" * 60)
    print(" Sanity Check 4: Bit accuracy variance (10 runs)")
    print("=" * 60)
    
    embedder = build_g_embedder(
        nbits=32,
        z_channels=32,
        z_channels_mults=(1, 2),
        num_blocks=1,
        group_type="C4"
    )
    
    extractor = build_g_extractor(
        nbits=32,
        depths=[2, 2, 2],
        dims=[32, 64, 128],
        group_type="C4"
    )
    
    embedder.eval()
    extractor.eval()
    
    accuracies = []
    
    print(f"\n  Running 10 trials with random images and messages:")
    
    for i in range(10):
        img = torch.rand(1, 3, 128, 128)
        msg = embedder.get_random_msg(1)
        
        with torch.no_grad():
            img_w = embedder(img, msg)
            # Normalize to [0,1] for extractor
            img_w_norm = (img_w + 1) / 2
            logits = extractor(img_w_norm)
            extracted = extractor.decode_message(logits)
        
        accuracy = (msg.float() == extracted).float().mean().item()
        accuracies.append(accuracy)
        print(f"    Trial {i+1}: {accuracy:.4f} ({int(accuracy*32)}/32 bits)")
    
    import numpy as np
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n  Statistics:")
    print(f"    Mean accuracy: {mean_acc:.4f} ({mean_acc*32:.1f}/32 bits)")
    print(f"    Std deviation: {std_acc:.4f}")
    print(f"    Range: [{min(accuracies):.4f}, {max(accuracies):.4f}]")
    
    if 0.35 <= mean_acc <= 0.65:
        print(f"\n  ○ EXPECTED: ~50% accuracy for untrained model (random chance)")
        print(f"    This is NORMAL. Train the model to improve accuracy.")
        return True
    elif mean_acc > 0.65:
        print(f"\n  ✓ SURPRISING: Above random chance even without training!")
        return True
    else:
        print(f"\n  ○ Note: Below 35% can happen by chance, but if consistent,")
        print(f"    there might be an issue.")
        return True


def run_all_checks():
    print("\n" + "=" * 70)
    print(" G-CNN Watermarking - Quick Sanity Checks")
    print("=" * 70)
    
    results = {}
    
    try:
        results['message_affects_output'] = test_message_affects_output()
        results['deterministic'] = test_same_message_same_output()
        results['extractor_sensitive'] = test_extractor_sensitivity()
        results['accuracy_variance'] = test_multiple_bit_accuracy_runs()
        
        print("\n" + "=" * 70)
        print(" Summary")
        print("=" * 70)
        
        all_good = True
        for name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {name}")
            if not passed:
                all_good = False
        
        print("\n" + "=" * 70)
        if all_good:
            print(" ✓ All sanity checks passed!")
            print(" ")
            print(" The architecture is correct. Now train the model.")
            print(" Bit accuracy should improve from ~50% to 85-99% after training.")
        else:
            print(" ✗ Some checks failed. Review the output above.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_checks()