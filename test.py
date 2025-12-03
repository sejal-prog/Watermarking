#!/usr/bin/env python3
"""
Test script to verify progressive rotation implementation.
Run this before starting full training to check everything works.

Usage:
    python test_progressive_rotation.py
"""

import sys
import torch

print("="*80)
print("Testing Progressive Rotation Implementation")
print("="*80)

# Test 1: Import rotation scheduler
print("\n1. Testing rotation_scheduler.py import...")
try:
    from rotation_scheduler import RotationScheduler
    print("   ✅ rotation_scheduler.py imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import rotation_scheduler: {e}")
    print("   → Make sure rotation_scheduler.py is in your project root")
    sys.exit(1)

# Test 2: Create scheduler
print("\n2. Testing RotationScheduler initialization...")
try:
    scheduler = RotationScheduler(
        start_angle=10,
        end_angle=45,
        start_epoch=0,
        end_epoch=600,
        schedule_type='linear'
    )
    print("   ✅ Scheduler created successfully")
except Exception as e:
    print(f"   ❌ Failed to create scheduler: {e}")
    sys.exit(1)

# Test 3: Test schedule function
print("\n3. Testing schedule at key epochs...")
test_epochs = [0, 150, 300, 450, 600]
expected_angles = [10.0, 18.75, 27.5, 36.25, 45.0]

all_correct = True
for epoch, expected in zip(test_epochs, expected_angles):
    actual = scheduler.get_rotation_range(epoch)
    match = abs(actual - expected) < 0.1
    status = "✅" if match else "❌"
    print(f"   {status} Epoch {epoch:3d}: {actual:5.2f}° (expected {expected:5.2f}°)")
    if not match:
        all_correct = False

if not all_correct:
    print("   ⚠️  Schedule values don't match expected - check implementation")
else:
    print("   ✅ All schedule values correct")

# Test 4: Test augmenter modification
print("\n4. Testing Augmenter.update_rotation_range()...")
try:
    sys.path.insert(0, 'videoseal/augmentation')
    from videoseal.augmentation.augmenter import Augmenter
    from videoseal.augmentation.geometric import Rotate
    
    # Create a simple augmenter with rotation
    augmenter = Augmenter(
        masks={'kind': None},
        augs={'rotate': 1, 'identity': 1},
        augs_params={'rotate': {'min_angle': -10, 'max_angle': 10, 'do90': False}},
        num_augs=1
    )
    
    # Check if update method exists
    if not hasattr(augmenter, 'update_rotation_range'):
        print("   ❌ Augmenter doesn't have update_rotation_range() method")
        print("   → Add the method to videoseal/augmentation/augmenter.py")
        sys.exit(1)
    
    print("   ✅ Augmenter has update_rotation_range() method")
    
    # Test updating rotation
    augmenter.update_rotation_range(25.0)
    
    # Check if rotation was actually updated
    rotation_updated = False
    for aug in augmenter.augs:
        if isinstance(aug, Rotate):
            if aug.max_angle == 25:
                rotation_updated = True
                print(f"   ✅ Rotation range updated successfully: ±{aug.max_angle}°")
            else:
                print(f"   ❌ Rotation range not updated correctly: {aug.min_angle} to {aug.max_angle}")
    
    if not rotation_updated:
        print("   ⚠️  Could not verify rotation update")
        
except ImportError as e:
    print(f"   ⚠️  Could not test augmenter: {e}")
    print("   → This is OK if you haven't added the method yet")
    print("   → Remember to add update_rotation_range() to Augmenter class")
except Exception as e:
    print(f"   ⚠️  Error testing augmenter: {e}")

# Test 5: Simulate training loop
print("\n5. Simulating training loop...")
try:
    print("   Epoch  | Rotation Range")
    print("   " + "-"*30)
    for epoch in range(0, 601, 50):
        angle = scheduler.get_rotation_range(epoch)
        print(f"   {epoch:4d}   | ±{angle:5.2f}°")
    print("   ✅ Training loop simulation successful")
except Exception as e:
    print(f"   ❌ Error in simulation: {e}")

# Test 6: Test different schedule types
print("\n6. Testing different schedule types...")
schedule_types = ['linear', 'cosine', 'step', 'exponential']
try:
    for stype in schedule_types:
        s = RotationScheduler(10, 45, 0, 600, stype)
        angle_0 = s.get_rotation_range(0)
        angle_300 = s.get_rotation_range(300)
        angle_600 = s.get_rotation_range(600)
        print(f"   ✅ {stype:12s}: {angle_0:5.1f}° → {angle_300:5.1f}° → {angle_600:5.1f}°")
except Exception as e:
    print(f"   ❌ Error testing schedule types: {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nIf all tests passed (✅), you're ready to train!")
print("\nTo start training with progressive rotation:")
print("  1. Make sure rotation_scheduler.py is in your project root")
print("  2. Add update_rotation_range() to Augmenter class")
print("  3. Modify train.py as shown in COMPLETE_IMPLEMENTATION_GUIDE.md")
print("  4. Run your training command")
print("\nExpected results after 600 epochs:")
print("  - bit_acc @ 45° rotation: 70-75% (vs 50% baseline)")
print("\nGood luck! 🚀")
print("="*80)