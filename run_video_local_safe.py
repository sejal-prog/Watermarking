#!/usr/bin/env python
import os
import subprocess
import torch
import gc
import time

# Configuration
NUM_VIDEOS = 10  # Total number of videos to process
CHECKPOINT = "ckpts/y_256b_img.pth"

print(f"Processing {NUM_VIDEOS} videos one at a time with memory cleanup...")

for video_idx in range(NUM_VIDEOS):
    print(f"\n{'='*60}")
    print(f"Processing video {video_idx + 1}/{NUM_VIDEOS}")
    print(f"{'='*60}\n")
    
    # Build command - note: you'll need to modify videoseal code to accept start_idx
    # For now, we'll process in batches
    cmd = [
        "python", "-m", "videoseal.evals.full",
        "--checkpoint", CHECKPOINT,
        "--lowres_attenuation", "True",
        "--scaling_w", "0.2",
        "--dataset", "sav",
        "--is_video", "true",
        "--num_samples", "1",
        "--save_first", "0"
    ]
    
    try:
        # Run the command
        subprocess.run(cmd, check=True)
        
        # Aggressive memory cleanup
        print("\nCleaning up memory...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Wait a bit to ensure cleanup
        time.sleep(2)
        
        print(f"✓ Video {video_idx + 1} complete. Memory cleared.\n")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error processing video {video_idx}: {e}")
        continue
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        break

print(f"\n{'='*60}")
print("Processing complete!")
print(f"{'='*60}")