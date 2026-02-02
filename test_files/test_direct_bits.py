"""
Test Direct Bit Encoding - Most Invertible Approach

The problem with summing embeddings:
- 32 vectors summed into 1 vector
- Hard to invert (extract individual bits from sum)

The solution - direct channel mapping:
- Each bit directly controls one channel
- Bit 0 = channel 0 is +1 or -1
- Bit 1 = channel 1 is +1 or -1
- ...
- Extractor just needs to check sign of each channel

This is trivially invertible!

Run from videoseal root:
    python test_direct_bits.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from escnn import gspaces
from escnn import nn as enn


def get_gspace(group_type="C4"):
    if group_type == "C4":
        return gspaces.rot2dOnR2(N=4)
    elif group_type == "C8":
        return gspaces.rot2dOnR2(N=8)
    elif group_type == "D4":
        return gspaces.flipRot2dOnR2(N=4)
    else:
        raise ValueError(f"Unknown group type: {group_type}")


class DirectBitEmbedder(nn.Module):
    """
    Embedder with direct bit-to-channel mapping.
    
    Instead of learning complex embeddings, simply:
    1. Lift image to P4
    2. Process with G-Conv
    3. Add message directly as channels (bit i -> channel i = ±1)
    4. Project back to Z2
    """
    
    def __init__(self, nbits=32, base_channels=32, group_type="C4"):
        super().__init__()
        
        self.nbits = nbits
        self.gspace = get_gspace(group_type)
        self.group_order = self.gspace.fibergroup.order()
        
        # Types
        self.in_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        self.lift_type = enn.FieldType(self.gspace, base_channels * [self.gspace.regular_repr])
        self.msg_type = enn.FieldType(self.gspace, nbits * [self.gspace.regular_repr])
        self.combined_type = enn.FieldType(
            self.gspace, (base_channels + nbits) * [self.gspace.regular_repr]
        )
        self.out_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        
        # Lift
        self.lift = enn.R2Conv(self.in_type, self.lift_type, kernel_size=3, padding=1)
        
        # Process combined features
        self.process = enn.SequentialModule(
            enn.R2Conv(self.combined_type, self.combined_type, kernel_size=3, padding=1),
            enn.InnerBatchNorm(self.combined_type),
            enn.ReLU(self.combined_type),
            enn.R2Conv(self.combined_type, self.lift_type, kernel_size=3, padding=1),
            enn.InnerBatchNorm(self.lift_type),
            enn.ReLU(self.lift_type),
        )
        
        # Project to RGB
        self.project = enn.R2Conv(self.lift_type, self.out_type, kernel_size=3, padding=1)
        
        # Watermark strength
        self.register_buffer('watermark_strength', torch.tensor(0.3))
        
    def forward(self, imgs, msgs):
        """
        Args:
            imgs: [B, 3, H, W] in [0, 1]
            msgs: [B, nbits] binary
        Returns:
            Watermarked images [B, 3, H, W] in [-1, 1]
        """
        B, _, H, W = imgs.shape
        
        # Preprocess
        imgs = imgs * 2 - 1  # [0,1] -> [-1,1]
        
        # Lift
        x = enn.GeometricTensor(imgs, self.in_type)
        x = self.lift(x)  # [B, base_channels * 4, H, W]
        
        # Create message channels - DIRECT MAPPING
        # Each bit becomes ±1 in its channel
        msg_signed = msgs.float() * 2 - 1  # [B, nbits], values in {-1, +1}
        
        # Expand spatially
        msg_spatial = msg_signed.unsqueeze(-1).unsqueeze(-1)  # [B, nbits, 1, 1]
        msg_spatial = msg_spatial.expand(-1, -1, H, W)  # [B, nbits, H, W]
        
        # Replicate for group (each channel repeated 4 times for C4)
        msg_group = msg_spatial.unsqueeze(2).expand(-1, -1, self.group_order, -1, -1)
        msg_group = msg_group.reshape(B, self.nbits * self.group_order, H, W)
        
        # Concatenate with image features
        combined = torch.cat([x.tensor, msg_group], dim=1)
        x = enn.GeometricTensor(combined, self.combined_type)
        
        # Process
        x = self.process(x)
        
        # Project
        x = self.project(x)
        watermark = torch.tanh(x.tensor)
        
        # Residual
        output = imgs + self.watermark_strength * watermark
        output = torch.clamp(output, -1, 1)
        
        return output
    
    def get_random_msg(self, bsz=1):
        return torch.randint(0, 2, (bsz, self.nbits))


class DirectBitExtractor(nn.Module):
    """
    Extractor optimized for direct bit encoding.
    
    Uses G-CNN backbone with GroupPooling for invariance,
    then FC head to decode bits.
    """
    
    def __init__(self, nbits=32, dims=[64, 128, 256], group_type="C4"):
        super().__init__()
        
        self.nbits = nbits
        self.gspace = get_gspace(group_type)
        self.group_order = self.gspace.fibergroup.order()
        
        # Types
        self.in_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        
        # Stem
        first_type = enn.FieldType(self.gspace, dims[0] * [self.gspace.regular_repr])
        self.stem = enn.SequentialModule(
            enn.R2Conv(self.in_type, first_type, kernel_size=3, padding=1),
            enn.InnerBatchNorm(first_type),
            enn.ReLU(first_type),
            enn.PointwiseAvgPool(first_type, kernel_size=2, stride=2),
        )
        
        # Stages
        self.stages = nn.ModuleList()
        prev_type = first_type
        
        for i, dim in enumerate(dims):
            curr_type = enn.FieldType(self.gspace, dim * [self.gspace.regular_repr])
            
            stage = enn.SequentialModule(
                enn.R2Conv(prev_type, curr_type, kernel_size=3, padding=1),
                enn.InnerBatchNorm(curr_type),
                enn.ReLU(curr_type),
                enn.R2Conv(curr_type, curr_type, kernel_size=3, padding=1),
                enn.InnerBatchNorm(curr_type),
                enn.ReLU(curr_type),
                enn.PointwiseAvgPool(curr_type, kernel_size=2, stride=2),
            )
            self.stages.append(stage)
            prev_type = curr_type
        
        # Group pooling for invariance
        self.final_type = prev_type
        self.group_pool = enn.GroupPooling(self.final_type)
        
        # FC head - larger for 32 bits
        final_dim = dims[-1]
        self.fc = nn.Sequential(
            nn.Linear(final_dim, final_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_dim * 2, final_dim),
            nn.ReLU(inplace=True),
            nn.Linear(final_dim, nbits),
        )
        
    def forward(self, imgs):
        """
        Args:
            imgs: [B, 3, H, W] in [0, 1]
        Returns:
            Logits [B, nbits]
        """
        # Preprocess
        imgs = imgs * 2 - 1
        
        # Stem
        x = enn.GeometricTensor(imgs, self.in_type)
        x = self.stem(x)
        
        # Stages
        for stage in self.stages:
            x = stage(x)
        
        # Group pooling (invariance)
        x = self.group_pool(x)
        
        # Spatial pooling
        features = x.tensor.mean(dim=[-2, -1])
        
        # FC
        logits = self.fc(features)
        
        return logits


def test_direct_bits():
    print("=" * 70)
    print(" Test Direct Bit Encoding")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    nbits = 32
    batch_size = 16
    img_size = 64
    
    # Build models
    embedder = DirectBitEmbedder(nbits=nbits, base_channels=32).to(device)
    extractor = DirectBitExtractor(nbits=nbits, dims=[64, 128, 256]).to(device)
    
    print(f"\n  Embedder params: {sum(p.numel() for p in embedder.parameters()):,}")
    print(f"  Extractor params: {sum(p.numel() for p in extractor.parameters()):,}")
    
    # Freeze embedder first
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    
    extractor.train()
    optimizer = Adam(extractor.parameters(), lr=1e-3)
    
    # Check watermark signal
    print("\n  Checking watermark signal...")
    with torch.no_grad():
        img = torch.rand(1, 3, img_size, img_size).to(device)
        msg0 = torch.zeros(1, nbits).to(device)
        msg1 = torch.ones(1, nbits).to(device)
        
        w0 = embedder(img, msg0)
        w1 = embedder(img, msg1)
        
        diff = (w0 - w1).abs().mean().item()
        print(f"  Watermark diff (zeros vs ones): {diff:.4f}")
    
    # Training
    num_iterations = 3000
    print(f"\n  Training extractor for {num_iterations} iterations...")
    print("-" * 70)
    
    best_acc = 0
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
        msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
        
        with torch.no_grad():
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
        
        logits = extractor(imgs_w_norm)
        loss = F.binary_cross_entropy_with_logits(logits, msgs)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
        
        if accuracy > best_acc:
            best_acc = accuracy
        
        if (iteration + 1) % 300 == 0 or iteration == 0:
            print(f"  Iter {iteration+1:5d} | Loss: {loss.item():.4f} | "
                  f"Bit Acc: {accuracy*100:.1f}% | Best: {best_acc*100:.1f}%")
    
    # Evaluate
    print("\n  Final Evaluation:")
    extractor.eval()
    
    test_accs = []
    with torch.no_grad():
        for _ in range(20):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            logits = extractor(imgs_w_norm)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predicted == msgs).float().mean().item()
            test_accs.append(accuracy)
    
    final_avg = sum(test_accs) / len(test_accs)
    print(f"\n  Final Test Accuracy: {final_avg*100:.1f}%")
    print(f"  Best during training: {best_acc*100:.1f}%")
    
    # Per-bit accuracy
    print("\n  Per-Bit Accuracy:")
    bit_correct = torch.zeros(nbits)
    bit_total = torch.zeros(nbits)
    
    with torch.no_grad():
        for _ in range(100):
            imgs = torch.rand(batch_size, 3, img_size, img_size).to(device)
            msgs = torch.randint(0, 2, (batch_size, nbits)).float().to(device)
            imgs_w = embedder(imgs, msgs)
            imgs_w_norm = (imgs_w + 1) / 2
            logits = extractor(imgs_w_norm)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            correct = (predicted == msgs).float().cpu()
            bit_correct += correct.sum(dim=0)
            bit_total += batch_size
    
    bit_accuracy = bit_correct / bit_total
    print(f"  Mean: {bit_accuracy.mean()*100:.1f}%")
    print(f"  Min:  {bit_accuracy.min()*100:.1f}% (bit {bit_accuracy.argmin().item()})")
    print(f"  Max:  {bit_accuracy.max()*100:.1f}% (bit {bit_accuracy.argmax().item()})")
    
    # Rotation test
    print("\n  Rotation Invariance Test:")
    with torch.no_grad():
        img = torch.rand(1, 3, img_size, img_size).to(device)
        msg = torch.randint(0, 2, (1, nbits)).float().to(device)
        
        img_w = embedder(img, msg)
        img_w_norm = (img_w + 1) / 2
        
        for k in range(4):
            img_rotated = torch.rot90(img_w_norm, k, dims=[-2, -1])
            logits = extractor(img_rotated)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            acc = (predicted == msg).float().mean().item()
            print(f"  Rotation {k*90:3d}°: {acc*100:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    if final_avg > 0.85:
        print(f"  ✓ SUCCESS: Direct bit encoding works! {final_avg*100:.1f}%")
    elif final_avg > 0.7:
        print(f"  ○ GOOD: Improvement over sum-based. {final_avg*100:.1f}%")
    else:
        print(f"  ○ Direct bits: {final_avg*100:.1f}%")
    print("=" * 70)
    
    return final_avg


if __name__ == "__main__":
    test_direct_bits()