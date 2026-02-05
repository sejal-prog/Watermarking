"""
Test Max Pooling vs Average Pooling in Extractor

Hypothesis: Max pooling might better capture the watermark signal
because it picks the strongest activation rather than averaging.

Run from videoseal root:
    python test_max_pooling.py
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
    else:
        raise ValueError(f"Unknown group type: {group_type}")


class DirectBitEmbedder(nn.Module):
    """Same embedder as before"""
    
    def __init__(self, nbits=32, base_channels=32, group_type="C4"):
        super().__init__()
        
        self.nbits = nbits
        self.gspace = get_gspace(group_type)
        self.group_order = self.gspace.fibergroup.order()
        
        self.in_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        self.lift_type = enn.FieldType(self.gspace, base_channels * [self.gspace.regular_repr])
        self.msg_type = enn.FieldType(self.gspace, nbits * [self.gspace.regular_repr])
        self.combined_type = enn.FieldType(
            self.gspace, (base_channels + nbits) * [self.gspace.regular_repr]
        )
        self.out_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        
        self.lift = enn.R2Conv(self.in_type, self.lift_type, kernel_size=3, padding=1)
        
        self.process = enn.SequentialModule(
            enn.R2Conv(self.combined_type, self.combined_type, kernel_size=3, padding=1),
            enn.InnerBatchNorm(self.combined_type),
            enn.ReLU(self.combined_type),
            enn.R2Conv(self.combined_type, self.lift_type, kernel_size=3, padding=1),
            enn.InnerBatchNorm(self.lift_type),
            enn.ReLU(self.lift_type),
        )
        
        self.project = enn.R2Conv(self.lift_type, self.out_type, kernel_size=3, padding=1)
        self.register_buffer('watermark_strength', torch.tensor(0.3))
        
    def forward(self, imgs, msgs):
        B, _, H, W = imgs.shape
        imgs = imgs * 2 - 1
        
        x = enn.GeometricTensor(imgs, self.in_type)
        x = self.lift(x)
        
        msg_signed = msgs.float() * 2 - 1
        msg_spatial = msg_signed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        msg_group = msg_spatial.unsqueeze(2).expand(-1, -1, self.group_order, -1, -1)
        msg_group = msg_group.reshape(B, self.nbits * self.group_order, H, W)
        
        combined = torch.cat([x.tensor, msg_group], dim=1)
        x = enn.GeometricTensor(combined, self.combined_type)
        
        x = self.process(x)
        x = self.project(x)
        watermark = torch.tanh(x.tensor)
        
        output = imgs + self.watermark_strength * watermark
        output = torch.clamp(output, -1, 1)
        
        return output
    
    def get_random_msg(self, bsz=1):
        return torch.randint(0, 2, (bsz, self.nbits))


class ExtractorWithPoolingChoice(nn.Module):
    """
    Extractor with configurable spatial pooling.
    
    Args:
        pooling_type: "avg", "max", or "both" (concatenate avg and max)
    """
    
    def __init__(self, nbits=32, dims=[64, 128, 256], group_type="C4", pooling_type="avg"):
        super().__init__()
        
        self.nbits = nbits
        self.pooling_type = pooling_type
        self.gspace = get_gspace(group_type)
        self.group_order = self.gspace.fibergroup.order()
        
        self.in_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        
        first_type = enn.FieldType(self.gspace, dims[0] * [self.gspace.regular_repr])
        self.stem = enn.SequentialModule(
            enn.R2Conv(self.in_type, first_type, kernel_size=3, padding=1),
            enn.InnerBatchNorm(first_type),
            enn.ReLU(first_type),
            enn.PointwiseAvgPool(first_type, kernel_size=2, stride=2),
        )
        
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
        
        self.final_type = prev_type
        self.group_pool = enn.GroupPooling(self.final_type)
        
        # FC input size depends on pooling type
        final_dim = dims[-1]
        if pooling_type == "both":
            fc_input_dim = final_dim * 2  # avg + max concatenated
        else:
            fc_input_dim = final_dim
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, final_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_dim * 2, final_dim),
            nn.ReLU(inplace=True),
            nn.Linear(final_dim, nbits),
        )
        
    def forward(self, imgs):
        imgs = imgs * 2 - 1
        
        x = enn.GeometricTensor(imgs, self.in_type)
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        # Group pooling (for rotation invariance)
        x = self.group_pool(x)
        tensor = x.tensor  # [B, C, H, W]
        
        # Spatial pooling - THE KEY DIFFERENCE
        if self.pooling_type == "avg":
            features = tensor.mean(dim=[-2, -1])
        elif self.pooling_type == "max":
            features = tensor.amax(dim=[-2, -1])  # Max over spatial dims
        elif self.pooling_type == "both":
            avg_features = tensor.mean(dim=[-2, -1])
            max_features = tensor.amax(dim=[-2, -1])
            features = torch.cat([avg_features, max_features], dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        logits = self.fc(features)
        
        return logits


def test_pooling_type(pooling_type, embedder, device, nbits=32, batch_size=16, img_size=64, num_iterations=2000):
    """Test a specific pooling type."""
    
    extractor = ExtractorWithPoolingChoice(
        nbits=nbits, 
        dims=[64, 128, 256], 
        pooling_type=pooling_type
    ).to(device)
    
    print(f"\n  Extractor params: {sum(p.numel() for p in extractor.parameters()):,}")
    
    extractor.train()
    optimizer = Adam(extractor.parameters(), lr=1e-3)
    
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
        
        if (iteration + 1) % 400 == 0:
            print(f"    Iter {iteration+1:5d} | Loss: {loss.item():.4f} | "
                  f"Bit Acc: {accuracy*100:.1f}% | Best: {best_acc*100:.1f}%")
    
    # Evaluate
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
    
    return sum(test_accs) / len(test_accs), best_acc


def main():
    print("=" * 70)
    print(" Test: Average vs Max Pooling in Extractor")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    nbits = 32
    batch_size = 16
    img_size = 64
    num_iterations = 2000
    
    # Build embedder (shared)
    embedder = DirectBitEmbedder(nbits=nbits, base_channels=32).to(device)
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    
    print(f"\n  Embedder params: {sum(p.numel() for p in embedder.parameters()):,}")
    
    results = {}
    
    # Test 1: Average Pooling
    print("\n" + "-" * 70)
    print("  Test 1: AVERAGE Pooling (current)")
    print("-" * 70)
    avg_final, avg_best = test_pooling_type("avg", embedder, device, nbits, batch_size, img_size, num_iterations)
    results["avg"] = (avg_final, avg_best)
    print(f"\n  → Average Pooling: Final={avg_final*100:.1f}%, Best={avg_best*100:.1f}%")
    
    # Test 2: Max Pooling
    print("\n" + "-" * 70)
    print("  Test 2: MAX Pooling")
    print("-" * 70)
    max_final, max_best = test_pooling_type("max", embedder, device, nbits, batch_size, img_size, num_iterations)
    results["max"] = (max_final, max_best)
    print(f"\n  → Max Pooling: Final={max_final*100:.1f}%, Best={max_best*100:.1f}%")
    
    # Test 3: Both (concatenated)
    print("\n" + "-" * 70)
    print("  Test 3: BOTH (Avg + Max concatenated)")
    print("-" * 70)
    both_final, both_best = test_pooling_type("both", embedder, device, nbits, batch_size, img_size, num_iterations)
    results["both"] = (both_final, both_best)
    print(f"\n  → Both Pooling: Final={both_final*100:.1f}%, Best={both_best*100:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"\n  {'Pooling':<15} {'Final':<12} {'Best':<12}")
    print("-" * 40)
    for name, (final, best) in results.items():
        print(f"  {name:<15} {final*100:.1f}%{'':<6} {best*100:.1f}%")
    
    # Find best
    best_method = max(results.items(), key=lambda x: x[1][0])
    print(f"\n  → Best: {best_method[0].upper()} pooling ({best_method[1][0]*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()